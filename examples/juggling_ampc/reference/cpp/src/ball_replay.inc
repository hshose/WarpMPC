// Offline recorded-data replay of the actual estimator/supervisor math. Parse
// saved JSON BEFORE timing; no SDK or robot process is created in this mode.
int replay(Options o){
 if(o.output.empty()||!fs::create_directory(o.output))throw std::runtime_error("Replay output must be new");
 if(o.calibration.empty())o.calibration=o.replay/"calibration_used.json";
 if(o.home_config.empty()&&fs::exists(o.replay/"juggling_home_used.json"))o.home_config=o.replay/"juggling_home_used.json";
 auto release=ex::read_json(o.replay/"release.json"),cfg=ex::read_json(o.replay/"configuration.json"),home=ex::read_json(o.replay/"fixture_home.json");
 const double start=release.get<double>("capture_time"),end=o.seconds==0?std::numeric_limits<double>::infinity():start+o.seconds;
 double recorded_end=start;{std::ifstream f(o.replay/"mpc.jsonl");std::string line,last;while(std::getline(f,line))if(!line.empty())last=line;if(!last.empty()){ex::pt::ptree v;std::istringstream s(last);ex::pt::read_json(s,v);recorded_end=v.get<double>("time");}}
 const double armed_at=fs::exists(o.replay/"ready.json")?ex::read_json(o.replay/"ready.json").get<double>("ready_time",start-.1):start-.1;
 o.extra_latency=cfg.get<double>("extra_latency_s");
 auto r=std::make_unique<Runtime>(o);auto alignment=ex::read_json(o.replay/"world_alignment.json");r->B_T_V=ex::pose(alignment.get_child("B_T_V"));r->V_T_B=r->B_T_V.inverse();r->have_alignment=true;
 std::vector<nt::NativePacket> states;std::vector<Frame> observations;std::string line;
 {
  std::ifstream f(o.replay/"robot/robot.jsonl");double previous=-1;
  while(std::getline(f,line)){ex::pt::ptree v;std::istringstream s(line);ex::pt::read_json(s,v);double t=v.get<double>("receive_time");if(t>end)break;if(t-previous<.0019)continue;previous=t;nt::NativePacket a;a.receive_time=t;
   auto assign=[&](const char* key,std::array<double,7>& dest){auto x=ex::vector<7>(v.get_child(key));std::copy(x.data(),x.data()+7,dest.begin());};assign("q",a.q);assign("dq",a.dq);assign("ddq_d",a.ddq_d);states.push_back(a);
  }
 }
 {
  std::ifstream f(o.replay/"vicon.jsonl");double previous_capture=0;std::uint64_t previous_frame=0;
  while(std::getline(f,line)){ex::pt::ptree v;std::istringstream s(line);ex::pt::read_json(s,v);Frame a;auto& p=a.value;p.receive_time=v.get<double>("receive_time");if(p.receive_time>end)break;p.frame=v.get<std::uint64_t>("frame");p.frame_rate_hz=v.get<double>("frame_rate_hz",100.);
   auto latency=v.get_optional<double>("latency_s");p.latency_valid=latency&&std::isfinite(*latency)&&*latency>=.001&&*latency<=.05;p.latency_s=p.latency_valid?*latency:0.;
   a.capture=v.get<double>("capture_time_s",p.latency_valid?p.receive_time-p.latency_s:(previous_frame?previous_capture+(p.frame-previous_frame)/p.frame_rate_hz:p.receive_time-.01));a.effective_latency=p.receive_time-a.capture;previous_frame=p.frame;previous_capture=a.capture;
   auto paddle=v.get_child_optional("paddle");if(paddle&&paddle->get_child_optional("position_m")){p.paddle_valid=1;auto pos=ex::vector<3>(paddle->get_child("position_m"));auto quat=ex::vector<4>(paddle->get_child("quaternion_xyzw"));ex::Pose T=ex::Pose::Identity();T.topLeftCorner<3,3>()=Eigen::Quaterniond(quat[3],quat[0],quat[1],quat[2]).toRotationMatrix();T.topRightCorner<3,1>()=pos;
    Eigen::Map<Eigen::Matrix<double,4,4,Eigen::RowMajor>>(p.paddle_T.data())=T;std::copy(pos.data(),pos.data()+3,p.paddle_position_m.begin());std::copy(quat.data(),quat.data()+4,p.paddle_quaternion_xyzw.begin());std::snprintf(p.segment.data(),p.segment.size(),"%s",paddle->get<std::string>("segment").c_str());std::snprintf(p.subject.data(),p.subject.size(),"hsh_pingpong_paddle");}
   if(auto points=v.get_child_optional("unlabeled_markers"))for(auto& point:*points){if(p.unlabeled_count==p.unlabeled.size())throw std::runtime_error("Too many replay markers");auto& marker=p.unlabeled[p.unlabeled_count++];marker.id=point.second.get<std::uint64_t>("id",0);auto pos=ex::vector<3>(point.second.get_child("position_m"));std::copy(pos.data(),pos.data()+3,marker.position_m.begin());}
   observations.push_back(a);
  }
 }
 if(states.empty()||observations.empty())throw std::runtime_error("Empty replay recordings");
 r->start_logging();Eigen::Vector3d offset=ex::vector<3>(home.get_child("face_offset_m"));r->controller.set_reference(offset,ex::vector<7>(home.get_child("q_home")));
 size_t si=0,fi=0,solves=0,reach_rejections=0,reach_in_recorded_control=0;double previous=0,next=start,first_failure_time=0;std::string failure;
 std::vector<double> update_ms,arm_ms;
 try{
  while(si<states.size()||fi<observations.size()){
   bool is_arm=si<states.size()&&(fi==observations.size()||states[si].receive_time<=observations[fi].value.receive_time);
   if(is_arm){double began=now();r->ingest_arm(states[si++],false);arm_ms.push_back(1000*(now()-began));}
   else {auto f=observations[fi++];r->pending.push_back(f);
    // Replay may feed minutes of startup data in milliseconds. Backpressure
    // here is outside the measured math and never changes live queue guards.
    while(!r->frame_logs.push(f)){if(r->io_failed)throw std::runtime_error("Replay logger failed");std::this_thread::sleep_for(std::chrono::microseconds(100));}}
   while(!r->pending.empty()){
    auto& f=r->pending.front();double capture=f.capture-o.extra_latency;
    if(f.value.receive_time>=armed_at)r->raw.allow_motion=true;
    if(capture>=start){if(!r->have_arm||r->arm.receive_time<std::max(f.value.receive_time,capture))break;
     if(!r->observer){r->observer=std::make_unique<ba::BallObserver>(r->observer_config);for(const auto& a:r->arms)r->observer->record_arm(r->geometry_sample(a));r->observer->initialize(ex::vector<3>(release.get_child("position"))-r->origin,start,ex::vector<3>(release.get_child("velocity")));r->raw.allow_motion=true;r->mpc=true;}
    }
    double began=now();r->process(f);update_ms.push_back(1000*(now()-began));r->pending.pop_front();
   }
   if(!is_arm||!r->observer||r->arm.receive_time<next)continue;
   double t=r->arm.receive_time,begin=now();MpcLog log;log.time=t;log.estimate=r->observer->estimate(t);log.estimator_ms=1000*(now()-begin);r->estimator_times.push_back(log.estimator_ms);
   if(log.estimate.stale)throw std::runtime_error("Recorded ball stale/rejected");log.state.head<6>()=log.estimate.x.head<6>();log.state.segment<7>(6)=ex::map(r->arm.q);log.state.segment<7>(13)=ex::map(r->arm.dq);log.state.tail<3>()=log.estimate.x.segment<3>(6);
   r->update_reference(log,offset,ex::vector<7>(home.get_child("q_home")));
   auto hit=ex::landing(log.state,r->gravity,g::drag,g::radius+log.reference_offset[2]);if(!ex::within_landing_reach(hit,offset.head<2>())||log.state[2]<-.15||log.state[2]>1.5){++reach_rejections;if(t<=recorded_end)++reach_in_recorded_control;if(!first_failure_time)first_failure_time=t;}
   Joint tau=r->geometry.rnea(ex::map(r->arm.q),ex::map(r->arm.dq),ex::map(r->arm.ddq_d));if(!previous){r->controller.initialize(log.state,tau);auto primal=r->controller.solution();auto dual=Vector::Zero(r->controller.dual().size()).eval();r->controller.set_warmstart(primal.data(),dual.data());}
   log.stats=r->controller.step(log.state,tau,previous?t-previous:0);log.plan=r->controller.physical_plan().leftCols(4).transpose();
   // Projection is evaluated identically; recorded joints remain the feedback.
   log.trajectory=ex::trajectory(r->geometry,log.plan,ex::map(r->arm.q),ex::map(r->arm.dq),ex::map(r->arm.ddq_d),log.raw_ratio,log.projection_ratio);
   log.source_to_send=now()-begin;log.supervisor_ms=1000*log.source_to_send-log.stats.total_ms;r->delivery_times.push_back(1000*log.source_to_send);r->solver_times.push_back(log.stats.total_ms);if(!r->mpc_logs.push(log))throw std::runtime_error("Replay MPC log overflow");++solves;previous=t;next=t+g::control_dt-.001;
  }
 }catch(const std::exception& e){failure=e.what();}
 r->close();auto maximum=[](const auto& v){return v.empty()?0:*std::max_element(v.begin(),v.end());};
 std::ostringstream s;s<<std::setprecision(17)<<"{\"physical_robot_used\":false,\"source\":"<<json_text(o.replay.string())<<",\"accepted\":"<<r->accepted<<",\"rejected\":"<<r->rejected<<",\"impacts\":"<<r->impact_times.size()<<",\"solves\":"<<solves<<",\"reach_rejections\":"<<reach_rejections<<",\"first_reach_guard_time\":"<<first_failure_time<<",\"max_marker_update_ms\":"<<maximum(update_ms)<<",\"max_arm_geometry_ms\":"<<maximum(arm_ms)<<",\"end_reason\":"<<json_text(failure)<<",\"live_guard_would_stop\":"<<(reach_rejections?"true":"false")<<",\"latency_note\":\"Offline typed-data replay; excludes file parsing and actual scheduling/transport delays. Recorded measured joints, not closed-loop execution of new commands.\"}";
 std::string result=s.str();result.pop_back();result+=",\"recorded_mpc_end\":"+std::to_string(recorded_end)+",\"reach_guards_during_recorded_control\":"+std::to_string(reach_in_recorded_control)+"}";
 ex::write(o.output/"replay.json",result);std::cout<<result<<'\n';return solves>0?0:1;
}

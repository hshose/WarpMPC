#include "runtime_description.hpp"
// All-C++ physical-ball supervisor. C++ Vicon acquisition and libfranka workers
// communicate through typed packets; JSON is archival output, never feedback.
#include "experiment_math.hpp"
#include "ball_estimator.hpp"
#include "native_transport.hpp"
#include "session_control.hpp"
#include "experiment_duration.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <deque>
#include <poll.h>
#include <spawn.h>
#include <sys/wait.h>
#include <thread>
#include <iostream>
#include <optional>
extern char** environ;
namespace {
using namespace juggling;
namespace ex=juggling::experiment;
namespace nt=juggling::native;
namespace ba=juggling::ball;
namespace fs=std::filesystem;
namespace g=juggling::generated;
using Clock=std::chrono::steady_clock;
double now(){return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();}
volatile std::sig_atomic_t interrupted=0;
void signal_handler(int){interrupted=1;}
struct Options {
 fs::path factors,robot_binary,vicon_binary,calibration,timing,output,profile,replay,home_config,gravity_config;
 std::string robot="",vicon="";
 bool execute=false,self_check=false;double seconds=0,release_timeout=180,delivery=.035,strike_offset=-.05,extra_latency=-.002;
};
struct Child {
 pid_t pid=-1;int output=-1,status=0;bool exited=false;
 void start(const std::vector<std::string>& command,const fs::path& error,bool pipe_output=false){
  int p[2]{-1,-1};if(pipe_output&&pipe2(p,O_CLOEXEC))throw std::runtime_error("Cannot create child pipe");
  posix_spawn_file_actions_t actions;posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions,0,"/dev/null",O_RDONLY,0);
  posix_spawn_file_actions_addopen(&actions,2,error.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
  if(pipe_output){posix_spawn_file_actions_adddup2(&actions,p[1],1);posix_spawn_file_actions_addclose(&actions,p[0]);posix_spawn_file_actions_addclose(&actions,p[1]);}
  else posix_spawn_file_actions_addopen(&actions,1,"/dev/null",O_WRONLY,0);
  // Keep the experiment lock in the robot worker until its controlled stop
  // completes, even if the supervisor crashes; Vicon need not retain it.
  // fd 9 belongs to the launcher's lock when present. A direct invocation
  // may instead allocate fd 9 to this pipe, which the actions above close.
  if(pipe_output&&p[0]!=9&&p[1]!=9&&::fcntl(9,F_GETFD)!=-1)posix_spawn_file_actions_addclose(&actions,9);
  std::vector<char*> argv;for(const auto& s:command)argv.push_back(const_cast<char*>(s.c_str()));argv.push_back(nullptr);
  int rc=posix_spawn(&pid,command[0].c_str(),&actions,nullptr,argv.data(),environ);posix_spawn_file_actions_destroy(&actions);
  if(pipe_output)::close(p[1]);if(rc){if(pipe_output)::close(p[0]);pid=-1;throw std::runtime_error("Cannot start "+command[0]+": "+std::strerror(rc));}output=p[0];
 }
 bool poll(){if(pid<0||exited)return exited;pid_t r=waitpid(pid,&status,WNOHANG);if(r==pid)exited=true;return exited;}
 int code()const{return WIFEXITED(status)?WEXITSTATUS(status):128+(WIFSIGNALED(status)?WTERMSIG(status):0);}
 void finish(double timeout,bool request_signal){if(pid<0)return;if(request_signal&&!poll())kill(pid,SIGTERM);double deadline=now()+timeout;while(!poll()&&now()<deadline)std::this_thread::sleep_for(std::chrono::milliseconds(2));
  if(!exited){kill(pid,SIGTERM);deadline=now()+1;while(!poll()&&now()<deadline)std::this_thread::sleep_for(std::chrono::milliseconds(2));}if(!exited){kill(pid,SIGKILL);waitpid(pid,&status,0);exited=true;}if(output>=0){::close(output);output=-1;}}
 ~Child(){finish(.1,true);}
};
struct Frame {nt::ViconPacket value;double capture=0,effective_latency=0;int timing_source=0;};
struct MpcLog {
 double time=0,source_to_compute=0,source_to_send=0,estimator_ms=0,supervisor_ms=0,raw_ratio=0,projection_ratio=0;
 State state=State::Zero();ba::Estimate estimate;
 Eigen::Vector3d reference_offset=Eigen::Vector3d::Zero();Joint reference_posture=Joint::Zero();double return_fraction=0;
 Eigen::Matrix<double,4,34> plan=Eigen::Matrix<double,4,34>::Zero();
 Eigen::Matrix<double,4,28,Eigen::RowMajor> trajectory=Eigen::Matrix<double,4,28,Eigen::RowMajor>::Zero();
 StepInfo stats;
};
struct UpdateLog{ba::UpdateInfo info;};
// Eigen fixed-size matrices are not trivially copyable, so use a bounded SPSC
// queue with explicit assignment, restricted here to one producer per queue.
template<class T,size_t N> class LogQueue{
 std::array<T,N> data_{};alignas(64)std::atomic<size_t> r_{0};alignas(64)std::atomic<size_t>w_{0};
 public:bool push(const T& x){auto w=w_.load(std::memory_order_relaxed);if(w-r_.load(std::memory_order_acquire)==N)return false;data_[w%N]=x;w_.store(w+1,std::memory_order_release);return true;}
 bool pop(T& x){auto r=r_.load(std::memory_order_relaxed);if(r==w_.load(std::memory_order_acquire))return false;x=data_[r%N];r_.store(r+1,std::memory_order_release);return true;}
};
std::string frame_json(const Frame& f){auto& v=f.value;std::ostringstream s;s<<std::setprecision(17);
 s<<"{\"receive_time\":"<<v.receive_time<<",\"frame\":"<<v.frame<<",\"latency_s\":";
 if(v.latency_valid)s<<v.latency_s;else s<<"null";
 s<<",\"frame_rate_hz\":"<<v.frame_rate_hz<<",\"capture_time_s\":"<<f.capture<<",\"effective_latency_s\":"<<f.effective_latency
  <<",\"timing_source\":"<<json_text(f.timing_source==0?"receive_minus_sdk_latency":f.timing_source==1?"frame_cadence_fallback":"receive_minus_10ms_fallback")<<",\"paddle\":";
 if(v.paddle_valid)s<<"{\"subject\":"<<json_text(v.subject.data())<<",\"segment\":"<<json_text(v.segment.data())<<",\"position_m\":"<<ex::json_vector(ex::map(v.paddle_position_m))<<",\"quaternion_xyzw\":"<<ex::json_vector(ex::map(v.paddle_quaternion_xyzw))<<'}';else s<<"null";
 s<<",\"unlabeled_markers\":[";for(unsigned i=0;i<v.unlabeled_count;++i){if(i)s<<',';s<<"{\"id\":"<<v.unlabeled[i].id<<",\"position_m\":"<<ex::json_vector(ex::map(v.unlabeled[i].position_m))<<'}';}
 s<<"],\"markers\":[";for(unsigned i=0;i<v.labeled_count;++i){if(i)s<<',';const auto& m=v.labeled[i];s<<"{\"subject\":"<<json_text(m.subject.data())<<",\"name\":"<<json_text(m.name.data())<<",\"position_m\":";
  if(m.valid)s<<ex::json_vector(ex::map(m.position_m));else s<<"null";s<<'}';}s<<"],\"physical_marker_diameter_available\":false}";return s.str();
}
std::string mpc_json(const MpcLog& l){std::ostringstream s;s<<std::setprecision(17);s<<"{\"time\":"<<l.time<<",\"state\":"<<ex::json_vector(l.state)<<",\"covariance\":"<<ex::json_matrix(l.estimate.P)<<",\"plan\":"<<ex::json_matrix(l.plan)<<",\"executed_trajectory\":"<<ex::json_matrix(l.trajectory)
 <<",\"measurement_age\":"<<l.estimate.measurement_age<<",\"last_impact\":";if(std::isfinite(l.estimate.last_impact))s<<l.estimate.last_impact;else s<<"null";
 s<<",\"ball_source\":\"vicon\",\"source_to_compute_s\":"<<l.source_to_compute<<",\"source_to_send_s\":"<<l.source_to_send<<",\"estimator_ms\":"<<l.estimator_ms<<",\"supervisor_ms\":"<<l.supervisor_ms
 <<",\"raw_acceleration_ratio\":"<<l.raw_ratio<<",\"acceleration_projection_ratio\":"<<l.projection_ratio
 <<",\"reference_offset_m\":"<<ex::json_vector(l.reference_offset)<<",\"reference_posture\":"<<ex::json_vector(l.reference_posture)<<",\"nominal_return_fraction\":"<<l.return_fraction
 <<",\"metrics\":["<<l.stats.total_ms<<','<<l.stats.shift_ms<<','<<l.stats.data_ms<<','<<l.stats.qp_ms<<','<<l.stats.merit_ms<<','<<l.stats.cost<<','<<l.stats.violation<<','<<l.stats.bound_violation<<','<<l.stats.alpha<<','<<l.stats.primal_residual<<','<<l.stats.dual_residual<<','<<l.stats.clipping<<','<<l.stats.iterations<<','<<l.stats.status<<','<<l.stats.warm_dual_nonzeros<<','<<l.stats.fallback_stage<<','<<l.stats.evaluator_threads<<','<<l.stats.sqp_iterations<<"],\"qp_primal_residual\":"<<l.stats.primal_residual<<",\"qp_dual_residual\":"<<l.stats.dual_residual<<'}';return s.str();}
struct Runtime {
 Options o;Controller controller;RobotModel geometry;ex::pt::ptree metadata,fixed;
 Eigen::Vector3d origin=ex::map(g::origin),gravity;Eigen::Matrix3d normal_rotation=Eigen::Matrix3d::Identity();
 ex::Pose mount,face_in_paddle,V_T_B=ex::Pose::Identity(),B_T_V=ex::Pose::Identity();
 ba::ObserverConfig observer_config;ba::RawBall raw;ba::ReleaseGate preliminary;std::unique_ptr<ba::ReleaseGate> gate;std::unique_ptr<ba::BallObserver> observer;
 nt::SharedRegion shared;Child robot,vicon;std::thread reader,writer;
 nt::SpscQueue<Frame,128> frames,frame_logs;LogQueue<MpcLog,512> mpc_logs;LogQueue<UpdateLog,2048> update_logs;
 std::atomic<bool> reading{false},logging{false},io_failed{false};std::string reader_error,writer_error;
 std::deque<Frame> pending;std::deque<nt::NativePacket> arms;nt::NativePacket arm{};
 bool have_arm=false,have_alignment=false,mpc=false;double last_frame=0,last_paddle=0,last_ball=0,pump_estimator_ms=0;
 std::string segment;std::vector<ex::Pair> pairs;bool collecting=false;std::vector<double> impact_times;
 std::vector<double> estimator_times,delivery_times,solver_times;std::uint64_t accepted=0,rejected=0;int move_id=0;
 double nominal_return_start=std::numeric_limits<double>::quiet_NaN();
 ex::HomeReference nominal_home;
 explicit Runtime(const Options& opt):o(opt),controller(opt.factors.string()),geometry(opt.factors.string()),metadata(ex::read_json(opt.factors/"metadata.json")),fixed(ex::read_json(opt.calibration)),raw(ba::RawBallConfig{3.,INFINITY,-INFINITY,INFINITY,.08,.001}){
  if(fixed.get<int>("schema_version")!=1||fixed.get<std::string>("source_kind")!="live_vicon_robot"||fixed.get<std::string>("flange_frame")!="panda_link8"||fixed.get<std::string>("paddle_subject")!="hsh_pingpong_paddle")throw std::runtime_error("Measured hand-eye calibration is required");
  mount=ex::pose(fixed.get_child("F_T_P"));face_in_paddle=ex::pose(fixed.get_child("P_T_CAD_face"));
  gravity={g::gravity_x,g::gravity_y,g::gravity_z};
  if(!o.gravity_config.empty()){
   auto calibration=ex::read_json(o.gravity_config);
   if(calibration.get<std::string>("frame")!="robot_base")throw std::runtime_error("Gravity calibration must use robot_base coordinates");
   gravity=ex::vector<3>(calibration.get_child("gravity_m_s2"));
   if(!gravity.allFinite()||gravity.z()>=0.)throw std::runtime_error("Gravity calibration must be finite and point down");
  }
  observer_config.gravity=gravity;observer_config.drag=g::drag;observer_config.ball_mass=g::mass;observer_config.radius=g::radius;observer_config.restitution=g::restitution;observer_config.friction=g::friction;observer_config.inertia_ratio=g::inertia_ratio;
  if(metadata.get<double>("config.quadratic_drag",metadata.get<double>("config.drag_quadratic",0.))!=0.)throw std::runtime_error("This hardware profile builder supports the validated linear-drag MPC only");
  observer_config.paddle_radius=g::paddle_radius;observer_config.contact_measurement_window=.015;observer_config.max_measurement_age=.1;observer_config.initial_velocity_std=.08;
  if(auto rv=metadata.get_child_optional("spec.geometry.paddle_normal_rotvec_deg")){Eigen::Vector3d v=ex::vector<3>(*rv)*(M_PI/180);if(v.norm()>1e-12)normal_rotation=Eigen::AngleAxisd(v.norm(),v.normalized()).toRotationMatrix();}
  nominal_home=ex::configured_home(geometry,o.home_config,normal_rotation);
  estimator_times.reserve(100000);solver_times.reserve(10000);delivery_times.reserve(10000);impact_times.reserve(1000);
 }
 ba::ArmSample geometry_sample(const nt::NativePacket& s){ba::ArmSample a;a.t=s.receive_time;Joint q=ex::map(s.q),dq=ex::map(s.dq);auto f=geometry.contact_fk(q);a.p=f[0]-origin;a.n=f[1];a.velocity=f[2]*dq;a.angular_velocity=f[3]*dq;
  Eigen::Matrix<double,6,7> J;J.topRows<3>()=f[2];J.bottomRows<3>()=f[3];a.inverse_spatial_inertia=J*geometry.mass(q).ldlt().solve(J.transpose());return a;}
 ex::Pose flange(const nt::NativePacket& s){return Eigen::Map<const ex::Pose>(s.O_T_EE.data())*Eigen::Map<const ex::Pose>(s.F_T_EE.data()).inverse();}
 void start_logging(){logging=true;writer=std::thread([this]{try{
   std::ofstream vf(o.output/"vicon.jsonl"),mf(o.output/"mpc.jsonl"),uf(o.output/"observer_updates.jsonl");Frame f;MpcLog m;UpdateLog u;double last_delivery_warning=-1.;
   for(;;){
    // Observe producer completion before draining: an empty check followed by
    // observing completion could otherwise miss a final intervening push.
    bool producers_done=!logging.load(std::memory_order_acquire);bool work=false;while(frame_logs.pop(f)){vf<<frame_json(f)<<'\n';work=true;}while(mpc_logs.pop(m)){mf<<mpc_json(m)<<'\n';work=true;
     if(m.source_to_send>o.delivery&&m.time-last_delivery_warning>=1.){
      std::cerr<<"WARNING: measured-state delivery "<<1000*m.source_to_send<<" ms exceeded "<<1000*o.delivery<<" ms budget; continuing.\n";last_delivery_warning=m.time;
     }
    }while(update_logs.pop(u)){auto& v=u.info;uf<<std::setprecision(17)<<"{\"capture_time\":"<<v.capture_time<<",\"arrival_time\":"<<v.arrival_time<<",\"accepted\":"<<(v.accepted?"true":"false")<<",\"reason\":"<<json_text(v.reason)<<",\"nis\":"<<v.nis<<",\"innovation\":"<<ex::json_vector(v.innovation)<<"}\n";work=true;}
    if(!vf||!mf||!uf)throw std::runtime_error("Recording write failed");if(!work){if(producers_done)break;std::this_thread::sleep_for(std::chrono::microseconds(200));}}
   vf.close();mf.close();uf.close();if(!vf||!mf||!uf)throw std::runtime_error("Recording flush failed");
  }catch(const std::exception& e){writer_error=e.what();io_failed=true;}});}
 void start_io(){start_logging();
  vicon.start({o.vicon_binary.string(),o.vicon,"0","hsh_pingpong_paddle","--binary"},o.output/"vicon.stderr",true);
  reading=true;reader=std::thread([this]{try{Frame f;size_t filled=0;std::uint64_t prev_frame=0;double prev_capture=0;
   while(reading){pollfd p{vicon.output,POLLIN,0};int ready=::poll(&p,1,20);if(ready<0){if(errno==EINTR)continue;throw std::runtime_error("Vicon pipe poll failed");}if(!ready)continue;
    auto n=::read(vicon.output,reinterpret_cast<char*>(&f.value)+filled,sizeof(f.value)-filled);if(n<=0){if(!reading)break;throw std::runtime_error("Vicon reader ended; inspect vicon.stderr");}filled+=n;if(filled<sizeof(f.value))continue;filled=0;nt::validate(f.value);
    const auto& v=f.value;if(!std::isfinite(v.receive_time))throw std::runtime_error("Invalid Vicon arrival time");
    double rate=std::isfinite(v.frame_rate_hz)&&v.frame_rate_hz>0?v.frame_rate_hz:100.;bool valid=v.latency_valid&&std::isfinite(v.latency_s)&&v.latency_s>=.001&&v.latency_s<=.05;
    double cadence=prev_frame&&v.frame>prev_frame?prev_capture+(v.frame-prev_frame)/rate:std::numeric_limits<double>::quiet_NaN();
    f.capture=valid?v.receive_time-v.latency_s:cadence;f.timing_source=valid?0:1;
    if(!std::isfinite(f.capture)||f.capture>v.receive_time||(prev_frame&&v.frame>prev_frame&&f.capture<=prev_capture)){if(std::isfinite(cadence)&&cadence<=v.receive_time){f.capture=cadence;f.timing_source=1;}else{f.capture=v.receive_time-.01;f.timing_source=2;}}
    f.effective_latency=v.receive_time-f.capture;prev_capture=f.capture;prev_frame=v.frame;
    if(!frame_logs.push(f)||!frames.push(f))throw std::runtime_error("Bounded Vicon/logging queue overflow");
   }
  }catch(const std::exception& e){reader_error=e.what();io_failed=true;}});
 }
 void ingest_arm(const nt::NativePacket& s,bool check=true){
  if(have_arm&&s.receive_time<=arm.receive_time)throw std::runtime_error("Robot measurement time is not increasing");
  if(check&&s.stopping)throw std::runtime_error("Native session stopped (fault "+std::to_string(s.fault)+"); inspect robot/session.json");
  if(!ex::map(s.q).allFinite()||!ex::map(s.dq).allFinite())throw std::runtime_error("Nonfinite robot feedback");arm=s;have_arm=true;arms.push_back(s);while(arms.size()>200)arms.pop_front();if(observer)observer->record_arm(geometry_sample(s));
 }
 void process(const Frame& f){double begin=now();const auto& v=f.value;double t=f.capture-o.extra_latency;
  if(v.paddle_valid){if(segment.empty())segment=v.segment.data();if(segment!=v.segment.data())throw std::runtime_error("Vicon paddle segment changed");last_paddle=v.receive_time;}
  std::optional<ba::AssociationPrediction> prediction;
  if(observer){if(t<=observer->posterior_time())return;auto estimate=observer->estimate(t);ba::AssociationPrediction p;Eigen::Matrix3d R=B_T_V.topLeftCorner<3,3>().transpose();p.x.head<3>()=R*(estimate.x.head<3>()+origin-B_T_V.topRightCorner<3,1>());p.x.tail<3>()=R*estimate.x.segment<3>(3);ba::Mat6 rotation=ba::Mat6::Zero();rotation.topLeftCorner<3,3>()=R;rotation.bottomRightCorner<3,3>()=R;p.P=rotation*estimate.P.topLeftCorner<6,6>()*rotation.transpose();p.position_std=std::abs(t-estimate.last_impact)<observer_config.contact_measurement_window?observer_config.contact_measurement_std:observer_config.position_std;prediction=p;}
  // Acquire over the fixed robot workspace, independent of the current paddle
  // pose. Keep association/tracking in Vicon coordinates after acquisition.
  std::vector<ba::Marker> points;points.reserve(v.unlabeled_count);
  for(unsigned i=0;i<v.unlabeled_count;++i){
   Eigen::Vector3d p=ex::map(v.unlabeled[i].position_m);
   if(!raw.selected()){
    if(!have_alignment)continue;
    Eigen::Vector3d base=B_T_V.topLeftCorner<3,3>()*p+B_T_V.topRightCorner<3,1>();
    Eigen::Vector2d xy=(base-origin).head<2>();
    if(!base.allFinite()||!ex::within_landing_reach(xy,xy)||base.z()<origin.z()+o.strike_offset+g::radius+1.)continue;
   }
   points.push_back({v.unlabeled[i].id,p});
  }
  std::optional<Eigen::Vector3d> face;
  if(have_alignment){Eigen::Vector3d center=origin+nominal_home.offset;center.z()=origin.z()+o.strike_offset;
   face=V_T_B.topLeftCorner<3,3>()*center+V_T_B.topRightCorner<3,1>();}
  auto selected=raw.update(v.frame,t,points,face,prediction);
  if(selected){last_ball=v.receive_time;
   if(!observer&&!gate)preliminary.update(t,*selected,v.frame);
   else{Eigen::Vector3d p=B_T_V.topLeftCorner<3,3>()**selected+B_T_V.topRightCorner<3,1>();
    if(observer){auto u=observer->update_position(t,p-origin,v.receive_time);if(u.accepted)++accepted;else ++rejected;if(!update_logs.push(UpdateLog{u}))throw std::runtime_error("Observer logging queue overflow");
     auto e=observer->estimate(t);if(std::isfinite(e.last_impact)&&(impact_times.empty()||e.last_impact>impact_times.back()+1e-7))impact_times.push_back(e.last_impact);
    }else if(gate)gate->update(t,p,v.frame);}
  }else if(!observer){if(gate)gate->missing(t);else preliminary.missing(t);}
  if(collecting&&v.paddle_valid&&!arms.empty()){
   auto a=std::min_element(arms.begin(),arms.end(),[&](const auto& x,const auto& y){return std::abs(x.receive_time-t)<std::abs(y.receive_time-t);});
   if(std::abs(a->receive_time-t)<=.015&&ex::map(a->dq).cwiseAbs().maxCoeff()<.015){ex::Pose P=Eigen::Map<const Eigen::Matrix<double,4,4,Eigen::RowMajor>>(v.paddle_T.data());pairs.push_back({flange(*a),P,t,a->receive_time});}
  }
  pump_estimator_ms+=1000*(now()-begin);
 }
 void pump(bool startup=false){pump_estimator_ms=0;if(interrupted)throw std::runtime_error("Interrupted by operator");if(io_failed)throw std::runtime_error("Native acquisition/recording failed; inspect runtime and stream logs");
  if(robot.pid>0){shared.get().heartbeat=now();if(robot.poll())throw std::runtime_error("Native robot process exited; inspect robot/session.json");nt::NativePacket s;while(shared.get().states.pop(s))ingest_arm(s);}
  Frame f;while(frames.pop(f)){if(pending.size()>=32)throw std::runtime_error("Vicon/robot synchronization queue overflow");last_frame=f.value.receive_time;pending.push_back(f);}
  while(!pending.empty()){
   auto& p=pending.front();double needed=std::max(p.capture-o.extra_latency,p.value.receive_time);
   if(observer&&needed>observer->latest_arm_time()){if(now()-p.value.receive_time>.05)throw std::runtime_error("Robot history unavailable for Vicon frame");break;}
   process(p);pending.pop_front();
  }
  double t=now();if(!startup){if(robot.pid>0&&(!have_arm||t-arm.receive_time>.05))throw std::runtime_error("Robot readback stale");if(!last_frame||t-last_frame>.1)throw std::runtime_error("Vicon stream stale");if(!mpc&&(!last_paddle||t-last_paddle>.15))throw std::runtime_error("Paddle tracking lost during setup");}
 }
 void wait(){std::this_thread::sleep_for(std::chrono::microseconds(200));}
 void send(const SessionCommand& command){if(!shared.get().commands.push(command))throw std::runtime_error("Robot command queue overflow");shared.get().heartbeat=now();}
 void collect(double seconds){pairs.clear();collecting=true;double deadline=now()+seconds;while(now()<deadline){pump();wait();}collecting=false;if(pairs.size()<30)throw std::runtime_error("Not enough settled hand-eye observations");}
 void move(const Joint& target){SessionCommand c;c.kind=SessionCommand::Move;c.id=++move_id;c.source=now();std::copy(target.data(),target.data()+7,c.goal.begin());send(c);
  double deadline=now()+65,settled=0;while(now()<deadline){pump();bool good=arm.move_id==move_id&&!arm.moving&&(ex::map(arm.q)-target).cwiseAbs().maxCoeff()<.01&&ex::map(arm.dq).cwiseAbs().maxCoeff()<.015;
   if(good){if(!settled)settled=now();if(now()-settled>=.4)return;}else settled=0;wait();}throw std::runtime_error("Robot failed to settle below ball");}
 void save_setup(){
  if(!o.gravity_config.empty())fs::copy_file(o.gravity_config,o.output/"gravity_calibration_used.json");
  ex::write(o.output/"effective_gravity.json","{\"frame\":\"robot_base\",\"gravity_m_s2\":"+ex::json_vector(gravity)+",\"policy_weights_changed\":false}");
  if(!o.home_config.empty())fs::copy_file(o.home_config,o.output/"juggling_home_used.json");
  if(fs::exists(o.factors/"runtime_values.txt"))fs::copy_file(o.factors/"runtime_values.txt",o.output/"runtime_values.txt");
  fs::copy_file(o.factors/"metadata.json",o.output/"model_metadata.json");fs::copy_file(o.factors/"model_config.hpp",o.output/"model_config.hpp");fs::copy_file(o.calibration,o.output/"calibration_used.json");fs::copy_file(o.timing,o.output/"timing_calibration_used.json");
  if(!o.profile.empty()){fs::copy_file(o.profile/"manifest.sha256",o.output/"profile_manifest.sha256");}
  std::ostringstream s;s<<std::setprecision(17)<<"{\"mode\":\"fixture\",\"runtime\":\"cpp_eigen\",\"physical_robot_used\":true,\"factors\":"<<json_text(o.factors.string())<<",\"vicon_host\":"<<json_text(o.vicon)<<",\"seconds\":"<<o.seconds<<",\"extra_latency_s\":"<<o.extra_latency<<",\"strike_plane_offset_m\":"<<o.strike_offset<<",\"rehearsal_delivery_budget_s\":"<<o.delivery<<",\"rehearsal_project_reference\":true,\"model_sha256\":"<<json_text(JUGGLING_MODEL_SHA256)<<",\"feedback_transport\":\"typed_shared_memory_and_binary_vicon\",\"logging_off_control_path\":true}";ex::write(o.output/"configuration.json",s.str());
  ex::write(o.output/"clock.json","{\"monotonic\":"+std::to_string(now())+",\"unix\":"+std::to_string(std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count())+",\"vicon_extra_latency_s\":"+std::to_string(o.extra_latency)+"}");
 }
 void verify_native(){
  Child description;description.start({o.robot_binary.string(),"--describe"},o.output/"native_description.stderr",true);std::string text;double until=now()+3;char buffer[2048];
  for(;;){pollfd p{description.output,POLLIN,0};int ready=::poll(&p,1,20);if(ready>0){auto n=::read(description.output,buffer,sizeof(buffer));if(n<=0)break;text.append(buffer,n);if(text.size()>8192)throw std::runtime_error("Oversized native description");}if(now()>until)throw std::runtime_error("Native description timeout");}
  description.finish(.1,false);if(description.code())throw std::runtime_error("Native description failed");ex::pt::ptree d;std::istringstream s(text);ex::pt::read_json(s,d);
  if(d.get<int>("protocol")!=2||d.get<std::string>("model_sha256")!=JUGGLING_MODEL_SHA256||d.get<bool>("robot_connected",true))throw std::runtime_error("Native robot executable/model does not match this MPC profile");ex::write(o.output/"native_description.json",text);
 }
 void update_reference(MpcLog& log,const Eigen::Vector3d& fixture_offset,const Joint& fixture_posture){
  if(!std::isfinite(nominal_return_start)&&std::isfinite(log.estimate.last_impact))nominal_return_start=log.estimate.last_impact;
  log.return_fraction=ex::nominal_return_fraction(log.time,nominal_return_start,g::return_to_nominal_s);
  log.reference_offset=(1.-log.return_fraction)*fixture_offset+log.return_fraction*nominal_home.offset;
  log.reference_posture=(1.-log.return_fraction)*fixture_posture+log.return_fraction*nominal_home.posture;
  controller.set_reference(log.reference_offset,log.reference_posture);
 }
 void run(){save_setup();verify_native();start_io();double deadline=now()+12;
  // Establish today's Vicon-to-base alignment while holding the current pose.
  // No move command is sent until a stationary ball has been acquired.
  while(!last_paddle||now()-last_paddle>.1){pump(true);if(now()>deadline)throw std::runtime_error("Vicon paddle acquisition failed; robot was not connected");wait();}
  if(segment!=fixed.get<std::string>("paddle_segment"))throw std::runtime_error("Tracked segment differs from saved mounting calibration");
  shared=nt::SharedRegion::create((o.output/"robot.ipc").string());shared.get().heartbeat=now();
  robot.start({o.robot_binary.string(),"--execute","--robot",o.robot,"--output",(o.output/"robot").string(),"--seconds","0","--allow-nonrealtime","--require-realtime-priority","--max-plan-delivery",std::to_string(o.delivery),"--ipc",(o.output/"robot.ipc").string()},o.output/"robot_stream.stderr");
  deadline=now()+10;while(!have_arm){pump(true);if(now()>deadline)throw std::runtime_error("Robot startup timeout");wait();}
  collect(.7);V_T_B=ex::align(mount,pairs);B_T_V=V_T_B.inverse();have_alignment=true;
  ex::write(o.output/"world_alignment.json","{\"schema_version\":1,\"acquisition_pose\":\"current_stationary_pose\",\"V_T_B\":"+ex::json_matrix(V_T_B)+",\"B_T_V\":"+ex::json_matrix(B_T_V)+",\"samples\":"+std::to_string(pairs.size())+"}");
  std::cout<<"Acquiring ball above robot workspace; holding current robot pose. Keep the ball stationary."<<std::endl;
  deadline=now()+12;int visible=0;double counted_frame=0;
  while(visible<10){pump();const double t=now();
   const bool healthy=raw.selected()&&last_frame&&t-last_frame<.1&&last_paddle&&t-last_paddle<.1&&last_ball&&t-last_ball<ba::max_ball_gap;
   if(!healthy)visible=0;else if(last_frame>counted_frame)++visible;
   counted_frame=last_frame;
   if(t>deadline)throw std::runtime_error("Stable ball acquisition above robot workspace failed; no robot move was commanded");wait();}
  auto selection=raw.selection().value();
  Eigen::Vector3d selected_base=B_T_V.topLeftCorner<3,3>()*selection.position+B_T_V.topRightCorner<3,1>();
  ex::write(o.output/"ball_selection.json","{\"position_m\":"+ex::json_vector(selection.position)+",\"position_base_m\":"+ex::json_vector(selected_base)+",\"coverage\":"+std::to_string(selection.coverage)+",\"samples\":"+std::to_string(selection.samples)+",\"scatter_p95_m\":"+std::to_string(selection.scatter_p95)+",\"eligible_candidates\":"+std::to_string(selection.eligible_candidates)+",\"method\":\"robot_workspace_nearest_nominal_xy_among_stable_95pct_3s\"}");
  std::cout<<"BALL LOCATED at robot-base "<<selected_base.transpose()<<" m. Keep holding the ball."<<std::endl;
  deadline=now()+3;while(!preliminary.anchor()){pump();if(now()>deadline)throw std::runtime_error("Ball is not stationary in fixture");wait();}
  Eigen::Vector3d anchor=B_T_V.topLeftCorner<3,3>()**preliminary.anchor()+B_T_V.topRightCorner<3,1>();
  Joint home=ex::fixture_home(geometry,anchor,o.strike_offset,normal_rotation);Eigen::Vector3d offset(anchor[0]-origin[0],anchor[1]-origin[1],o.strike_offset);
  ex::write(o.output/"fixture_home.json","{\"q_home\":"+ex::json_vector(home)+",\"ball_fixture_base_m\":"+ex::json_vector(anchor)+",\"face_offset_m\":"+ex::json_vector(offset)+",\"face_target_base_m\":"+ex::json_vector(origin+offset)+",\"release_height_supported\":true,\"free_fall_gap_m\":"+std::to_string(anchor[2]-origin[2]-offset[2]-g::radius)+"}");
  ex::write(o.output/"juggling_reference.json","{\"initial_offset_m\":"+ex::json_vector(offset)+",\"nominal_face_base_m\":"+ex::json_vector(origin+nominal_home.offset)+",\"nominal_posture\":"+ex::json_vector(nominal_home.posture)+",\"transition_seconds\":"+std::to_string(g::return_to_nominal_s)+",\"trigger\":\"first_estimated_impact\",\"transition\":\"quintic\"}");
  // Python reuses the already measured stationary fixture anchor in base axes.
  gate=std::make_unique<ba::ReleaseGate>(ba::ReleaseConfig{},gravity,anchor);
  move(home);collect(.4);for(const auto& p:pairs){auto expected=(V_T_B*p.base_flange*mount).eval();if((expected.topRightCorner<3,1>()-p.vicon_paddle.topRightCorner<3,1>()).norm()>.005||ex::angle(expected.topLeftCorner<3,3>().transpose()*p.vicon_paddle.topLeftCorner<3,3>())>M_PI/180)throw std::runtime_error("Mounting/world alignment inconsistent below fixture");}
  controller.set_reference(offset,home);State seed=State::Zero();seed.head<3>()=anchor-origin;seed.segment<7>(6)=ex::map(arm.q);seed.segment<7>(13)=ex::map(arm.dq);auto tau=geometry.rnea(ex::map(arm.q),ex::map(arm.dq),ex::map(arm.ddq_d));controller.step(seed,tau,0);pump();
  raw.allow_motion=true;gate->arm();ex::write(o.output/"ready.json","{\"release_armed\":true,\"ready_time\":"+std::to_string(now())+",\"ball_base_m\":"+ex::json_vector(anchor)+",\"logging_active\":true}");
  std::cout<<"RELEASE BALL NOW: paddle settled below ball; C++ tracking healthy; recording active."<<std::endl;
  deadline=now()+o.release_timeout;while(!gate->released()){pump();gate->missing(now()-o.extra_latency);if(now()>deadline)throw std::runtime_error("Release wait timed out");wait();}
  auto release=gate->released().value();std::ostringstream release_json;release_json<<std::setprecision(17)<<"{\"capture_time\":"<<release.capture_time<<",\"position\":"<<ex::json_vector(release.position)<<",\"velocity\":"<<ex::json_vector(release.velocity)<<",\"observed_drop_m\":"<<release.observed_drop<<",\"frame\":"<<release.frame<<'}';ex::write(o.output/"release.json",release_json.str());
  deadline=now()+.03;while(arm.receive_time<release.capture_time){pump();if(now()>deadline)throw std::runtime_error("No robot history bracketing release");wait();}
  observer=std::make_unique<ba::BallObserver>(observer_config);for(const auto& a:arms)observer->record_arm(geometry_sample(a));observer->initialize(release.position-origin,release.capture_time,release.velocity);mpc=true;
  double start=now(),previous=0,next=0;while(!experiment_duration_expired(now()-start,o.seconds)){pump();if(now()-last_ball>ba::max_ball_gap)throw std::runtime_error("Ball tracking lost during MPC");double t=arm.receive_time;if(t<next){wait();continue;}
   double begin=now();MpcLog log;log.time=t;log.source_to_compute=begin-t;log.estimate=observer->estimate(t);log.estimator_ms=pump_estimator_ms+1000*(now()-begin);estimator_times.push_back(log.estimator_ms);
   if(log.estimate.stale)throw std::runtime_error("Ball observer measurements stale or rejected");log.state.head<6>()=log.estimate.x.head<6>();log.state.segment<7>(6)=ex::map(arm.q);log.state.segment<7>(13)=ex::map(arm.dq);log.state.tail<3>()=log.estimate.x.segment<3>(6);
   update_reference(log,offset,home);
   if(log.state[2]<-.15||log.state[2]>1.5)throw std::runtime_error("Ball left MPC workspace");if(log.state[2]>g::radius+log.reference_offset[2]){auto hit=ex::landing(log.state,gravity,g::drag,g::radius+log.reference_offset[2]);if(!ex::within_landing_reach(hit,offset.head<2>()))throw std::runtime_error("Predicted ball landing is out of reach");}
   tau=geometry.rnea(ex::map(arm.q),ex::map(arm.dq),ex::map(arm.ddq_d));if(!previous){controller.initialize(log.state,tau);auto primal=controller.solution();auto dual=Vector::Zero(controller.dual().size()).eval();controller.set_warmstart(primal.data(),dual.data());}
   log.stats=controller.step(log.state,tau,previous?t-previous:0);log.plan=controller.physical_plan().leftCols(4).transpose();
   log.trajectory=ex::trajectory(geometry,log.plan,ex::map(arm.q),ex::map(arm.dq),ex::map(arm.ddq_d),log.raw_ratio,log.projection_ratio);
   log.source_to_send=now()-t;log.supervisor_ms=1000*(now()-begin)-log.stats.total_ms;
   SessionCommand command;command.kind=SessionCommand::Plan;command.source=t;std::copy(log.trajectory.data(),log.trajectory.data()+112,command.trajectory.begin());send(command);
   if(!mpc_logs.push(log))throw std::runtime_error("MPC recording queue overflow");delivery_times.push_back(1000*log.source_to_send);solver_times.push_back(log.stats.total_ms);previous=t;next=t+g::control_dt-.001;
  }
  std::cout<<"JUGGLING COMPLETE: stopping robot."<<std::endl;
 }
 void drain_shutdown_feedback(){
  // Acquisition and archival producers remain active throughout controlled
  // stopping. Their bounded feedback queues must still have a consumer.
  Frame frame;while(frames.pop(frame)){}
  if(robot.pid>0){nt::NativePacket packet;while(shared.get().states.pop(packet)){};}
 }
 void close(){if(robot.pid>0){shared.get().stop=1;const double until=now()+4;
   while(!robot.poll()&&now()<until){drain_shutdown_feedback();std::this_thread::sleep_for(std::chrono::milliseconds(2));}
   robot.finish(0,false);
  }
  // Preserve the final rebound/edge failure after the controller stops.
  if(reader.joinable()){double until=now()+.75;while(now()<until&&!io_failed){drain_shutdown_feedback();std::this_thread::sleep_for(std::chrono::milliseconds(2));}reading=false;reader.join();}vicon.finish(1,true);logging=false;if(writer.joinable())writer.join();
  if(!fs::exists(o.output))return;
  std::ostringstream s;s<<std::setprecision(17)<<"{\"updates_accepted\":"<<accepted<<",\"updates_rejected\":"<<rejected<<",\"updates_file\":\"observer_updates.jsonl\",\"impacts\":[";for(size_t k=0;k<impact_times.size();++k){if(k)s<<',';s<<impact_times[k];}s<<"]}";ex::write(o.output/"observer.json",s.str());
  auto summary=[](std::vector<double> a){if(a.empty())return std::string("null");std::sort(a.begin(),a.end());std::ostringstream s;s<<std::setprecision(10)<<"{\"count\":"<<a.size()<<",\"median_ms\":"<<a[a.size()/2]<<",\"mean_ms\":"<<std::accumulate(a.begin(),a.end(),0.)/a.size()<<",\"p95_ms\":"<<a[size_t(.95*(a.size()-1))]<<",\"max_ms\":"<<a.back()<<'}';return s.str();};
  ex::write(o.output/"runtime_timing.json","{\"estimator\":"+summary(estimator_times)+",\"mpc\":"+summary(solver_times)+",\"state_to_send\":"+summary(delivery_times)+",\"reader_error\":"+json_text(reader_error)+",\"writer_error\":"+json_text(writer_error)+"}");
  if(robot.pid>0&&robot.code()!=0)throw std::runtime_error("Native robot session reported a fault; inspect robot/session.json");if(io_failed)throw std::runtime_error("Acquisition or recording failed: "+reader_error+" "+writer_error);
 }
 ~Runtime(){if(robot.pid>0)shared.get().stop=1;reading=false;if(reader.joinable())reader.join();robot.finish(4,false);vicon.finish(.5,true);logging=false;if(writer.joinable())writer.join();}
};
void self_check(const Options& o){RobotModel model(o.factors.string());auto m=ex::read_json(o.factors/"metadata.json");Eigen::Matrix3d R=Eigen::Matrix3d::Identity();if(auto rv=m.get_child_optional("spec.geometry.paddle_normal_rotvec_deg")){Eigen::Vector3d v=ex::vector<3>(*rv)*(M_PI/180);if(v.norm()>1e-12)R=Eigen::AngleAxisd(v.norm(),v.normalized()).toRotationMatrix();}
 if(experiment_duration_expired(121,0)||experiment_duration_expired(601,0)||experiment_duration_expired(1e9,0)||experiment_duration_expired(119,120)||!experiment_duration_expired(120,120))throw std::runtime_error("Experiment duration semantics failed");
 const double front=g::fixture_landing_forward>0?std::min(g::landing_forward,g::fixture_landing_forward):g::landing_forward;
 const double side=g::fixture_landing_forward>0?std::min(g::landing_side,g::fixture_landing_side):g::landing_side;
 if(!ex::within_landing_reach({front-.001,0},{0,0})||!ex::within_landing_reach({0,side-.001},{0,0})||ex::within_landing_reach({-.351,0},{0,0}))throw std::runtime_error("Directional landing reach failed");
 if(ex::nominal_return_fraction(1,INFINITY,5)!=0||ex::nominal_return_fraction(1,1,5)!=0||ex::nominal_return_fraction(3.5,1,5)!=.5||ex::nominal_return_fraction(6,1,5)!=1)throw std::runtime_error("Nominal return transition failed");
 Eigen::Vector3d ball=ex::map(g::origin)+Eigen::Vector3d(.01,-.01,1.1);auto q=ex::fixture_home(model,ball,-.05,R);auto f=model.contact_fk(q);
 State x=State::Zero();x.head<3>()=ball-ex::map(g::origin);x.segment<7>(6)=q;Controller c(o.factors.string());c.set_reference(Eigen::Vector3d(.01,-.01,-.05),q);Joint zero=Joint::Zero(),tau=model.rnea(q,zero,zero);auto stats=c.step(x,tau,0);
 const auto nominal=ex::configured_home(model,o.home_config,R);auto nominal_fk=model.contact_fk(nominal.posture);
 if((nominal_fk[0]-ex::map(g::origin)-nominal.offset).norm()>.001)throw std::runtime_error("Configured home IK position error");
 const auto warm=c.solution().eval();c.set_reference(nominal.offset,nominal.posture);if((c.solution()-warm).norm()!=0)throw std::runtime_error("Moving reference reset warm start");stats=c.step(x,tau,0);
 if(!c.physical_plan().allFinite())throw std::runtime_error("Nonfinite MPC plan");
 // Warm-up uses a held ball and never executes its plan. Check the projection
 // separately on a known feasible static reference; recorded falling-ball
 // replays exercise the actual MPC plans and the unchanged 1.25 guard.
 Matrix Y=Matrix::Zero(4,34);for(int k=0;k<4;++k){Y.row(k).segment<7>(6)=q;Y.row(k).segment<7>(23)=tau;}
 double a,p;auto T=ex::trajectory(model,Y,q,zero,zero,a,p);
 if(!T.allFinite()||(T.row(3).head<7>().transpose()-q).norm()>1e-8)throw std::runtime_error("Invalid static reference projection");std::cout<<"{\"robot_connected\":false,\"fixture_ik_passed\":true,\"static_projection_passed\":true,\"mpc_warmup_finite\":true,\"mpc_ms\":"<<stats.total_ms<<",\"nominal_home_base_m\":"<<ex::json_vector(ex::map(g::origin)+nominal.offset)<<",\"nominal_home_posture\":"<<ex::json_vector(nominal.posture)<<",\"normal_z\":"<<f[1](2)<<"}\n";
}
#include "ball_replay.inc"
}
int main(int argc,char** argv){Options o;try{
 if(argc==2&&std::string(argv[1])=="--describe"){std::cout<<"{\"runtime\":\"cpp_eigen\",\"robot_connected\":false,\"protocol\":1,\"model_sha256\":\""<<JUGGLING_MODEL_SHA256<<"\",\"callback_guard_s\":0.010,\"plan_delivery_budget_s\":0.035,\"ball_gap_s\":0.080,\"default_duration_s\":0,\"return_to_nominal_s\":"<<g::return_to_nominal_s<<",\"runtime_tuning\":"<<juggling::runtime_tuning_description()<<"}\n";return 0;}
 for(int i=1;i<argc;++i){std::string a=argv[i];auto value=[&](){if(i+1>=argc)throw std::runtime_error("Missing option value");return std::string(argv[++i]);};
  if(a=="--execute")o.execute=true;else if(a=="--self-check")o.self_check=true;else if(a=="--replay")o.replay=value();else if(a=="--factors")o.factors=value();else if(a=="--robot-binary")o.robot_binary=value();else if(a=="--vicon-binary")o.vicon_binary=value();else if(a=="--calibration")o.calibration=value();else if(a=="--timing-calibration")o.timing=value();else if(a=="--output")o.output=value();else if(a=="--profile-dir")o.profile=value();else if(a=="--home-config")o.home_config=value();else if(a=="--gravity-config")o.gravity_config=value();else if(a=="--seconds")o.seconds=std::stod(value());else if(a=="--release-timeout")o.release_timeout=std::stod(value());else if(a=="--robot")o.robot=value();else if(a=="--vicon-host")o.vicon=value();else throw std::runtime_error("Unknown option "+a);
 }
 if(o.self_check){self_check(o);return 0;}if(!o.replay.empty()){if(o.execute)throw std::runtime_error("Replay cannot execute hardware");return replay(o);}if(!o.execute){std::cout<<"No connection. Use cpp/scripts/run_ball_cpp.sh for a physical experiment.\n";return 0;}
 if(o.robot.empty()||o.vicon.empty())throw std::runtime_error("Pass --robot and --vicon-host explicitly");
 if(o.factors.empty()||o.robot_binary.empty()||o.vicon_binary.empty()||o.calibration.empty()||o.output.empty())throw std::runtime_error("Missing runtime path argument");
 if(!std::isfinite(o.seconds)||o.seconds<0||!std::isfinite(o.release_timeout)||o.release_timeout<=0||o.release_timeout>300)throw std::runtime_error("Invalid experiment duration (0 means no time limit)");
 auto timing=ex::read_json(o.timing);if(timing.get<std::string>("vicon_host")!=o.vicon)throw std::runtime_error("Timing calibration belongs to another Vicon server");o.extra_latency=timing.get<double>("extra_latency_s");if(!std::isfinite(o.extra_latency)||o.extra_latency<-.01||o.extra_latency>=.05)throw std::runtime_error("Invalid timing correction");
 if(!fs::create_directory(o.output))throw std::runtime_error("Output directory must be new");
 std::signal(SIGINT,signal_handler);std::signal(SIGTERM,signal_handler);std::signal(SIGHUP,signal_handler);std::signal(SIGPIPE,SIG_IGN);
 auto runtime=std::make_unique<Runtime>(o);std::string failure;try{runtime->run();}catch(const std::exception& e){failure=e.what();std::cerr<<"ABORT: "<<failure<<". Stopping robot; do not release."<<std::endl;}
 try{runtime->close();}catch(const std::exception& e){if(failure.empty())failure=e.what();else failure+="; "+std::string(e.what());}
 if(!failure.empty()){ex::write(o.output/"failure.json","{\"error\":"+json_text(failure)+",\"stop_requested\":true}");std::cerr<<failure<<std::endl;return 1;}
 return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}

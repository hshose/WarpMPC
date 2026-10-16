// Prepared physical-arm / simulated-ball harness. No robot is connected unless
// --execute or --preflight-only and all required inputs are supplied explicitly.
#include "hil_c_api.h"
#include "motion_guard.hpp"
#include "joint_homing.hpp"
#include "joint_velocity_tracking.hpp"
#include "json_text.hpp"
#include <franka/robot.h>
#include <franka/model.h>
#include <franka/exception.h>
#include <Eigen/Dense>
#include <array>
#include <cmath>
#include "model_config.hpp"
#include <algorithm>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <pthread.h>
#include <string>
#include <system_error>
#include <vector>

namespace {
volatile std::sig_atomic_t stop_requested=0;
void request_stop(int){stop_requested=1;}
struct Sample {
 double t=0,dt=0,callback_us=0,age=0,success_rate=0;int fault=0,stopping=0;
 std::array<double,7> q{},dq{},q_ref{},dq_ref{},ddq_ref{},mpc_total{},command{},tau_desired{},tau_measured{};
 std::array<double,7> velocity_command{},torque_monitored{},torque_rate_monitored{},acceleration_desired{};
};
struct HomingSample {
 double t=0,dt=0,callback_us=0,success_rate=0;
 std::array<double,7> q{},dq{},q_ref{},dq_ref{},ddq_ref{},q_desired{};
};
struct HomingRecord {
 std::vector<HomingSample> samples;size_t count=0;double duration=0,elapsed=0;bool entered=false,completed=false;
};
void save_homing(const std::string& output,const HomingRecord& home){
 std::ofstream f(output+"/homing.csv");f<<std::setprecision(17)<<"time,dt,callback_us,success_rate";
 for(const char* name:{"q","dq","q_ref","dq_ref","ddq_ref","q_desired"})for(int j=0;j<7;++j)f<<','<<name<<j;
 f<<'\n';for(size_t i=0;i<home.count;++i){const auto& s=home.samples[i];f<<s.t<<','<<s.dt<<','<<s.callback_us<<','<<s.success_rate;
  for(const auto* values:{&s.q,&s.dq,&s.q_ref,&s.dq_ref,&s.ddq_ref,&s.q_desired})for(double v:*values)f<<','<<v;
  f<<'\n';}
}
void move_home(franka::Robot& robot,HomingRecord& home){
 namespace g=juggling::generated;
 home.samples.resize(63000); // Initialized/touched before entering the callback.
 std::optional<juggling::JointHoming> path;double settled=0;
 home.entered=true;
 robot.control([&](const franka::RobotState& state,franka::Duration period)->franka::JointPositions{
  auto entered=std::chrono::steady_clock::now();double dt=period.toSec();
  if(stop_requested)throw std::runtime_error("Homing interrupted; SDK cancels the motion, MPC will not start");
  if(dt<0||dt>.005||home.count>=home.samples.size())throw std::runtime_error("Homing clock/log limit exceeded; MPC will not start");
  if(!path){path.emplace(state.q_d,g::home,g::joint_lower,g::joint_upper);home.duration=path->duration();}
  else home.elapsed+=dt;
  auto ref=path->at(home.elapsed);double error=0,velocity=0;
  for(int j=0;j<7;++j){
   if(!std::isfinite(state.q[j])||!std::isfinite(state.dq[j])||state.q[j]<g::joint_lower[j]||state.q[j]>g::joint_upper[j])throw std::runtime_error("Invalid joint state during homing");
   if(std::abs(state.q[j]-ref.q[j])>.1)throw std::runtime_error("Homing tracking error exceeded 0.1 rad; MPC will not start");
   error=std::max(error,std::abs(state.q[j]-g::home[j]));velocity=std::max(velocity,std::abs(state.dq[j]));
  }
  if(home.elapsed>=home.duration&&error<.01&&velocity<.015)settled+=dt;else settled=0;
  if(home.elapsed>home.duration+2.)throw std::runtime_error("Robot did not settle after homing; MPC will not start");
  auto& s=home.samples[home.count++];s.t=home.elapsed;s.dt=dt;s.q=state.q;s.dq=state.dq;s.q_desired=state.q_d;
  s.q_ref=ref.q;s.dq_ref=ref.dq;s.ddq_ref=ref.ddq;s.success_rate=state.control_command_success_rate;
  s.callback_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-entered).count();
  franka::JointPositions command(ref.q);
  if(settled>=.25)return franka::MotionFinished(command);
  return command;
 },franka::ControllerMode::kJointImpedance,true,100.0);
 home.completed=true;
}
void save_samples(const std::string& output,const std::vector<Sample>& samples,size_t count){
 std::ofstream f(output+"/hardware_tracking.csv");f<<std::setprecision(17)<<"time,dt,callback_us,command_age,success_rate,fault,stopping";
 for(const char* name:{"q","dq","q_ref","dq_ref","ddq_ref","mpc_total","command","tau_desired","tau_measured","velocity_command","torque_monitored","torque_rate_monitored","acceleration_desired"})for(int j=0;j<7;++j)f<<','<<name<<j;
 f<<'\n';for(size_t i=0;i<count;++i){const auto& s=samples[i];f<<s.t<<','<<s.dt<<','<<s.callback_us<<','<<s.age<<','<<s.success_rate<<','<<s.fault<<','<<s.stopping;
  for(const auto* values:{&s.q,&s.dq,&s.q_ref,&s.dq_ref,&s.ddq_ref,&s.mpc_total,&s.command,&s.tau_desired,&s.tau_measured,&s.velocity_command,&s.torque_monitored,&s.torque_rate_monitored,&s.acceleration_desired})for(double v:*values)f<<','<<v;
  f<<'\n';}
}
}

int main(int argc,char** argv){
 std::string ip,factors,payload,output;double duration=20.;bool execute=false,preflight_only=false,allow_nonrealtime=false,use_configured_load=false,home_first=false,home_only=false;
 for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--execute")execute=true;
  else if(a=="--allow-nonrealtime")allow_nonrealtime=true;
  else if(a=="--preflight-only")preflight_only=true;
  else if(a=="--use-configured-load")use_configured_load=true;
  else if(a=="--home-first")home_first=true;
  else if(a=="--home-only")home_first=home_only=true;
  else if(i+1<argc&&a=="--robot")ip=argv[++i];else if(i+1<argc&&a=="--factors")factors=argv[++i];
  else if(i+1<argc&&a=="--payload")payload=argv[++i];else if(i+1<argc&&a=="--output")output=argv[++i];
  else if(i+1<argc&&a=="--seconds")duration=std::stod(argv[++i]);else{std::cerr<<"Unknown/incomplete option: "<<a<<'\n';return 2;}}
 if((!execute&&!preflight_only)||ip.empty()||factors.empty()||(!use_configured_load&&payload.empty())||output.empty()){
  std::cout<<"Prepared HIL harness; no robot connection made.\nUsage: juggling_franka (--execute | --preflight-only) --robot HOST --factors DIR (--payload VERIFIED_PAYLOAD.txt | --use-configured-load) --output RUN_DIR [--seconds 20] [--allow-nonrealtime] [--home-first | --home-only]\nPayload: mass, flange-frame COM xyz, then 3x3 column-major COM inertia (13 SI numbers). Start stationary. --home-first moves slowly to HOME, settles, then starts MPC; --home-only stops after homing. Without these options, start near HOME. --allow-nonrealtime selects libfranka RealtimeConfig::kIgnore; reflexes and motion guards stay enabled. --preflight-only reads state and never changes configuration or starts control.\n";return execute||preflight_only?2:0;}
 namespace g=juggling::generated;
 if(execute&&preflight_only){std::cerr<<"Select execution or read-only preflight\n";return 2;}
 if(use_configured_load&&!payload.empty()){std::cerr<<"Select exactly one load source\n";return 2;}
 if(!std::isfinite(duration)||duration<=0||duration>60){std::cerr<<"Duration must be in (0,60] seconds\n";return 2;}
 int realtime_kernel=0;std::ifstream("/sys/kernel/realtime")>>realtime_kernel;
 if(!allow_nonrealtime&&realtime_kernel!=1){std::cerr<<"Realtime kernel unavailable. No robot connection or configuration change made. Explicit --allow-nonrealtime selects the SDK's non-realtime mode.\n";return 3;}
 double mass=0;std::array<double,3> com{};std::array<double,9> inertia{};
 if(!use_configured_load){std::ifstream file(payload);
  if(!(file>>mass)){std::cerr<<"Cannot read payload mass\n";return 2;}for(double& v:com)if(!(file>>v))return 2;for(double& v:inertia)if(!(file>>v))return 2;
  Eigen::Map<const Eigen::Matrix3d> I(inertia.data());Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(I);
  if(!std::isfinite(mass)||mass<=0||mass>3||!I.allFinite()||(I-I.transpose()).norm()>1e-10||eigen.eigenvalues().minCoeff()<=0||!std::all_of(com.begin(),com.end(),[](double x){return std::isfinite(x)&&std::abs(x)<.5;})){std::cerr<<"Invalid payload data\n";return 2;}}
 if(!std::filesystem::create_directory(output)){std::cerr<<"Output directory must be new (create its parent first)\n";return 2;}
 char error[4096]{};void* raw=nullptr;
 // Size/value-initialize before control: no allocation or first-touch logging pages in callback.
 std::vector<Sample> samples(static_cast<size_t>((duration+2)*1000));size_t count=0;
 double time=0,stopping=0;int fault=0;bool control_entered=false,control_completed=false;
 HomingRecord home;
 std::string failure;
 auto save=[&]{save_samples(output,samples,count);if(home_first)save_homing(output,home);std::ofstream summary(output+"/hardware_session.json");summary<<std::boolalpha<<std::setprecision(17)
  <<"{\"physical_robot_used\":true,\"physical_ball_used\":false,\"control_entered\":"<<control_entered<<",\"control_completed\":"<<control_completed
  <<",\"motion_commanded\":"<<(count>0||home.count>0)<<",\"fault\":"<<fault<<",\"elapsed_s\":"<<time<<",\"requested_duration_s\":"<<duration
  <<",\"samples\":"<<count<<",\"realtime_enforced\":"<<!allow_nonrealtime<<",\"realtime_kernel\":"<<(realtime_kernel==1)
  <<",\"preflight_only\":"<<preflight_only
  <<",\"home_first\":"<<home_first<<",\"home_only\":"<<home_only<<",\"homing_entered\":"<<home.entered<<",\"homing_completed\":"<<home.completed
  <<",\"homing_duration_s\":"<<home.duration<<",\"homing_elapsed_s\":"<<home.elapsed<<",\"homing_samples\":"<<home.count
  <<",\"preserved_configured_load\":"<<use_configured_load<<",\"signal_stop\":"<<(stop_requested!=0)
#ifdef JUGGLING_JOINT_VELOCITY
  <<",\"execution_mode\":\"sdk_joint_velocity_impedance\""
  <<",\"mpc_torque_feedback\":\"model_rnea_from_sdk_reference_acceleration\",\"torque_monitor_filter_s\":0.005"
#else
  <<",\"execution_mode\":\"computed_torque\""
#endif
  <<",\"fci_filter_hz\":100,\"rate_limiting\":true,\"error\":"<<juggling::json_text(failure)<<"}\n";};
 try{
  franka::Robot robot(ip,allow_nonrealtime?franka::RealtimeConfig::kIgnore:franka::RealtimeConfig::kEnforce);auto before=robot.readOnce();
  std::ofstream(output+"/robot_before.json")<<std::setprecision(17)<<before<<'\n';
  if(before.robot_mode!=franka::RobotMode::kIdle)throw std::runtime_error("Robot must be idle before starting");
  if(!use_configured_load&&std::abs(before.m_ee)>1e-6)throw std::runtime_error("An end effector is already configured. Use --use-configured-load to avoid counting the assembly twice.");
  for(int j=0;j<7;++j)if(!std::isfinite(before.q[j])||!std::isfinite(before.dq[j])||std::abs(before.dq[j])>.03)throw std::runtime_error("Robot must be stationary before starting");
  if(home_first){juggling::JointHoming planned(before.q_d,g::home,g::joint_lower,g::joint_upper);home.duration=planned.duration();}
  else for(int j=0;j<7;++j)if(std::abs(before.q[j]-g::home[j])>.08)throw std::runtime_error("Robot must be near HOME or use --home-first. No motion started.");
  if(preflight_only){save();std::cout<<"Read-only preflight passed; no load or motion commands sent.\n";return 0;}
  // libfranka 0.9.2 setLoad takes flange-frame COM, independently of the EE
  // transform. The normal configured-end-effector path needs no setters.
  if(!use_configured_load)robot.setLoad(mass,com,inertia);
  std::signal(SIGINT,request_stop);std::signal(SIGTERM,request_stop);std::signal(SIGHUP,request_stop);
  if(home_first){
   std::cout<<"Moving to HOME using joint impedance: planned "<<home.duration<<" s, caps 0.15 rad/s, 0.25 rad/s^2, 1 rad/s^3.\n"<<std::flush;
   int policy=0;sched_param scheduling{};
   int scheduling_error=pthread_getschedparam(pthread_self(),&policy,&scheduling);
   if(scheduling_error)throw std::system_error(scheduling_error,std::generic_category(),"Read pre-homing scheduler");
   move_home(robot,home);
   // SDK control can raise this thread to FIFO. Restore its previous policy
   // before creating MPC workers, so they do not inherit callback priority.
   scheduling_error=pthread_setschedparam(pthread_self(),policy,&scheduling);
   if(scheduling_error)throw std::system_error(scheduling_error,std::generic_category(),"Restore scheduler after homing");
   std::ofstream(output+"/robot_after_homing.json")<<std::setprecision(17)<<robot.readOnce()<<'\n';
   if(home_only){save();std::cout<<"Homing completed; MPC was not started.\n";return 0;}
  }
  if(stop_requested)throw std::runtime_error("Stop requested before MPC start");
  auto model=robot.loadModel();auto initial=robot.readOnce();auto gravity=model.gravity(initial);std::array<double,7> total;
  for(int j=0;j<7;++j)if(std::abs(initial.q[j]-g::home[j])>.08||std::abs(initial.dq[j])>.03)throw std::runtime_error("Arm moved away from its settled starting pose; MPC not started");
  std::ofstream(output+"/robot_initial.json")<<std::setprecision(17)<<initial<<'\n';
  for(int j=0;j<7;++j)total[j]=initial.tau_J_d[j]+gravity[j];
#ifdef JUGGLING_JOINT_VELOCITY
  raw=juggling_hil_create_acceleration(factors.c_str(),initial.q.data(),initial.dq.data(),initial.ddq_d.data(),.06,error,sizeof(error));
#else
  raw=juggling_hil_create(factors.c_str(),initial.q.data(),initial.dq.data(),total.data(),.06,error,sizeof(error));
#endif
  if(!raw)throw std::runtime_error(error);if(juggling_hil_start(raw,error,sizeof(error)))throw std::runtime_error(error);
  juggling::MotionGuard guard;
#ifdef JUGGLING_JOINT_VELOCITY
  juggling::JointVelocityTracking velocity_servo(initial.dq_d);
  std::array<double,7> filtered_torque=initial.tau_J,torque_rate{};
#endif
  std::signal(SIGINT,request_stop);std::signal(SIGTERM,request_stop);std::signal(SIGHUP,request_stop);
  std::cout<<"Starting no-ball HIL; realtime enforcement="<<(!allow_nonrealtime)<<". MPC/ball workers are asynchronous; libfranka callback never waits for them.\n"<<std::flush;
  // Existing robot collision thresholds are preserved. This harness does not
  // disable reflexes or recover errors automatically; homing is opt-in above.
  control_entered=true;
#ifdef JUGGLING_JOINT_VELOCITY
  using ControlCommand=franka::JointVelocities;
#else
  using ControlCommand=franka::Torques;
#endif
  robot.control([&](const franka::RobotState& state,franka::Duration period)->ControlCommand{
    auto entered=std::chrono::steady_clock::now();
    double dt=period.toSec();time+=dt;auto g_now=model.gravity(state);std::array<double,7> applied,next,command,mpc_total;Eigen::Matrix<double,21,1> reference;double age=0;
    for(int j=0;j<7;++j)applied[j]=state.tau_J_d[j]+g_now[j];
#ifdef JUGGLING_JOINT_VELOCITY
    int status=juggling_hil_tick_acceleration(raw,time,state.q.data(),state.dq.data(),state.ddq_d.data(),next.data(),reference.data(),&age);
#else
    int status=juggling_hil_tick(raw,time,state.q.data(),state.dq.data(),applied.data(),next.data(),reference.data(),&age);
#endif
    if(status&&!fault)fault=status;
    mpc_total=next;if(count>=samples.size()&&!fault)fault=10;
#ifdef JUGGLING_JOINT_VELOCITY
    // tau_J_d is zero in this internal-servo mode, not the execution torque.
    // Monitor raw torque sensors for magnitude; differentiate a 5 ms low-pass
    // for a noise-limited rate monitor. Log both raw and filtered quantities.
    for(int j=0;j<7;++j){
      if(dt>0){double next=filtered_torque[j]-std::expm1(-dt/.005)*(state.tau_J[j]-filtered_torque[j]);torque_rate[j]=(next-filtered_torque[j])/dt;filtered_torque[j]=next;}
      if(!fault&&(!std::isfinite(state.tau_J[j])||std::abs(state.tau_J[j])>g::torque_limit[j]||std::abs(torque_rate[j])>g::torque_rate_limit[j]))fault=11;
    }
#endif
    bool stop=fault||time>=duration||stop_requested;if(stop)stopping+=dt;
#ifdef JUGGLING_JOINT_VELOCITY
    std::array<double,21> ref;std::copy(reference.data(),reference.data()+21,ref.begin());
    command=velocity_servo.step(ref,state.q,state.dq,dt,stop);
#else
    auto mass_now=model.mass(state);auto coriolis=model.coriolis(state);juggling::MotionGuard::V bias;
    for(int j=0;j<7;++j)bias[j]=g_now[j]+coriolis[j];
    juggling::MotionGuard::M mass=Eigen::Map<const juggling::MotionGuard::M>(mass_now.data());
    juggling::MotionGuard::V acceleration=reference.tail<7>()+100.*(reference.head<7>()-Eigen::Map<const juggling::MotionGuard::V>(state.q.data()))+20.*(reference.segment<7>(7)-Eigen::Map<const juggling::MotionGuard::V>(state.dq.data()));
    if(stop)for(int j=0;j<7;++j)acceleration[j]=-8.*state.dq[j];
    juggling::MotionGuard::V requested=mass*acceleration+bias;
    auto bounded=guard.filter(mass,bias,requested,Eigen::Map<const juggling::MotionGuard::V>(applied.data()),dt);
    std::copy(bounded.data(),bounded.data()+7,next.begin());
    for(int j=0;j<7;++j){
      // MPC returns total torque; FCI adds gravity/friction internally. Do not
      // add Coriolis again: it is already included in the MPC's total torque.
      double requested=next[j]-g_now[j];
      double change=g::torque_rate_limit[j]*dt;command[j]=std::clamp(requested,state.tau_J_d[j]-change,state.tau_J_d[j]+change);
    }
#endif
    if(count<samples.size()){auto& s=samples[count++];s.t=time;s.dt=dt;s.age=age;s.success_rate=state.control_command_success_rate;s.fault=fault;s.stopping=stop;
      s.q=state.q;s.dq=state.dq;s.mpc_total=mpc_total;s.tau_desired=state.tau_J_d;s.tau_measured=state.tau_J;
#ifdef JUGGLING_JOINT_VELOCITY
      s.velocity_command=command;s.torque_monitored=filtered_torque;s.torque_rate_monitored=torque_rate;s.acceleration_desired=state.ddq_d;
#else
      s.command=command;
#endif
      for(int j=0;j<7;++j){s.q_ref[j]=reference[j];s.dq_ref[j]=reference[7+j];s.ddq_ref[j]=reference[14+j];}
      s.callback_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-entered).count();}
    auto out=ControlCommand(command);
#ifdef JUGGLING_JOINT_VELOCITY
    bool settled=true;for(int j=0;j<7;++j)settled=settled&&std::abs(command[j])<.01&&std::abs(state.dq[j])<.02;
    if(stopping>=.5&&settled)return franka::MotionFinished(out);
    if(stopping>1.5)throw std::runtime_error("Velocity servo did not settle during controlled stop");
#else
    if(stopping>=.5)return franka::MotionFinished(out);
#endif
    return out;
#ifdef JUGGLING_JOINT_VELOCITY
  },franka::ControllerMode::kJointImpedance,true,100.0);
#else
  },true,100.0);
#endif
  control_completed=true;
  juggling_hil_stop(raw);if(juggling_hil_save(raw,output.c_str(),error,sizeof(error)))throw std::runtime_error(error);
  juggling_hil_destroy(raw);raw=nullptr;save();return fault?1:0;
 }catch(const std::exception& e){failure=e.what();if(raw){juggling_hil_stop(raw);juggling_hil_save(raw,output.c_str(),error,sizeof(error));juggling_hil_destroy(raw);}save();std::cerr<<e.what()<<'\n';return 1;}
}

#include "runtime_description.hpp"
// Supervised calibration / physical-ball session. No robot connection without
// --execute. The supervisor, SDK, and callback use the SAME host monotonic clock.
// stdin: heartbeat | move ID SOURCE_TIME Q[7] | plan SOURCE_TIME (Q,DQ,TAU,DDQ)[4x28] | stop
// stdout: measured state JSONL at 500 Hz. All parsing, logging and I/O off callback.
#include "session_control.hpp"
#include "experiment_duration.hpp"
#include "joint_velocity_tracking.hpp"
#include "json_text.hpp"
#include "native_transport.hpp"
#include <franka/robot.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <sstream>
#include <thread>
#include <unistd.h>
namespace {
namespace g=juggling::generated;
bool normal_thread_priority(){sched_param p{};return pthread_setschedparam(pthread_self(),SCHED_OTHER,&p)==0;}
using Clock=std::chrono::steady_clock;
double now(){return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();}
volatile std::sig_atomic_t interrupted=0;
void interrupt(int){interrupted=1;}
template<class T,size_t N> class Queue {
 std::array<T,N> data_{};std::atomic<size_t> r_{0},w_{0};
 public:
 bool push(const T& x){auto w=w_.load(std::memory_order_relaxed),n=(w+1)%N;if(n==r_.load(std::memory_order_acquire))return false;data_[w]=x;w_.store(n,std::memory_order_release);return true;}
 bool pop(T& x){auto r=r_.load(std::memory_order_relaxed);if(r==w_.load(std::memory_order_acquire))return false;x=data_[r];r_.store((r+1)%N,std::memory_order_release);return true;}
};
struct Sample {franka::RobotState state;double time=0,elapsed=0,dt=0;int id=0;bool moving=false,mpc=false,stopping=false,publish=false;std::array<double,7> command{},filtered_torque{},filtered_torque_rate{};int fault=0;std::array<double,21> reference{};double plan_age=0;bool reference_valid=false;unsigned long long timing_warnings=0;double timing_warning_age=0.;};
juggling::native::NativePacket native_packet(const Sample& s){
 juggling::native::NativePacket p;
 p.receive_time=s.time;p.elapsed=s.elapsed;p.dt=s.dt;p.move_id=s.id;p.fault=s.fault;
 p.moving=s.moving;p.mpc=s.mpc;p.stopping=s.stopping;p.reference_valid=s.reference_valid;p.mpc_plan_age_s=s.plan_age;
 p.q=s.state.q;p.dq=s.state.dq;p.q_d=s.state.q_d;p.dq_d=s.state.dq_d;p.ddq_d=s.state.ddq_d;p.tau_J=s.state.tau_J;
 p.O_T_EE=s.state.O_T_EE;p.F_T_EE=s.state.F_T_EE;p.velocity_command=s.command;
 p.guard_filtered_torque=s.filtered_torque;p.guard_filtered_torque_rate=s.filtered_torque_rate;p.reference_q_dq_ddq=s.reference;
 p.robot_time=s.state.time.toSec();p.success_rate=s.state.control_command_success_rate;return p;
}
template<size_t N> void array(std::ostream& s,const std::array<double,N>& a){s<<'[';for(size_t i=0;i<N;++i){if(i)s<<',';s<<a[i];}s<<']';}
std::string json(const Sample& s,bool compact=false){std::ostringstream o;o<<std::setprecision(17)<<std::boolalpha;
 o<<"{\"receive_time\":"<<s.time<<",\"elapsed\":"<<s.elapsed<<",\"dt\":"<<s.dt<<",\"move_id\":"<<s.id<<",\"moving\":"<<s.moving<<",\"mpc\":"<<s.mpc<<",\"stopping\":"<<s.stopping;
 o<<",\"reference_valid\":"<<s.reference_valid<<",\"mpc_plan_age_s\":"<<s.plan_age<<",\"reference_q_dq_ddq\":";array(o,s.reference);
 o<<",\"delivery_timing_warnings\":"<<s.timing_warnings<<",\"delivery_warning_age_s\":"<<s.timing_warning_age;
 o<<",\"fault\":"<<s.fault<<",\"guard_filtered_torque\":";array(o,s.filtered_torque);o<<",\"guard_filtered_torque_rate\":";array(o,s.filtered_torque_rate);
 o<<",\"q\":";array(o,s.state.q);o<<",\"dq\":";array(o,s.state.dq);o<<",\"q_d\":";array(o,s.state.q_d);
 o<<",\"ddq_d\":";array(o,s.state.ddq_d);o<<",\"tau_J\":";array(o,s.state.tau_J);o<<",\"velocity_command\":";array(o,s.command);
 if(compact){
  // Live hand-eye alignment uses the measured EE pose and configured flange
  // transform even though MPC feedback itself needs only joint state.
  o<<",\"O_T_EE\":";array(o,s.state.O_T_EE);o<<",\"F_T_EE\":";array(o,s.state.F_T_EE);
  o<<'}';return o.str();}
 o<<",\"dq_d\":";array(o,s.state.dq_d);o<<",\"tau_J_d\":";array(o,s.state.tau_J_d);o<<",\"dtau_J\":";array(o,s.state.dtau_J);
 o<<",\"tau_ext_hat_filtered\":";array(o,s.state.tau_ext_hat_filtered);o<<",\"O_F_ext_hat_K\":";array(o,s.state.O_F_ext_hat_K);o<<",\"K_F_ext_hat_K\":";array(o,s.state.K_F_ext_hat_K);
 o<<",\"joint_contact\":";array(o,s.state.joint_contact);o<<",\"cartesian_contact\":";array(o,s.state.cartesian_contact);
 o<<",\"joint_collision\":";array(o,s.state.joint_collision);o<<",\"cartesian_collision\":";array(o,s.state.cartesian_collision);
 o<<",\"O_T_EE\":";array(o,s.state.O_T_EE);o<<",\"F_T_EE\":";array(o,s.state.F_T_EE);
 o<<",\"robot_time\":"<<s.state.time.toSec()<<",\"success_rate\":"<<s.state.control_command_success_rate<<'}';return o.str();}
}
int main(int argc,char** argv){
 if(argc==2&&std::string(argv[1])=="--describe"){
  std::cout<<"{\"protocol\":2,\"model_sha256\":\""<<JUGGLING_MODEL_SHA256<<"\",\"measured_torque_rate_guard_Nm_s\":"<<juggling::measured_torque_rate_guard<<",\"controlled_stop_deadline_s\":"<<juggling::controlled_stop_deadline<<",\"readback_period_s\":"<<juggling::readback_period<<",\"max_callback_period_s\":"<<juggling::robot_callback_period_limit<<",\"runtime_tuning\":"<<juggling::runtime_tuning_description()<<",\"robot_connected\":false}\n";return 0;
 }
 std::string ip,output,ipc_path;bool execute=false,nonrt=false,homing_only=false,require_priority=false,compact_readback=false;int control_cpu=-1;double seconds=300,max_plan_delivery=.06;
 for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--execute")execute=true;else if(a=="--allow-nonrealtime")nonrt=true;
  else if(a=="--homing-only")homing_only=true;
  else if(a=="--require-realtime-priority")require_priority=true;
  else if(a=="--compact-readback")compact_readback=true;
  else if(a=="--ipc"&&i+1<argc)ipc_path=argv[++i];
  else if(a=="--control-cpu"&&i+1<argc)control_cpu=std::stoi(argv[++i]);
  else if(a=="--robot"&&i+1<argc)ip=argv[++i];else if(a=="--output"&&i+1<argc)output=argv[++i];
  else if(a=="--max-plan-delivery"&&i+1<argc)max_plan_delivery=std::stod(argv[++i]);
  else if(a=="--seconds"&&i+1<argc)seconds=std::stod(argv[++i]);else{std::cerr<<"Unknown option\n";return 2;}}
 if(!execute||ip.empty()||output.empty()){std::cout<<"No connection. Usage: juggling_franka_session --execute --robot HOST --output NEW_DIR [--allow-nonrealtime] [--seconds 300]\n";return execute?2:0;}
 if(!std::isfinite(seconds)||seconds<0){std::cerr<<"Session duration must be nonnegative (0 means no time limit)\n";return 2;}
 if(control_cpu < -1 || control_cpu>=CPU_SETSIZE){std::cerr<<"Invalid control CPU\n";return 2;}
 int rt=0;std::ifstream("/sys/kernel/realtime")>>rt;if(!nonrt&&rt!=1){std::cerr<<"Realtime kernel required unless --allow-nonrealtime\n";return 2;}
 if(!std::filesystem::create_directory(output)){std::cerr<<"Output must be new\n";return 2;}
 static_assert(std::atomic<double>::is_always_lock_free);
 Queue<juggling::SessionCommand,32> commands;auto samples=std::make_unique<Queue<Sample,4096>>();
 size_t logged_samples=0;bool log_ok=true;
 std::atomic<bool> running{true},external_stop{false};std::atomic<double> heartbeat{now()};
 juggling::native::SharedRegion region; juggling::native::SharedRobot* ipc=nullptr;
 std::thread input,writer;std::string failure;char control_error[256]{};bool entered=false,complete=false;int fault=0;
 std::signal(SIGINT,interrupt);std::signal(SIGTERM,interrupt);std::signal(SIGHUP,interrupt);std::signal(SIGPIPE,SIG_IGN);
 try{
  if(!ipc_path.empty()){region=juggling::native::SharedRegion::open(ipc_path);ipc=&region.get();}
  franka::Robot robot(ip,nonrt?franka::RealtimeConfig::kIgnore:franka::RealtimeConfig::kEnforce);
  int scheduling_policy=0;sched_param scheduling{};
  if(pthread_getschedparam(pthread_self(),&scheduling_policy,&scheduling)!=0)throw std::runtime_error("Cannot inspect control-thread scheduling");
  if(require_priority&&(scheduling_policy!=SCHED_FIFO||scheduling.sched_priority<=0)){
   scheduling.sched_priority=sched_get_priority_max(SCHED_FIFO);
   if(pthread_setschedparam(pthread_self(),SCHED_FIFO,&scheduling)!=0)
    throw std::runtime_error("Realtime scheduling permission unavailable; refresh login resource limits");
   scheduling_policy=SCHED_FIFO;
  }
  std::ofstream(output+"/scheduling.json")<<"{\"control_policy\":"<<scheduling_policy
    <<",\"control_priority\":"<<scheduling.sched_priority<<",\"realtime_kernel\":"<<rt
    <<",\"helper_threads_policy\":\"SCHED_OTHER\",\"requested_control_cpu\":"<<control_cpu
    <<",\"compact_readback\":"<<std::boolalpha<<compact_readback<<"}\n";
  auto initial=robot.readOnce();std::ofstream(output+"/robot_initial.json")<<std::setprecision(17)<<initial<<'\n';
  if(initial.robot_mode!=franka::RobotMode::kIdle||initial.current_errors)throw std::runtime_error("Robot must be idle and error-free");
  for(int j=0;j<7;++j)if(!std::isfinite(initial.q[j])||std::abs(initial.dq[j])>.03||initial.q[j]<g::joint_lower[j]||initial.q[j]>g::joint_upper[j])throw std::runtime_error("Robot must start stationary in joint bounds");
  // Preserve Desk end-effector/load/impedance/collision settings; no recovery.
  if(!ipc)input=std::thread([&]{try{
   if(!normal_thread_priority())throw std::runtime_error("Cannot set input thread to normal scheduling");
   std::string pending;char buffer[4096];
   while(running){pollfd fd{STDIN_FILENO,POLLIN,0};int ready=poll(&fd,1,20);if(ready<0)continue;if(!ready)continue;
    auto n=read(STDIN_FILENO,buffer,sizeof(buffer));if(n<=0){external_stop=true;break;}pending.append(buffer,n);
    if(pending.size()>32768)throw std::runtime_error("Oversized command");
    size_t end;while((end=pending.find('\n'))!=std::string::npos){std::istringstream line(pending.substr(0,end));pending.erase(0,end+1);std::string kind;line>>kind;
     if(kind=="heartbeat"){std::string extra;if(line>>extra)throw std::runtime_error("Malformed heartbeat");heartbeat=now();continue;}
     juggling::SessionCommand cmd;
     if(kind=="stop"){external_stop=true;break;}
     else if(kind=="move"){cmd.kind=juggling::SessionCommand::Move;if(!(line>>cmd.id>>cmd.source)||cmd.id<=0)throw std::runtime_error("Bad move");for(double& x:cmd.goal)if(!(line>>x)||!std::isfinite(x))throw std::runtime_error("Bad joint target");}
     else if(kind=="plan"){if(homing_only)throw std::runtime_error("MPC plans prohibited in homing-only session");cmd.kind=juggling::SessionCommand::Plan;if(!(line>>cmd.source))throw std::runtime_error("Bad plan time");for(double& x:cmd.trajectory)if(!(line>>x)||!std::isfinite(x))throw std::runtime_error("Bad plan value");}
     else throw std::runtime_error("Unknown supervisor command");
     std::string extra;if(line>>extra)throw std::runtime_error("Extra command values");if(!commands.push(cmd))throw std::runtime_error("Command queue overflow");heartbeat=now();
    }
   }
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';external_stop=true;}});
  writer=std::thread([&]{std::ofstream log(output+"/robot.jsonl");Sample s;double last_timing_warning=-1.;unsigned long long reported_timing_warnings=0;
   if(!normal_thread_priority()){log_ok=false;external_stop=true;return;}
   for(;;){
    if(!samples->pop(s)){if(!running)break;std::this_thread::sleep_for(std::chrono::microseconds(200));continue;}
    if(s.timing_warnings>reported_timing_warnings&&s.time-last_timing_warning>=1.){
     std::cerr<<"WARNING: native command delivery/plan age "<<1000*s.timing_warning_age<<" ms ("<<s.timing_warnings<<" timing warnings); continuing.\n";
     reported_timing_warnings=s.timing_warnings;last_timing_warning=s.time;
    }
    // Optional compact feedback goes out before full archival serialization.
    // The complete 1 kHz record remains on disk in every mode.
    if(!ipc&&s.publish&&compact_readback)std::cout<<json(s,true)<<'\n'<<std::flush;
    auto text=json(s);log<<text<<'\n';++logged_samples;
    if(!ipc&&s.publish&&!compact_readback)std::cout<<text<<'\n'<<std::flush;
    if(!log||(!ipc&&!std::cout)){log_ok=false;external_stop=true;break;}
   }
  });
  juggling::SessionControl control(initial.q_d);control.delivery_limit(max_plan_delivery);juggling::JointVelocityTracking servo(initial.dq_d);
  juggling::StopCompletion stop_completion;
  auto filtered=initial.tau_J;double elapsed=0,stopping=0,next_sample=0;heartbeat=now();
  // Helpers retain their inherited affinity. Pin only the control thread.
  if(control_cpu>=0){cpu_set_t cpus;CPU_ZERO(&cpus);CPU_SET(control_cpu,&cpus);
   if(pthread_setaffinity_np(pthread_self(),sizeof(cpus),&cpus)!=0)throw std::runtime_error("Cannot pin control thread");}
  std::ofstream(output+"/control_cpu.json")<<"{\"requested\":"<<control_cpu<<",\"observed_before_control\":"<<sched_getcpu()<<"}\n";
  entered=true;if(ipc)ipc->producer_state.store(1,std::memory_order_release);
  robot.control([&](const franka::RobotState& s,franka::Duration period)->franka::JointVelocities{
   double dt=period.toSec(),t=now();elapsed+=dt;
   if(!juggling::valid_robot_callback_period(dt))throw std::runtime_error("Invalid callback period: "+std::to_string(dt)+" s");
   std::array<double,7> filtered_rate{};
   for(int j=0;j<7;++j){
    if(!std::isfinite(s.q[j])||!std::isfinite(s.dq[j])||s.q[j]<g::joint_lower[j]||s.q[j]>g::joint_upper[j]||std::abs(s.dq[j])>g::velocity_limit[j])throw std::runtime_error("Invalid measured joint state");
    double rate=0;if(dt>0){double next=filtered[j]-std::expm1(-dt/.005)*(s.tau_J[j]-filtered[j]);rate=(next-filtered[j])/dt;filtered[j]=next;}
    filtered_rate[j]=rate;
    if((!std::isfinite(s.tau_J[j])||std::abs(s.tau_J[j])>g::torque_limit[j]||std::abs(rate)>juggling::measured_torque_rate_guard)&&!fault)fault=11;
   }
   bool shutdown_requested=external_stop||interrupted||juggling::experiment_duration_expired(elapsed,seconds)||control.stopped()||(ipc&&ipc->stop.load(std::memory_order_acquire));
   if(juggling::session_heartbeat_expired(t,ipc?ipc->heartbeat.load(std::memory_order_acquire):heartbeat.load(),shutdown_requested)&&!fault)fault=12;
   juggling::SessionCommand cmd;
   std::array<double,21> ref{};
   if(!fault&&!shutdown_requested){
    try{while(ipc?ipc->commands.pop(cmd):commands.pop(cmd)){
      if(cmd.kind!=juggling::SessionCommand::Move&&cmd.kind!=juggling::SessionCommand::Plan&&cmd.kind!=juggling::SessionCommand::Stop)
       throw std::runtime_error("Invalid native command kind");
      if(cmd.kind==juggling::SessionCommand::Move&&cmd.id<=0)throw std::runtime_error("Invalid native move ID");
      if(homing_only&&cmd.kind==juggling::SessionCommand::Plan)throw std::runtime_error("MPC plans prohibited in homing-only session");
      control.accept(cmd,s.q_d,t);
     }ref=control.reference(t);
     for(int j=0;j<7;++j)if(std::abs(ref[j]-s.q[j])>.1)throw std::runtime_error("Joint tracking error >0.1 rad");
    }catch(const std::exception& e){std::snprintf(control_error,sizeof(control_error),"%s",e.what());fault=13;}
   }
   bool stop=fault||shutdown_requested||control.stopped();if(stop)stopping+=dt;
   auto velocity=servo.step(ref,s.q,s.dq,dt,stop,(fault||interrupted)?12.:juggling::JointVelocityTracking::normal_stop_gain);
   // Record every SDK callback (~1 kHz); publish readback at 500 Hz. No file I/O
   // or serialization occurs in the callback. Overflow requests a stop.
   bool publish=elapsed+1e-9>=next_sample;
   Sample sample{s,t,elapsed,dt,control.id(),control.moving(t),control.mpc(),stop,publish,velocity};
   sample.filtered_torque=filtered;sample.filtered_torque_rate=filtered_rate;sample.fault=fault;sample.reference=ref;sample.reference_valid=!stop;sample.plan_age=control.mpc()?t-control.plan_source_time():0;
   sample.timing_warnings=control.timing_warnings();sample.timing_warning_age=control.timing_warning_age();
   if(!samples->push(sample)&&!fault)fault=14;
   // Direct callback-to-supervisor feedback is independent of the JSON writer.
   // A stalled supervisor cannot silently lose state: queue overflow stops.
   if(ipc&&publish&&!ipc->states.push(native_packet(sample))&&!fault)fault=15;
   if(ipc&&fault)ipc->fault.store(static_cast<unsigned>(fault),std::memory_order_release);
   if(publish)do{next_sample+=juggling::readback_period;}while(next_sample<=elapsed+1e-9);
   if(stop&&stop_completion.update(servo.zero_hold(),s.dq,s.dq_d,s.ddq_d,dt))return franka::MotionFinished(franka::JointVelocities(velocity));
   if(stopping>juggling::controlled_stop_deadline)throw std::runtime_error("Controlled stop failed to settle");
   return franka::JointVelocities(velocity);
  },franka::ControllerMode::kJointImpedance,true,100.);
  complete=true;std::ofstream(output+"/robot_final.json")<<std::setprecision(17)<<robot.readOnce()<<'\n';
 }catch(const std::exception& e){failure=e.what();std::cerr<<failure<<'\n';}
 if(ipc){ipc->fault.store(static_cast<unsigned>(fault),std::memory_order_release);ipc->producer_state.store(failure.empty()&&fault==0?2:3,std::memory_order_release);}
 running=false;if(input.joinable())input.join();if(writer.joinable())writer.join();
 if(failure.empty()&&control_error[0])failure=control_error;
 if(ipc&&(!log_ok||!failure.empty()||fault))ipc->producer_state.store(3,std::memory_order_release);
 std::ofstream(output+"/session.json")<<std::boolalpha<<"{\"control_entered\":"<<entered<<",\"control_completed\":"<<complete<<",\"fault\":"<<fault<<",\"logged_samples\":"<<logged_samples<<",\"log_ok\":"<<log_ok<<",\"error\":"<<juggling::json_text(failure)<<"}\n";
 return failure.empty()&&fault==0&&log_ok?0:1;
}

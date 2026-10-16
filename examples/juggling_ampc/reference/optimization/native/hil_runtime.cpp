#include "hil_runtime.hpp"
#include <cmath>
#include "model_config.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>

#include "ball_gravity.hpp"
#ifndef JUGGLING_BALL_DRAG
#define JUGGLING_BALL_DRAG .02
#endif
namespace juggling {
namespace g=generated;
using Steady=std::chrono::steady_clock;
VirtualBall::VirtualBall(const std::string& factors,const ArmFrame& initial):robot_(factors),previous_(initial){state_.setZero();state_[2]=.67;state_.segment<7>(6)=Eigen::Map<const Joint>(initial.q.data());state_.segment<7>(13)=Eigen::Map<const Joint>(initial.dq.data());}
Joint VirtualBall::prediction_torque(const ArmFrame& frame){
  return robot_.rnea(Eigen::Map<const Joint>(frame.q.data()),Eigen::Map<const Joint>(frame.dq.data()),Eigen::Map<const Joint>(frame.reference_acceleration.data()));
}
State VirtualBall::free(const State& before,double start,double duration,const ArmFrame& next){
  double end=start+duration,f=(end-previous_.t)/(next.t-previous_.t),time=start+.5*duration;State out=before;
  Eigen::Vector3d force(ball_gravity[0]+.12*std::cos(.8*time)+.04*std::sin(1.7*time),ball_gravity[1]+.10*std::sin(.8*time)+.03*std::cos(1.3*time),ball_gravity[2]);
  const double drag=JUGGLING_BALL_DRAG;double decay=std::exp(-drag*duration),phi1=drag>0?-std::expm1(-drag*duration)/drag:duration,phi2=drag>0?(duration-phi1)/drag:.5*duration*duration;
  out.head<3>()+=phi1*before.segment<3>(3)+phi2*force;out.segment<3>(3)=decay*before.segment<3>(3)+phi1*force;
  for(int j=0;j<7;++j){out[6+j]=(1-f)*previous_.q[j]+f*next.q[j];out[13+j]=(1-f)*previous_.dq[j]+f*next.dq[j];}return out;
}
void VirtualBall::advance(const ArmFrame& next){
  if(next.t<=previous_.t)return;
  if(next.t-previous_.t>.01)throw std::runtime_error("Arm measurement gap exceeds virtual-ball limit");
  double t=previous_.t;int contacts=0;
  while(t<next.t-1e-12){double dt=std::min(.0005,next.t-t);State end=free(state_,t,dt,next);auto a=robot_.geometry(state_),b=robot_.geometry(end);
    if(a.gap>=-1e-10&&b.gap<=0&&(a.speed<0||b.speed<0)){
      double lo=0,hi=dt;while(hi-lo>1e-10){double mid=.5*(lo+hi);if(robot_.geometry(free(state_,t,mid,next)).gap>0)lo=mid;else hi=mid;}
      double hit=.5*(lo+hi);State collision=free(state_,t,hit,next);auto geo=robot_.geometry(collision);
      if(geo.radial<=g::paddle_radius&&geo.speed<0){Joint measured_dq=collision.segment<7>(13);apply_ball_impact(robot_,collision);
        // Virtual impact affects only the virtual ball. There is no physical
        // ball, so no synthetic collision torque or recoil is sent to hardware.
        collision.segment<7>(13)=measured_dq;state_=collision;t+=hit;++impacts_;if(++contacts>8)throw std::runtime_error("Repeated virtual contact");continue;}}
    state_=end;t+=dt;
  }
  previous_=next;
}
HilRuntime::HilRuntime(const std::string& factors,const ArmFrame& initial,double max_age):controller_(factors),ball_(factors,initial),initial_(initial),max_age_(max_age){
  if(!std::isfinite(max_age)||max_age<=0||max_age>g::stage_times[3])throw std::runtime_error("Invalid command age limit");
  ball_log_.reserve(120000);solve_log_.reserve(10000);
  Joint boundary=Eigen::Map<const Joint>(initial.torque.data());if(initial.acceleration_input)boundary=ball_.prediction_torque(initial);
  auto info=controller_.step(ball_.state(),boundary,0.);solve_log_.push_back({0.,info});
  active_.source_time=0;
  for(int k=0;k<4;++k)for(int j=0;j<7;++j){active_.trajectory[k*21+j]=controller_.physical_plan()(6+j,k);active_.trajectory[k*21+7+j]=controller_.physical_plan()(13+j,k);active_.trajectory[k*21+14+j]=controller_.physical_plan()(23+j,k);}
}
HilRuntime::~HilRuntime(){stop();}
void HilRuntime::start(){if(running_.exchange(true))throw std::runtime_error("HIL already running");physics_=std::thread(&HilRuntime::physics_loop,this);mpc_=std::thread(&HilRuntime::mpc_loop,this);}
int HilRuntime::tick(const ArmFrame& frame,std::array<double,7>& target,std::array<double,21>& reference,double& age) noexcept {
  // This method belongs exclusively to the libfranka callback thread.
  double dt=frame.t-last_tick_;
  auto invalid=[&](int code){int first=record_fault(code);target.fill(0.);reference.fill(0.);age=0.;return first;};
  if(!std::isfinite(frame.t)||dt<0||dt>.005)return invalid(1);
  for(int j=0;j<7;++j)if(!std::isfinite(frame.q[j])||!std::isfinite(frame.dq[j])||!std::isfinite(frame.torque[j])||(frame.acceleration_input&&!std::isfinite(frame.reference_acceleration[j]))||frame.q[j]<g::joint_lower[j]||frame.q[j]>g::joint_upper[j]||std::abs(frame.dq[j])>g::velocity_limit[j])return invalid(2);
  if(dt>0&&!arms_.push(frame))record_fault(3);
  CommandFrame next;
  if(commands_.latest(next)){if(frame.t-next.source_time<=max_age_){active_=next;}else record_fault(4);}
  age=frame.t-active_.source_time;if(age>max_age_)record_fault(4);
  // Evaluate the planned kinematics at actual command age. Use the bounded
  // interval acceleration rather than differentiating a cubic through the small
  // position defects left by one RTI step. The adapter supplies computed-torque
  // tracking and an acceleration/jerk guard with the measured robot model.
  double elapsed=std::clamp(age,0.,g::stage_times[3]);int k=std::min(int(std::upper_bound(g::stage_times.begin(),g::stage_times.begin()+4,elapsed)-g::stage_times.begin())-1,2);double s=elapsed-g::stage_times[k],f=s/g::stage_steps[k];
  for(int j=0;j<7;++j){const double* x=active_.trajectory.data()+21*k;const double* y=x+21;double acceleration=(y[7+j]-x[7+j])/g::stage_steps[k];
    reference[j]=x[j]+s*x[7+j]+.5*s*s*acceleration;reference[7+j]=x[7+j]+s*acceleration;reference[14+j]=acceleration;
    target[j]=(1-f)*x[14+j]+f*y[14+j];}
  last_tick_=frame.t;return fault_.load();
}
void HilRuntime::physics_loop(){
  try{double next_mpc=g::control_dt;ArmFrame frame;
    for(;;){
      if(!arms_.pop(frame)){if(!running_.load(std::memory_order_acquire))break;std::this_thread::sleep_for(std::chrono::microseconds(100));continue;}
      if(frame.acceleration_input){Joint boundary=ball_.prediction_torque(frame);std::copy(boundary.data(),boundary.data()+7,frame.torque.begin());}
      ball_.advance(frame);StateFrame state;state.t=frame.t;std::copy(ball_.state().data(),ball_.state().data()+23,state.state.begin());state.torque=frame.torque;
      ball_log_.push_back(state);
      if(!ball_.state().allFinite()||ball_.state()[2]<-.3||ball_.state()[2]>2.){record_fault(5);break;}
      if(running_.load(std::memory_order_acquire)&&frame.t>=next_mpc-1e-10){if(!states_.push(state)){record_fault(6);break;}next_mpc=frame.t+g::control_dt;}
    }
  }catch(...){record_fault(7);}
}
void HilRuntime::mpc_loop(){
  try{double previous=0.;StateFrame frame;
    while(running_.load(std::memory_order_acquire)){
      if(!states_.latest(frame)){std::this_thread::sleep_for(std::chrono::microseconds(100));continue;}
      auto info=controller_.step(Eigen::Map<const State>(frame.state.data()),Eigen::Map<const Joint>(frame.torque.data()),frame.t-previous);previous=frame.t;
      solve_log_.push_back({frame.t,info});CommandFrame command;command.source_time=frame.t;
      for(int k=0;k<4;++k)for(int j=0;j<7;++j){command.trajectory[k*21+j]=controller_.physical_plan()(6+j,k);command.trajectory[k*21+7+j]=controller_.physical_plan()(13+j,k);command.trajectory[k*21+14+j]=controller_.physical_plan()(23+j,k);}if(!commands_.push(command)){record_fault(8);break;}
    }
  }catch(...){record_fault(9);}
}
void HilRuntime::stop(){running_.store(false,std::memory_order_release);if(physics_.joinable())physics_.join();if(mpc_.joinable())mpc_.join();}
void HilRuntime::save(const std::string& directory){
  if(running_)throw std::runtime_error("Stop HIL before exporting logs");std::filesystem::create_directories(directory);
  std::ofstream ball(directory+"/hil_ball.csv");ball<<std::setprecision(17)<<"time";for(int i=0;i<23;++i)ball<<",x"<<i;for(int i=0;i<7;++i)ball<<",tau"<<i;ball<<'\n';
  for(const auto& s:ball_log_){ball<<s.t;for(double x:s.state)ball<<','<<x;for(double x:s.torque)ball<<','<<x;ball<<'\n';}
  std::ofstream mpc(directory+"/hil_mpc.csv");mpc<<std::setprecision(17)<<"time,total_ms,shift_ms,data_ms,qp_ms,merit_ms,iterations,alpha,violation,clipping\n";
  for(const auto& s:solve_log_)mpc<<s.source_time<<','<<s.info.total_ms<<','<<s.info.shift_ms<<','<<s.info.data_ms<<','<<s.info.qp_ms<<','<<s.info.merit_ms<<','<<s.info.iterations<<','<<s.info.alpha<<','<<s.info.violation<<','<<s.info.clipping<<'\n';
  std::ofstream summary(directory+"/hil_runtime.json");summary<<"{\"fault\":"<<fault_.load()<<",\"virtual_impacts\":"<<ball_.impacts()<<",\"arm_samples\":"<<ball_log_.size()<<",\"mpc_steps\":"<<solve_log_.size()<<",\"max_command_age_s\":"<<max_age_<<"}\n";
}
}

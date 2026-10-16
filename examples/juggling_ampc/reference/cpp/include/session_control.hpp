#pragma once
#include "joint_homing.hpp"
#include "session_command.hpp"
#include "model_config.hpp"
#include <optional>
#include <stdexcept>

namespace juggling {
// Experimental abort threshold on the 5 ms filtered MEASURED torque derivative.
// Separate from generated MPC torque-rate constraints and Franka's 1000 Nm/s
// commanded-torque interface limit; those are different signals.
inline constexpr double measured_torque_rate_guard=800.;
inline constexpr double controlled_stop_deadline=2.;
inline constexpr double readback_period=.002;
// Once shutdown is requested, the callback completes its bounded stop without
// further supervisor input. A missing heartbeat still faults active control.
inline bool session_heartbeat_expired(double now,double last,bool shutdown_requested){
  return !shutdown_requested&&now-last>.15;
}
// Pure callback state machine; also exercised without libfranka in checks.
class SessionControl {
 public:
  using V=std::array<double,7>;
  explicit SessionControl(const V& q):hold_(q),max_age_(std::min(.06,generated::stage_times[3])){}
  // Offline empirical experiments can hold the physical plant fixed while
  // changing the controller's transmitted knot spacing. Hardware uses the
  // generated defaults unless this method is explicitly called before MPC.
  void execution_steps(const std::array<double,3>& steps){
    if(mpc_||stopped_)throw std::runtime_error("Cannot change active execution mesh");
    for(double step:steps)if(!std::isfinite(step)||step<=0||step>1.)throw std::runtime_error("Invalid execution mesh");
    stage_steps_=steps;stage_times_[0]=0.;for(int k=0;k<3;++k)stage_times_[k+1]=stage_times_[k]+steps[k];
    max_age_=std::min(.06,stage_times_[3]);
  }
  void delivery_limit(double seconds){
    if(mpc_||!std::isfinite(seconds)||seconds<=0||seconds>max_age_)throw std::runtime_error("Invalid plan delivery limit");
    delivery_limit_=seconds;
  }
  void accept(const SessionCommand& c,const V& q_d,double now){
    if(c.kind==SessionCommand::Stop){stopped_=true;return;}
    if(stopped_)throw std::runtime_error("Session has stopped");
    if(!std::isfinite(c.source))throw std::runtime_error("Nonfinite session command timestamp");
    const double delivery_age=now-c.source;
    if(delivery_age<-.002||delivery_age>(c.kind==SessionCommand::Plan?delivery_limit_:max_age_)){
      ++timing_warnings_;timing_warning_age_=delivery_age;
    }
    if(c.kind==SessionCommand::Move){
      if(mpc_)throw std::runtime_error("Cannot home after enabling MPC");
      for(int j=0;j<7;++j)if(std::abs(c.goal[j]-generated::home[j])>1.0)
        throw std::runtime_error("Move goal outside local HOME envelope");
      path_.emplace(q_d,c.goal,generated::joint_lower,generated::joint_upper,2.);
      began_=now;hold_=c.goal;id_=c.id;
    }else{
      if(moving(now))throw std::runtime_error("Cannot start MPC during a homing move");
      // Ignore an out-of-order delivery instead of replacing the newer plan.
      if(c.source<=last_source_){++timing_warnings_;timing_warning_age_=delivery_age;return;}
      for(int k=0;k<4;++k)for(int j=0;j<7;++j){
        double q=c.trajectory[28*k+j],v=c.trajectory[28*k+7+j],tau=c.trajectory[28*k+14+j],acc=c.trajectory[28*k+21+j];
        // An endpoint is needed only if its preceding interval can execute
        // before plan expiry. Finite values are required for all wire fields.
        // E.g. mesh24 node 3 (~94 ms) is unused by the 60 ms callback window.
        bool used=k==0||stage_times_[k-1]<max_age_;
        if(!std::isfinite(q)||!std::isfinite(v)||!std::isfinite(tau)||!std::isfinite(acc)||
           (used&&(q<generated::joint_lower[j]||q>generated::joint_upper[j]
           ||std::abs(v)>generated::velocity_limit[j]||std::abs(tau)>generated::torque_limit[j]
           // Future knots can retain small fixed-budget SQP defects. The
           // executed interpolant still has the stricter 1.05 check below,
           // and the velocity servo caps commanded acceleration at 0.95.
           ||std::abs(acc)>(k==0?1.05:1.10)*generated::acceleration_limit[j])))
          throw std::runtime_error("MPC plan outside state/torque bounds");
      }
      plan_=c;mpc_=true;last_source_=c.source;path_.reset();expiry_warned_=false;
    }
  }
  std::array<double,21> reference(double now){
    std::array<double,21> ref{};
    if(mpc_){
      double age=now-plan_.source;const bool hold=age>max_age_;
      if((age<-.002||hold)&&!expiry_warned_){++timing_warnings_;timing_warning_age_=age;expiry_warned_=true;}
      // A late update is diagnostic only. Hold the last valid-window position
      // beyond its end instead of extrapolating acceleration without a bound.
      age=std::clamp(age,0.,max_age_);int k=0;while(k<2&&age>=stage_times_[k+1])++k;
      // Integrate the acceleration ramp implied by planned torques, anchored
      // to measured q/dq. Fixed-budget shooting velocities can have dynamics
      // defects; forcing those knots into the reference creates torque spikes.
      for(int j=0;j<7;++j){double q=plan_.trajectory[j],v=plan_.trajectory[7+j];
        for(int i=0;i<=k;++i){const double* a=plan_.trajectory.data()+28*i;const double* b=a+28;
          double h=stage_steps_[i],dt=i<k?h:age-stage_times_[i];
          double acceleration=a[j+21],jerk=(b[j+21]-acceleration)/h;
          q+=v*dt+.5*acceleration*dt*dt+jerk*dt*dt*dt/6;
          v+=acceleration*dt+.5*jerk*dt*dt;
          ref[j+7]=v;ref[j+14]=acceleration+jerk*dt;
        }
        ref[j]=q;
        if(q<generated::joint_lower[j]||q>generated::joint_upper[j]||std::abs(ref[j+7])>generated::velocity_limit[j])throw std::runtime_error("MPC interpolated reference outside joint bounds");
        if(std::abs(ref[j+14])>generated::acceleration_limit[j]*1.05)throw std::runtime_error("MPC acceleration outside envelope");
        if(hold){ref[j+7]=0.;ref[j+14]=0.;}
      }
    }else if(path_){auto p=path_->at(now-began_);for(int j=0;j<7;++j){ref[j]=p.q[j];ref[j+7]=p.dq[j];ref[j+14]=p.ddq[j];}}
    else for(int j=0;j<7;++j)ref[j]=hold_[j];
    return ref;
  }
  double plan_source_time() const{return plan_.source;}
  unsigned long long timing_warnings() const{return timing_warnings_;}
  double timing_warning_age() const{return timing_warning_age_;}
  bool moving(double now) const{return path_&&now-began_<path_->duration();}
  bool mpc() const{return mpc_;}bool stopped() const{return stopped_;}int id() const{return id_;}
 private:
  V hold_;std::optional<JointHoming> path_;SessionCommand plan_;
  std::array<double,3> stage_steps_{generated::stage_steps[0],generated::stage_steps[1],generated::stage_steps[2]};
  std::array<double,4> stage_times_{generated::stage_times[0],generated::stage_times[1],generated::stage_times[2],generated::stage_times[3]};
  double began_=0,last_source_=-1,max_age_,delivery_limit_=.06;int id_=0;bool mpc_=false,stopped_=false;
  unsigned long long timing_warnings_=0;double timing_warning_age_=0.;bool expiry_warned_=false;
};
}

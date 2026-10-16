#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include "model_config.hpp"
#ifndef JUGGLING_SERVO_JERK_FRACTION
#define JUGGLING_SERVO_JERK_FRACTION .2
#endif
#ifndef JUGGLING_SERVO_POSITION_GAIN
#define JUGGLING_SERVO_POSITION_GAIN 100.
#define JUGGLING_SERVO_VELOCITY_GAIN 20.
#endif
#ifndef JUGGLING_PARAMETRIC_FACTORS
static_assert(JUGGLING_SERVO_JERK_FRACTION>0 && JUGGLING_SERVO_JERK_FRACTION<=.2);
#endif

namespace juggling {
// Operator-authorized callback tolerance; shared by the SDK bridge and servo.
inline constexpr double robot_callback_period_limit=.010;
inline bool valid_robot_callback_period(double dt){
  return std::isfinite(dt)&&dt>=0&&dt<=robot_callback_period_limit;
}
// Integrate a continuous velocity command for the SDK joint-impedance servo.
// The reference is the same MPC q/dq/ddq used by the torque execution layer.
class JointVelocityTracking {
 public:
  using V=std::array<double,7>;
  static constexpr double normal_stop_gain=4.5;
  explicit JointVelocityTracking(const V& velocity):velocity_(velocity) {}
  void execution_gains(double position,double velocity,double jerk_fraction){
    if(!std::isfinite(position)||!std::isfinite(velocity)||!std::isfinite(jerk_fraction)||position<=0||position>100.
       ||velocity<=0||velocity>20.||jerk_fraction<=0||jerk_fraction>.2)
      throw std::runtime_error("Invalid velocity-servo execution gains");
    position_gain_=position;velocity_gain_=velocity;jerk_fraction_=jerk_fraction;
  }
  V step(const std::array<double,21>& reference,const V& q,const V& dq,double dt,bool stop,double stop_gain=12.){
    if(!valid_robot_callback_period(dt))throw std::runtime_error("Invalid velocity-servo callback period");
    if(dt==0)return velocity_;
    if(stop&&!tail_){
      bool slow=true;for(int j=0;j<7;++j)slow=slow&&std::abs(velocity_[j])<.02&&std::abs(acceleration_[j])<.3;
      if(slow){tail_=true;tail_velocity_=velocity_;tail_acceleration_=acceleration_;}
    }
    if(tail_){
      // Cubic velocity tail matches current velocity/acceleration and reaches
      // exact zero velocity/acceleration in finite time; no threshold snap.
      constexpr double T=.3;tail_time_=std::min(T,tail_time_+dt);double s=tail_time_/T;
      for(int j=0;j<7;++j){
        velocity_[j]=(1-3*s*s+2*s*s*s)*tail_velocity_[j]+T*(s-2*s*s+s*s*s)*tail_acceleration_[j];
        acceleration_[j]=(-6*s+6*s*s)/T*tail_velocity_[j]+(1-4*s+3*s*s)*tail_acceleration_[j];
      }
      return velocity_;
    }
    for(int j=0;j<7;++j){
      double desired=stop?-stop_gain*velocity_[j]:reference[14+j]+position_gain_*(reference[j]-q[j])+velocity_gain_*(reference[7+j]-dq[j]);
      double fraction=(stop&&stop_gain==normal_stop_gain)?std::min(.08,jerk_fraction_):jerk_fraction_;
      if(stop && acceleration_[j]*velocity_[j]>0.){
        // Outward acceleration persists while jerk ramps through zero. Reserve
        // tracking margin for the unavoidable rise a^2/(2*j) in speed.
        double room=std::max(.001,.97*generated::velocity_limit[j]-std::abs(velocity_[j]));
        double required=acceleration_[j]*acceleration_[j]/(2*room*generated::jerk_limit[j]);
        fraction=std::max(fraction,std::min(jerk_fraction_,required));
      }
      double limit=.95*generated::acceleration_limit[j],change=fraction*generated::jerk_limit[j]*dt;
      double next=std::clamp(std::clamp(desired,-limit,limit),acceleration_[j]-change,acceleration_[j]+change);
      double velocity=velocity_[j]+dt*next;
      if(!std::isfinite(velocity)||std::abs(velocity)>generated::velocity_limit[j])
        throw std::runtime_error("Velocity-servo command exceeds research bound");
      acceleration_[j]=next;velocity_[j]=velocity;
    }
    return velocity_;
  }
  bool zero_hold() const{return tail_&&tail_time_>=.3;}
 private:
  double position_gain_=JUGGLING_SERVO_POSITION_GAIN,velocity_gain_=JUGGLING_SERVO_VELOCITY_GAIN,jerk_fraction_=JUGGLING_SERVO_JERK_FRACTION;
  V velocity_{},acceleration_{},tail_velocity_{},tail_acceleration_{};
  bool tail_=false;double tail_time_=0;
};
// SDK desired states must settle too: low measured speed alone is insufficient.
class StopCompletion {
 public:
  bool update(bool zero_hold,const std::array<double,7>& dq,const std::array<double,7>& dq_d,const std::array<double,7>& ddq_d,double dt){
    bool settled=zero_hold;
    for(int j=0;j<7;++j)settled=settled&&std::abs(dq[j])<.02&&std::abs(dq_d[j])<1e-4&&std::abs(ddq_d[j])<.01;
    settled_time_=settled?settled_time_+dt:0.;return settled_time_>=.1;
  }
 private:double settled_time_=0;
};
}

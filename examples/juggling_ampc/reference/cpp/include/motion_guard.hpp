#pragma once
#include <Eigen/Dense>
#include <cmath>
#include "model_config.hpp"
#include <algorithm>

namespace juggling {
// Fixed-size, allocation-free 1 kHz actuator projection. The hardware adapter
// supplies libfranka M and C+g; the mock supplies the same simulation quantities.
// This executes outside the MPC and bounds corrections from trajectory tracking.
class MotionGuard {
 public:
  using V=Eigen::Matrix<double,7,1>;using M=Eigen::Matrix<double,7,7>;
  V filter(const M& mass,const V& bias,const V& requested,const V& applied,double dt){
    Eigen::LDLT<M> factor(mass);V base=factor.solve(applied-bias);
    if(!initialized_){previous_=base;initialized_=true;}
    if(dt<=0)return applied;
    V desired=factor.solve(requested-bias);
    for(int j=0;j<7;++j){double a=.98*generated::acceleration_limit[j],jerk=.95*generated::jerk_limit[j]*dt;
      double center=std::clamp(previous_[j],-a,a);desired[j]=std::clamp(desired[j],std::max(-a,center-jerk),std::min(a,center+jerk));}
    V candidate=mass*desired+bias,delta=candidate-applied;double fraction=1.;
    for(int j=0;j<7;++j){double rate=.95*generated::torque_rate_limit[j]*dt,limit=.98*generated::torque_limit[j];
      if(std::abs(delta[j])>rate)fraction=std::min(fraction,rate/std::abs(delta[j]));
      if(delta[j]>0)fraction=std::min(fraction,std::max(0.,(limit-applied[j])/delta[j]));
      else if(delta[j]<0)fraction=std::min(fraction,std::max(0.,(-limit-applied[j])/delta[j]));}
    V output=applied+fraction*delta;previous_=factor.solve(output-bias);return output;
  }
 private:
  V previous_=V::Zero();bool initialized_=false;
};
}

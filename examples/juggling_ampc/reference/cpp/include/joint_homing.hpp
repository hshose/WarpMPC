#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace juggling {
// Synchronized minimum-jerk position reference for the robot's joint impedance
// controller. The convex joint-space path stays inside the endpoint joint bounds;
// it does not check obstacles in the physical workspace.
class JointHoming {
 public:
  using Joints=std::array<double,7>;
  struct Reference {Joints q{},dq{},ddq{},jerk{};};
  static constexpr double velocity_limit=.15,acceleration_limit=.25,jerk_limit=1.;
  JointHoming(const Joints& start,const Joints& goal,const Joints& lower,const Joints& upper,double speed=1.):start_(start),goal_(goal){
    if(!std::isfinite(speed)||speed<1.||speed>2.)throw std::invalid_argument("Homing speed outside [1,2]");
    double distance=0;
    for(int j=0;j<7;++j){
      if(!std::isfinite(start[j])||!std::isfinite(goal[j])||!std::isfinite(lower[j])||!std::isfinite(upper[j])||lower[j]>upper[j]
         ||start[j]<lower[j]||start[j]>upper[j]||goal[j]<lower[j]||goal[j]>upper[j])throw std::invalid_argument("Homing endpoint outside finite joint bounds");
      distance=std::max(distance,std::abs(goal[j]-start[j]));
    }
    // Analytic maxima of 10s^3-15s^4+6s^5 and its first three derivatives.
    duration_=std::max({3.,1.875*distance/velocity_limit,std::sqrt((10./std::sqrt(3.))*distance/acceleration_limit),std::cbrt(60.*distance/jerk_limit)});
    if(duration_>60.)throw std::invalid_argument("Homing trajectory exceeds 60 seconds");
    duration_/=speed; // At 2x: <=0.30 rad/s, 1 rad/s^2, 8 rad/s^3.
  }
  double duration() const{return duration_;}
  Reference at(double time) const{
    Reference out;
    if(time<=0){out.q=start_;return out;}
    if(time>=duration_){out.q=goal_;return out;}
    double s=time/duration_,s2=s*s,s3=s2*s;
    double p=s3*(10.+s*(-15.+6.*s)),v=30.*s2*(1.-s)*(1.-s)/duration_;
    double a=60.*s*(1.-s)*(1.-2.*s)/(duration_*duration_);
    double j=(60.-360.*s+360.*s2)/(duration_*duration_*duration_);
    for(int k=0;k<7;++k){double d=goal_[k]-start_[k];out.q[k]=start_[k]+d*p;out.dq[k]=d*v;out.ddq[k]=d*a;out.jerk[k]=d*j;}
    return out;
  }
 private:
  Joints start_,goal_;double duration_;
};
}

#pragma once
// Position-only hybrid EKF, raw-marker identity tracking, and release gate.
// This file has no SDK, JSON, file I/O, or robot-control dependency. All vectors
// in one instance use the same caller-selected Cartesian frame and time base.
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace juggling::ball {
using Vec3=Eigen::Vector3d;
using Mat3=Eigen::Matrix3d;
using Vec6=Eigen::Matrix<double,6,1>;
using Mat6=Eigen::Matrix<double,6,6>;
using Vec12=Eigen::Matrix<double,12,1>;
using Mat12=Eigen::Matrix<double,12,12>;
constexpr double max_ball_gap=.08;
inline Mat3 skew(const Vec3& v) { Mat3 m; m<<0.,-v.z(),v.y(),v.z(),0.,-v.x(),-v.y(),v.x(),0.; return m; }
inline void require(bool ok,const char* reason) { if(!ok) throw std::runtime_error(reason); }

template<class T,std::size_t N> class Ring {
 public:
  std::size_t size() const { return count_; }
  bool empty() const { return count_==0; }
  const T& operator[](std::size_t k) const { return data_[(first_+k)%N]; }
  T& operator[](std::size_t k) { return data_[(first_+k)%N]; }
  const T& front() const { return (*this)[0]; }
  const T& back() const { return (*this)[count_-1]; }
  void pop_front() { require(count_>0,"Empty history"); first_=(first_+1)%N; --count_; }
  void push_back(const T& x) { require(count_<N,"Bounded estimator history overflow"); data_[(first_+count_)%N]=x; ++count_; }
  void clear() { first_=count_=0; }
 private:
  std::array<T,N> data_{};
  std::size_t first_=0,count_=0;
};
struct ObserverConfig {
  double position_std=.001, drag=.34, quadratic_drag=0.;
  Vec3 gravity{0.,0.,-9.81};
  double acceleration_time_constant=1., acceleration_noise=.3,velocity_noise=.015,spin_noise=.1;
  double initial_velocity_std=.3,initial_spin_std=20.,initial_acceleration_std=.2;
  double restitution=.54,friction=.3,ball_mass=.0039,radius=.02,paddle_radius=.075,inertia_ratio=2./3.;
  double contact_delay=.001,impact_time_std=.001,impact_velocity_std=.15,impact_tangent_std=.04,impact_spin_std=1.;
  // Additive contact-velocity scatter, independent of the friction/spin impulse.
  double impact_independent_tangent_std=0.;
  double contact_measurement_std=.004,contact_measurement_window=.004,impact_refractory=.03;
  double innovation_gate=25.,max_measurement_age=.1,history_seconds=.25;
};
struct ArmSample {
  double t=0.;
  Vec3 p=Vec3::Zero(),n=Vec3::UnitZ(),velocity=Vec3::Zero(),angular_velocity=Vec3::Zero();
  // [linear Jacobian; angular Jacobian] M^-1 [linear Jacobian; angular Jacobian]^T.
  Mat6 inverse_spatial_inertia=Mat6::Zero();
};
struct Estimate {
  Vec12 x=Vec12::Zero(); Mat12 P=Mat12::Zero();
  double time=0.,last_impact=-std::numeric_limits<double>::infinity(),measurement_age=0.;
  bool stale=true;
};
struct UpdateInfo {
  bool accepted=false,near_impact=false;
  const char* reason="uninitialized";
  double capture_time=0.,arrival_time=0.,nis=0.;
  Vec3 innovation=Vec3::Zero();
};

class BallObserver {
 public:
  explicit BallObserver(const ObserverConfig& config={}):cfg(config) {
    for(double value : {cfg.position_std,cfg.drag,cfg.quadratic_drag,cfg.acceleration_time_constant,cfg.acceleration_noise,cfg.velocity_noise,cfg.spin_noise,cfg.initial_velocity_std,cfg.initial_spin_std,cfg.initial_acceleration_std,cfg.restitution,cfg.friction,cfg.ball_mass,cfg.radius,cfg.paddle_radius,cfg.inertia_ratio,cfg.contact_delay,cfg.impact_time_std,cfg.impact_velocity_std,cfg.impact_tangent_std,cfg.impact_spin_std,cfg.impact_independent_tangent_std,cfg.contact_measurement_std,cfg.contact_measurement_window,cfg.impact_refractory,cfg.innovation_gate,cfg.max_measurement_age,cfg.history_seconds}) require(std::isfinite(value)&&value>=0.,"Nonfinite or negative observer parameter");
    require(cfg.gravity.allFinite()&&cfg.gravity.z()<0.&&std::isfinite(cfg.drag)&&cfg.drag>=0.&&std::isfinite(cfg.quadratic_drag)&&cfg.quadratic_drag>=0.,"Invalid observer flight model");
    require(cfg.position_std>0.&&cfg.acceleration_time_constant>0.&&cfg.history_seconds>cfg.max_measurement_age&&cfg.max_measurement_age>0.&&cfg.contact_measurement_window>0.&&cfg.contact_measurement_std>0.&&cfg.innovation_gate>0.,"Invalid observer noise/history");
    require(cfg.radius>0.&&cfg.paddle_radius>cfg.radius&&cfg.ball_mass>0.&&cfg.inertia_ratio>0.&&cfg.restitution>=0.&&cfg.friction>=0.&&cfg.impact_time_std>=0.,"Invalid observer contact model");
  }
  ObserverConfig cfg;
  bool initialized() const { return initialized_; }
  double posterior_time() const { return state_.time; }
  double latest_arm_time() const { require(!history_.empty(),"Missing arm history");return history_.back().t; }
  std::size_t impacts() const { return impacts_; }
  void record_arm(const ArmSample& input) {
    ArmSample a=input;
    require(std::isfinite(a.t)&&a.p.allFinite()&&a.n.allFinite()&&a.velocity.allFinite()&&a.angular_velocity.allFinite()&&a.inverse_spatial_inertia.allFinite()&&a.n.norm()>.5,"Invalid observer arm sample");
    require(history_.empty()||a.t>history_.back().t,"Arm timestamps must increase");
    a.n.normalize();history_.push_back(a);
    const double cutoff=a.t-cfg.history_seconds;
    if(initialized_&&state_.time<cutoff) commit_advance(cutoff);
    while(history_.size()>2&&history_[1].t<cutoff) history_.pop_front();
  }
  void initialize(const Vec3& position,double t,const Vec3& velocity=Vec3::Zero()) {
    require(position.allFinite()&&velocity.allFinite()&&std::isfinite(t),"Invalid observer initialization");
    require(!history_.empty()&&t>=history_.front().t-1e-9&&t<=history_.back().t+1e-9,"Initialize requires bracketing arm history");
    state_=Estimate{};state_.time=t;state_.x.template head<3>()=position;state_.x.template segment<3>(3)=velocity;
    const std::array<double,4> stds{cfg.position_std,cfg.initial_velocity_std,cfg.initial_spin_std,cfg.initial_acceleration_std};
    for(int k=0;k<4;++k)state_.P.template block<3,3>(3*k,3*k)=Mat3::Identity()*stds[k]*stds[k];
    last_accepted_=t;last_capture_=-std::numeric_limits<double>::infinity();initialized_=true;impacts_=0;
  }
  UpdateInfo update_position(double capture_time,const Vec3& z,double arrival_time) {
    UpdateInfo out;out.capture_time=capture_time;out.arrival_time=arrival_time;
    if(!initialized_)out.reason="uninitialized";
    else if(!std::isfinite(capture_time)||!std::isfinite(arrival_time))out.reason="invalid timestamp";
    else if(!z.allFinite())out.reason="invalid position";
    else if(capture_time>arrival_time+1e-10)out.reason="future timestamp";
    else if(history_.empty()||arrival_time>history_.back().t+1e-10)out.reason="missing arm history";
    else if(capture_time<=last_capture_+1e-10||capture_time<state_.time-1e-10)out.reason="old or duplicate timestamp";
    else if(arrival_time-capture_time>cfg.max_measurement_age+1e-10)out.reason="measurement too old";
    else {
      commit_advance(capture_time);last_capture_=capture_time;
      out.innovation=z-state_.x.template head<3>();
      out.near_impact=std::abs(capture_time-state_.last_impact)<cfg.contact_measurement_window;
      const double std=out.near_impact?cfg.contact_measurement_std:cfg.position_std;
      const Mat3 R=std*std*Mat3::Identity(),S=state_.P.template topLeftCorner<3,3>()+R;
      Eigen::LDLT<Mat3> solver(S);require(solver.info()==Eigen::Success&&solver.isPositive(),"Observer innovation covariance invalid");
      out.nis=out.innovation.dot(solver.solve(out.innovation));
      if(std::isfinite(out.nis)&&out.nis<=cfg.innovation_gate) {
        const Eigen::Matrix<double,12,3> K=solver.solve(state_.P.template topRows<3>()).transpose();
        state_.x+=K*out.innovation;Mat12 A=Mat12::Identity();A.template leftCols<3>()-=K;
        state_.P=(A*state_.P*A.transpose()+K*R*K.transpose()).eval();state_.P=.5*(state_.P+state_.P.transpose()).eval();
        last_accepted_=capture_time;out.accepted=true;out.reason="accepted";
      } else out.reason="innovation gate";
    }
    return out;
  }
  Estimate estimate(double t) const {
    auto result=advance(t);result.measurement_age=t-last_accepted_;result.stale=result.measurement_age>cfg.max_measurement_age;return result;
  }
  bool contact_pending(double t) const {
    if(history_.size()<2)return false;const auto e=estimate(t);
    const auto geo=geometry(e.x,history_[history_.size()-2],history_.back(),t);
    return geo.radial<cfg.paddle_radius&&((t-e.last_impact>=0.&&t-e.last_impact<.004)||(geo.gap>-.01&&geo.gap<.008&&geo.speed<-.2));
  }
  // Public for deterministic parity checks and nominal landing prediction.
  Vec12 free(const Vec12& x,double dt) const { return free_state(x,dt,cfg); }
  static Vec12 free_state(const Vec12& x,double dt,const ObserverConfig& cfg) {
    require(std::isfinite(dt)&&std::abs(dt)<=4.,"Invalid observer integration interval");
    if(cfg.quadratic_drag==0.) {
      const double d=cfg.drag,k=1./cfg.acceleration_time_constant,ed=std::exp(-d*dt),ek=std::exp(-k*dt);
      const double vd=d?-std::expm1(-d*dt)/d:dt,pd=d?(dt-vd)/d:.5*dt*dt;
      const double va=std::abs(d-k)>1e-9?(ek-ed)/(d-k):dt*ed;
      const double pa=std::abs(d-k)>1e-9?((1.-ek)/k-vd)/(d-k):(vd-dt*ed)/d;
      Vec12 y=x;y.template head<3>()+=vd*x.template segment<3>(3)+pd*cfg.gravity+pa*x.template tail<3>();
      y.template segment<3>(3)=ed*x.template segment<3>(3)+vd*cfg.gravity+va*x.template tail<3>();y.template tail<3>()=ek*x.template tail<3>();return y;
    }
    const int count=std::max(1,int(std::ceil(std::abs(dt)/.02)));const double h=dt/count;Vec12 y=x;
    const auto rhs=[&](const Vec12& state){ Vec12 value=Vec12::Zero();const Vec3 v=state.template segment<3>(3);value.template head<3>()=v;value.template segment<3>(3)=cfg.gravity-(cfg.drag+cfg.quadratic_drag*std::sqrt(v.squaredNorm()+1e-10))*v+state.template tail<3>();value.template tail<3>()=-state.template tail<3>()/cfg.acceleration_time_constant;return value;};
    for(int i=0;i<count;++i){const Vec12 a=rhs(y),b=rhs(y+.5*h*a),c=rhs(y+.5*h*b),d=rhs(y+h*c);y+=h/6.*(a+2.*b+2.*c+d);}return y;
  }
 private:
  struct Flight {Vec12 x;Mat12 F,Q;};
  struct Geometry {double gap,radial,speed;};
  struct Hit {Vec12 x;double time=0.;Vec3 n=Vec3::UnitZ();bool occurred=false;Vec12 post_impact=Vec12::Zero();};
  Ring<ArmSample,1024> history_;
  Estimate state_;
  bool initialized_=false;
  double last_capture_=-std::numeric_limits<double>::infinity(),last_accepted_=-std::numeric_limits<double>::infinity();
  std::size_t impacts_=0;
  Vec12 rhs(const Vec12& x) const {
    Vec12 result=Vec12::Zero();const Vec3 v=x.template segment<3>(3);const double speed=std::sqrt(v.squaredNorm()+1e-10);
    result.template head<3>()=v;result.template segment<3>(3)=cfg.gravity-(cfg.drag+cfg.quadratic_drag*speed)*v+x.template tail<3>();result.template tail<3>()=-x.template tail<3>()/cfg.acceleration_time_constant;return result;
  }
  Mat12 jacobian(const Vec12& x) const {
    Mat12 A=Mat12::Zero();const Vec3 v=x.template segment<3>(3);const double speed=std::sqrt(v.squaredNorm()+1e-10);
    A.template block<3,3>(0,3)=Mat3::Identity();A.template block<3,3>(3,3)=-(cfg.drag+cfg.quadratic_drag*speed)*Mat3::Identity()-cfg.quadratic_drag/speed*(v*v.transpose());
    A.template block<3,3>(3,9)=Mat3::Identity();A.template block<3,3>(9,9)=-Mat3::Identity()/cfg.acceleration_time_constant;return A;
  }
  Flight flight(const Vec12& x,double dt) const {
    Flight f{x,Mat12::Identity(),Mat12::Zero()};Mat12 L=Mat12::Zero();
    L.template block<3,3>(3,3)=Mat3::Identity()*cfg.velocity_noise*cfg.velocity_noise;
    L.template block<3,3>(6,6)=Mat3::Identity()*cfg.spin_noise*cfg.spin_noise;
    L.template block<3,3>(9,9)=Mat3::Identity()*cfg.acceleration_noise*cfg.acceleration_noise;
    const int count=std::max(1,int(std::ceil(dt/.02)));const double h=dt/count;
    auto derivative=[&](const Flight& v)->Flight{const Mat12 A=jacobian(v.x);return {rhs(v.x),A*v.F,A*v.Q+v.Q*A.transpose()+L};};
    auto shifted=[](const Flight& a,const Flight& b,double scale)->Flight{return {a.x+scale*b.x,a.F+scale*b.F,a.Q+scale*b.Q};};
    for(int i=0;i<count;++i){const Flight a=derivative(f),b=derivative(shifted(f,a,.5*h)),c=derivative(shifted(f,b,.5*h)),d=derivative(shifted(f,c,h));f.x+=h/6.*(a.x+2*b.x+2*c.x+d.x);f.F+=h/6.*(a.F+2*b.F+2*c.F+d.F);f.Q+=h/6.*(a.Q+2*b.Q+2*c.Q+d.Q);}
    // Exact linear-drag mean avoids cancellation drift and matches Python's
    // closed form. RK4 F/Q error on a <=20ms interval is well below sensor noise.
    if(cfg.quadratic_drag==0.)f.x=free(x,dt);
    f.Q=.5*(f.Q+f.Q.transpose()).eval();return f;
  }
  static ArmSample arm(const ArmSample& a,const ArmSample& b,double t) {
    const double u=(t-a.t)/(b.t-a.t);ArmSample s;s.t=t;s.p=a.p+u*(b.p-a.p);s.n=(a.n+u*(b.n-a.n)).normalized();
    s.velocity=a.velocity+u*(b.velocity-a.velocity);s.angular_velocity=a.angular_velocity+u*(b.angular_velocity-a.angular_velocity);s.inverse_spatial_inertia=a.inverse_spatial_inertia+u*(b.inverse_spatial_inertia-a.inverse_spatial_inertia);return s;
  }
  Geometry geometry(const Vec12& x,const ArmSample& a,const ArmSample& b,double t) const {
    const auto s=arm(a,b,t);const Vec3 rel=x.template head<3>()-s.p;const double duration=b.t-a.t,u=(t-a.t)/duration;
    Vec3 ndot=(b.n-a.n)/duration;ndot=(ndot-s.n*s.n.dot(ndot))/(a.n+u*(b.n-a.n)).norm();
    return {s.n.dot(rel)-cfg.radius,(rel-s.n*s.n.dot(rel)).norm(),s.n.dot(x.template segment<3>(3)-(b.p-a.p)/duration)+rel.dot(ndot)};
  }
  Hit hit_map(const Vec12& x,double dt,const ArmSample& a,const ArmSample& b,double t,double hit
#ifdef JUGGLING_CONDITIONED_CONTACT_JACOBIAN
              ,bool conditioned=false,double event_level=0.
#endif
              ) const {
#ifdef JUGGLING_CONDITIONED_CONTACT_JACOBIAN
    if(conditioned){
      // Differentiate the already selected impact mode. A micron state
      // perturbation can put the same event just outside this arm interval;
      // clipping it and returning free flight would differentiate a mode jump.
      // Continue the local event surface through the nominal hit instead. The
      // level offset also anchors a nominal hit clamped at the interval start,
      // without searching far backwards for an already penetrated surface.
      const double nominal_hit=hit;
      for(int k=0;k<4;++k){
        const auto geo=geometry(free(x,hit),a,b,t+hit);
        hit-=(geo.gap+std::max(-geo.speed,0.)*cfg.contact_delay-event_level)/std::min(geo.speed,-.05);
        require(std::isfinite(hit)&&std::abs(hit-nominal_hit)<=.001,"Contact Jacobian left local event neighborhood");
      }
    }else{
#endif
    hit=std::clamp(hit,0.,dt);
    for(int k=0;k<3;++k){const auto geo=geometry(free(x,hit),a,b,t+hit);hit=std::clamp(hit-(geo.gap+std::max(-geo.speed,0.)*cfg.contact_delay)/std::min(geo.speed,-.05),0.,dt);}
#ifdef JUGGLING_CONDITIONED_CONTACT_JACOBIAN
    }
#endif
    auto at=free(x,hit);const auto geo=geometry(at,a,b,t+hit);
    if(
#ifdef JUGGLING_CONDITIONED_CONTACT_JACOBIAN
       !conditioned&&
#endif
       geo.gap+std::max(-geo.speed,0.)*cfg.contact_delay>1e-8)return {free(x,dt),0.,Vec3::UnitZ(),false};
    const auto s=arm(a,b,t+hit);const Vec3 rel=at.template head<3>()-s.p,tangent=rel-s.n*s.n.dot(rel);
    const Vec3 relative=at.template segment<3>(3)-at.template segment<3>(6).cross(cfg.radius*s.n)-s.velocity-s.angular_velocity.cross(tangent);
    const double vn=s.n.dot(relative);if(vn>=0.){
#ifdef JUGGLING_CONDITIONED_CONTACT_JACOBIAN
      require(!conditioned,"Contact Jacobian perturbation became nonapproaching");
#endif
      return {free(x,dt),0.,s.n,false};
    }
    Eigen::Matrix<double,3,6> G;G.template leftCols<3>()=Mat3::Identity();G.template rightCols<3>()=-skew(tangent);
    const Mat3 K=Mat3::Identity()+(Mat3::Identity()-s.n*s.n.transpose())/cfg.inertia_ratio+cfg.ball_mass*G*s.inverse_spatial_inertia*G.transpose();
    Eigen::Matrix<double,3,2> T;T.col(0)=s.n.unitOrthogonal();T.col(1)=s.n.cross(T.col(0));
    const Eigen::Matrix2d tangential=T.transpose()*K*T;const auto factor=tangential.ldlt();require(factor.isPositive(),"Invalid paddle inverse inertia");
    auto impulse=[&](double j)->Vec3{const Eigen::Vector2d desired=-factor.solve(T.transpose()*(relative+K*s.n*j));const double cap=cfg.friction*j;return j*s.n+T*(desired*cap/std::sqrt(cap*cap+desired.squaredNorm()+1e-10));};
    double lo=0.,hi=std::max(1.,-3*vn);auto residual=[&](double j){return s.n.dot(K*impulse(j))+(1+cfg.restitution)*vn;};
    for(int k=0;k<8&&residual(hi)<0.;++k)hi*=2.;require(residual(hi)>=0.,"Cannot bracket ball impulse");
    for(int k=0;k<48;++k){const double mid=.5*(lo+hi);if(residual(mid)>0.)hi=mid;else lo=mid;}
    const Vec3 u=impulse(.5*(lo+hi));at.template segment<3>(3)+=u;at.template segment<3>(6)-=s.n.cross(u)/(cfg.inertia_ratio*cfg.radius);
    // The conditioned derivative may have hit>dt: this short signed flight
    // compares the same impact branch at the common original report time.
    return {free(at,dt-hit),hit,s.n,true,at};
  }
  void step(Estimate& s,const ArmSample& a,const ArmSample& b,double end) const {
    const double dt=end-s.time;const auto f=flight(s.x,dt);const auto ga=geometry(s.x,a,b,s.time),gb=geometry(f.x,a,b,end);
    const double ea=ga.gap+std::max(-ga.speed,0.)*cfg.contact_delay,eb=gb.gap+std::max(-gb.speed,0.)*cfg.contact_delay;
    if(s.time-s.last_impact>cfg.impact_refractory&&eb<=0.&&std::max(ga.radial,gb.radial)<cfg.paddle_radius&&gb.speed<-.2&&ga.gap>-.02){
      const auto h=hit_map(s.x,dt,a,b,s.time,dt*std::clamp(ea/std::max(ea-eb,1e-12),0.,1.));
      if(h.occurred){
#ifdef JUGGLING_CONDITIONED_CONTACT_JACOBIAN
        const auto event_geo=geometry(free(s.x,h.time),a,b,s.time+h.time);
        const double event_level=event_geo.gap+std::max(-event_geo.speed,0.)*cfg.contact_delay;
        Mat12 D;for(int j=0;j<12;++j){const double eps=j>=6&&j<9?2e-5:2e-6;Vec12 delta=Vec12::Zero();delta[j]=eps;D.col(j)=(hit_map(s.x+delta,dt,a,b,s.time,h.time,true,event_level).x-hit_map(s.x-delta,dt,a,b,s.time,h.time,true,event_level).x)/(2*eps);}
#else
        Mat12 D;for(int j=0;j<12;++j){const double eps=j>=6&&j<9?2e-5:2e-6;Vec12 delta=Vec12::Zero();delta[j]=eps;D.col(j)=(hit_map(s.x+delta,dt,a,b,s.time,h.time).x-hit_map(s.x-delta,dt,a,b,s.time,h.time).x)/(2*eps);}
#endif
        const Mat3 tangent=Mat3::Identity()-h.n*h.n.transpose(),impulse_cov=cfg.impact_velocity_std*cfg.impact_velocity_std*(h.n*h.n.transpose())+cfg.impact_tangent_std*cfg.impact_tangent_std*tangent;
        Eigen::Matrix<double,12,3> G=Eigen::Matrix<double,12,3>::Zero();G.template block<3,3>(3,0)=Mat3::Identity();G.template block<3,3>(6,0)=-skew(h.n)/(cfg.inertia_ratio*cfg.radius);
        s.P=(D*s.P*D.transpose()+f.Q+G*impulse_cov*G.transpose()).eval();Vec12 timing=Vec12::Zero();timing.template head<3>()=h.x.template segment<3>(3)-f.x.template segment<3>(3);
        if(cfg.impact_independent_tangent_std>0.){
          // This perturbation is a ball-only kick AFTER the frictional reset:
          // no spin reaction or velocity-spin covariance is injected. Carry
          // it from the event to this report time, including its position
          // covariance and position-velocity cross covariance. The flight
          // Jacobian also handles the optional quadratic-drag model.
          const auto post=flight(h.post_impact,std::max(0.,dt-h.time));
          const Eigen::Matrix<double,12,3> injection=post.F.template block<12,3>(0,3);
          s.P+=cfg.impact_independent_tangent_std*cfg.impact_independent_tangent_std
            *injection*tangent*injection.transpose();
        }
        s.P+=cfg.impact_time_std*cfg.impact_time_std*timing*timing.transpose();s.P.template block<3,3>(6,6)+=cfg.impact_spin_std*cfg.impact_spin_std*tangent;
        s.P=.5*(s.P+s.P.transpose()).eval();s.x=h.x;s.last_impact=s.time+h.time;s.time=end;return;
      }
    }
    s.x=f.x;s.P=(f.F*s.P*f.F.transpose()+f.Q).eval();s.P=.5*(s.P+s.P.transpose()).eval();s.time=end;
  }
  Estimate advance(double end) const {
    require(initialized_,"Initialize observer before prediction");require(!history_.empty()&&std::isfinite(end)&&end>=state_.time-1e-10&&end<=history_.back().t+1e-10,"Prediction requires causal arm history");
    require(state_.time>=history_.front().t-1e-10,"Measurement gap exceeds retained arm history");Estimate result=state_;
    for(std::size_t k=1;k<history_.size()&&result.time<end-1e-10;++k){const auto& a=history_[k-1];const auto& b=history_[k];if(b.t<=result.time+1e-10)continue;step(result,a,b,std::min(b.t,end));}
    require(result.x.allFinite()&&result.P.allFinite(),"Nonfinite ball estimate");return result;
  }
  void commit_advance(double end){const double old=state_.last_impact;state_=advance(end);if(state_.last_impact!=old)++impacts_;}
};

struct Marker {std::uint64_t id=0;Vec3 position=Vec3::Zero();};
struct AssociationPrediction {Vec6 x=Vec6::Zero();Mat6 P=Mat6::Zero();double position_std=.001;};
struct AssociationInfo {bool accepted=false;const char* reason="acquiring";double time=0.,nis=0.;std::size_t candidates=0;std::uint64_t marker_id=0;Vec3 prediction=Vec3::Zero(),residual=Vec3::Zero();};
struct RawBallConfig {double acquisition_seconds=3.,lateral=.35,min_height=.35,max_height=3.,max_gap=max_ball_gap,position_std=.001;};
struct RawEstimate {Vec6 x;Mat6 P;double measurement_age;bool stale;};
struct RawSelection {Vec3 position=Vec3::Zero();double coverage=0.,span=0.,scatter_p95=0.,lateral=0.,height=0.;std::size_t samples=0,eligible_candidates=0;};

class RawBall {
 public:
  explicit RawBall(const RawBallConfig& config={}):cfg(config){require(cfg.acquisition_seconds>0.&&cfg.max_gap>0.&&cfg.max_gap<=.1&&cfg.position_std>0.,"Invalid raw-ball configuration");}
  RawBallConfig cfg;bool allow_motion=false;Vec3 acceleration=Vec3::Zero();AssociationInfo last_association;
  bool selected() const{return selection_.has_value();}const std::optional<RawSelection>& selection()const{return selection_;}
  const Vec3& position()const{return position_;}double seen_time()const{return seen_time_;}
  const std::string& fault()const{return fault_;}
  RawEstimate estimate(double t)const {require(selected(),"Raw ball not acquired");const auto prediction=predict(t);return {prediction.x,prediction.P,t-seen_time_,!fault_.empty()||t-seen_time_>cfg.max_gap};}
  std::optional<Vec3> update(std::uint64_t frame,double t,const std::vector<Marker>& points,const std::optional<Vec3>& face,const std::optional<AssociationPrediction>& supplied=std::nullopt) {
    if(!fault_.empty())fail(fault_.c_str());if(!selected()&&!face)return std::nullopt;if(!std::isfinite(t))fail("Invalid raw marker timestamp");
    if(have_frame_&&(frame<=last_frame_||t<=last_time_)){if(selected())fail("Raw marker frame/time restarted or duplicated");frames_.clear();tracks_.clear();}
    have_frame_=true;last_frame_=frame;last_time_=t;for(const auto& p:points)if(!p.position.allFinite())fail("Invalid unlabeled marker position");
    if(selected()) {
      if(t-seen_time_>cfg.max_gap+1e-9)fail("Selected raw ball tracking lost; reacquisition requires a new session");
      AssociationPrediction p=supplied?*supplied:predict(t);require(p.x.allFinite()&&p.P.allFinite()&&std::isfinite(p.position_std)&&p.position_std>0.,"Invalid association prediction");
      const Mat3 R=Mat3::Identity()*p.position_std*p.position_std,S=p.P.template topLeftCorner<3,3>()+R;const auto solver=S.ldlt();require(solver.isPositive(),"Invalid association covariance");
      last_association={};last_association.time=t;last_association.prediction=p.x.template head<3>();const Marker* chosen=nullptr;
      for(const auto& candidate:points){const Vec3 residual=candidate.position-p.x.template head<3>();if(residual.norm()>.020||(!allow_motion&&(candidate.position-selection_->position).norm()>.030))continue;
        const double nis=residual.dot(solver.solve(residual));if(nis<=25.){++last_association.candidates;chosen=&candidate;last_association.nis=nis;last_association.residual=residual;}}
      if(last_association.candidates!=1){last_association.reason=last_association.candidates?"ambiguous":"missing_or_outlier";return std::nullopt;}
      const Eigen::Matrix<double,6,3> K=solver.solve(p.P.template topRows<3>()).transpose();x_=p.x+K*last_association.residual;Mat6 A=Mat6::Identity();A.template leftCols<3>()-=K;P_=(A*p.P*A.transpose()+K*R*K.transpose()).eval();P_=.5*(P_+P_.transpose()).eval();
      filter_time_=t;position_=chosen->position;seen_time_=t;last_association.accepted=true;last_association.reason="accepted";last_association.marker_id=chosen->id;return position_;
    }
    const double cutoff=t-cfg.acquisition_seconds-.025;while(!frames_.empty()&&frames_.front()<cutoff)frames_.pop_front();frames_.push_back(t);
    for(auto& tr:tracks_)while(!tr.observations.empty()&&tr.observations.front().t<cutoff)tr.observations.pop_front();
    tracks_.erase(std::remove_if(tracks_.begin(),tracks_.end(),[](const Track& tr){return tr.observations.empty();}),tracks_.end());
    std::array<bool,64> used{};
    for(const auto& candidate:points){const Vec3 d=candidate.position-*face;if(d.template head<2>().norm()>cfg.lateral||d.z()<cfg.min_height||d.z()>cfg.max_height)continue;
      std::size_t best=tracks_.size();double distance=.012;for(std::size_t k=0;k<tracks_.size();++k){const double v=(candidate.position-tracks_[k].center).norm();if(v<distance){distance=v;best=k;}}
      if(best==tracks_.size()){if(tracks_.size()>=64)fail("Too many stationary marker candidates");tracks_.emplace_back();tracks_.back().center=candidate.position;}else{if(used[best])fail("Multiple raw points occupy one stationary candidate");tracks_[best].center=.9*tracks_[best].center+.1*candidate.position;}
      used[best]=true;tracks_[best].observations.push_back({t,candidate.position});
    }
    if(frames_.size()<30||t-frames_.front()<cfg.acquisition_seconds)return std::nullopt;
    std::vector<Candidate> candidates;
    for(const auto& tr:tracks_){const auto& observations=tr.observations;if(observations.empty())continue;const Vec3 center=median(observations);std::vector<double> radii;radii.reserve(observations.size());for(std::size_t k=0;k<observations.size();++k)radii.push_back((observations[k].p-center).norm());std::sort(radii.begin(),radii.end());
      const double coverage=double(observations.size())/frames_.size(),span=observations.back().t-observations.front().t,percentile=quantile(radii,.95);const Vec3 delta=center-*face;
      if(coverage>=.95&&span>=cfg.acquisition_seconds-.03&&t-observations.back().t<=.02&&percentile<=.004&&radii.back()<=.008)candidates.push_back({{center,coverage,span,percentile,delta.template head<2>().norm(),delta.z(),observations.size()},observations.back()});}
    // Match Python: persistence/stationarity qualify candidates, then XY
    // proximity wins. Coverage/scatter/position only break exact distance ties.
    std::sort(candidates.begin(),candidates.end(),[](const Candidate& a,const Candidate& b){
      const auto& x=a.selection;const auto& y=b.selection;
      if(x.lateral!=y.lateral)return x.lateral<y.lateral;
      if(x.coverage!=y.coverage)return x.coverage>y.coverage;
      if(x.scatter_p95!=y.scatter_p95)return x.scatter_p95<y.scatter_p95;
      for(int j=0;j<3;++j)if(x.position[j]!=y.position[j])return x.position[j]<y.position[j];return false;
    });
    if(candidates.empty())return std::nullopt;
    const auto& best=candidates[0];selection_=best.selection;selection_->eligible_candidates=candidates.size();position_=best.last.p;seen_time_=filter_time_=best.last.t;x_.setZero();x_.template head<3>()=position_;P_.setZero();P_.template topLeftCorner<3,3>()=Mat3::Identity()*cfg.position_std*cfg.position_std;P_.template bottomRightCorner<3,3>()=Mat3::Identity()*.03*.03;tracks_.clear();frames_.clear();return position_;
  }
 private:
  struct Observation{double t=0.;Vec3 p=Vec3::Zero();};
  struct Track{Vec3 center=Vec3::Zero();Ring<Observation,1024> observations;};
  struct Candidate{RawSelection selection;Observation last;};
  Ring<double,1024> frames_;std::vector<Track> tracks_;std::optional<RawSelection> selection_;
  Vec3 position_=Vec3::Zero();Vec6 x_=Vec6::Zero();Mat6 P_=Mat6::Zero();double seen_time_=0.,filter_time_=0.,last_time_=0.;std::uint64_t last_frame_=0;bool have_frame_=false;std::string fault_;
  [[noreturn]] void fail(const char* text){if(fault_.empty())fault_=text;throw std::runtime_error(fault_);}
  AssociationPrediction predict(double t)const {require(std::isfinite(t)&&t>=filter_time_-1e-9,"Cannot predict raw ball backwards");const double dt=std::max(0.,t-filter_time_);Mat6 F=Mat6::Identity();F.template topRightCorner<3,3>()=dt*Mat3::Identity();Mat6 Q=Mat6::Zero();Q.template topLeftCorner<3,3>()=4*dt*dt*dt/3*Mat3::Identity();Q.template topRightCorner<3,3>()=Q.template bottomLeftCorner<3,3>()=2*dt*dt*Mat3::Identity();Q.template bottomRightCorner<3,3>()=4*dt*Mat3::Identity();Vec6 x=F*x_;x.template head<3>()+=.5*dt*dt*acceleration;x.template tail<3>()+=dt*acceleration;return {x,F*P_*F.transpose()+Q,cfg.position_std};}
  static double quantile(const std::vector<double>& x,double u){const double k=u*(x.size()-1);const auto i=std::size_t(k);return x[i]+(k-i)*(x[std::min(i+1,x.size()-1)]-x[i]);}
  static Vec3 median(const Ring<Observation,1024>& observations){Vec3 result;std::vector<double> values;values.reserve(observations.size());for(int axis=0;axis<3;++axis){values.clear();for(std::size_t k=0;k<observations.size();++k)values.push_back(observations[k].p[axis]);std::sort(values.begin(),values.end());result[axis]=quantile(values,.5);}return result;}
};

struct ReleaseConfig {double stable_seconds=.6,max_gap=max_ball_gap,stationary_radius=.004,drop=.008,downward_speed=.15;int confirm_frames=3;double max_lateral=.025;};
struct Release {double capture_time=0.;Vec3 position=Vec3::Zero(),velocity=Vec3::Zero();double observed_drop=0.;std::uint64_t frame=0;};
class ReleaseGate {
 public:
  explicit ReleaseGate(const ReleaseConfig& config={},const Vec3& gravity=Vec3(0.,0.,-9.81),const std::optional<Vec3>& anchor=std::nullopt):cfg(config),gravity_(gravity),anchor_(anchor){require(gravity.allFinite()&&gravity.z()<0.,"Invalid release gravity");}
  ReleaseConfig cfg;
  const std::optional<Vec3>& anchor()const{return anchor_;}bool armed()const{return armed_;}const std::optional<Release>& released()const{return released_;}
  void missing(double now){if(have_frame_&&now-last_time_>cfg.max_gap)fail("Ball tracking lost; release is inhibited");}
  void arm(){require(fault_.empty()&&anchor_.has_value(),"No stable fixture position");armed_=true;prearm_motion_frames_=0;}
  std::optional<Release> update(double t,const Vec3& p,std::uint64_t frame) {
    if(!fault_.empty())fail(fault_.c_str());if(!p.allFinite()||!std::isfinite(t))fail("Invalid ball observation");if(have_frame_&&(frame<=last_frame_||t<=last_time_))return std::nullopt;
    if(have_frame_&&t-last_time_>cfg.max_gap)fail("Ball frame gap; release is inhibited");have_frame_=true;last_frame_=frame;last_time_=t;
    while(!history_.empty()&&history_.front().t<t-std::max(.8,cfg.stable_seconds+.1))history_.pop_front();history_.push_back({t,p});
    if(!anchor_){std::vector<Observation> recent;for(std::size_t k=0;k<history_.size();++k)if(history_[k].t>=t-cfg.stable_seconds-.02)recent.push_back(history_[k]);
      if(recent.size()>=30&&recent.back().t-recent.front().t>=cfg.stable_seconds){Vec3 center;std::vector<double> values;for(int axis=0;axis<3;++axis){values.clear();for(const auto& a:recent)values.push_back(a.p[axis]);std::sort(values.begin(),values.end());const auto m=values.size()/2;center[axis]=values.size()%2?values[m]:.5*(values[m-1]+values[m]);}double radius=0.;for(const auto& a:recent)radius=std::max(radius,(a.p-center).norm());if(radius<=cfg.stationary_radius)anchor_=center;}return std::nullopt;}
    const Vec3 drift=p-*anchor_;if(released_)return released_;if(!armed_){
      // Vicon can report an isolated ~6.4 mm lateral jump for the same held
      // marker, then return on the next frame. Require persistent movement,
      // just as actual release detection requires consecutive falling frames.
      prearm_motion_frames_=drift.norm()>.015?prearm_motion_frames_+1:0;
      if(prearm_motion_frames_>=cfg.confirm_frames)fail("Ball moved/released before robot was ready");
      return std::nullopt;
    }
    if(drift.template head<2>().norm()>cfg.max_lateral||drift.z()>.008)fail("Fixture moved or ball pushed outside the release corridor");
    const auto velocity=fit(t,.06,false).second;const bool falling=drift.z()<-cfg.drop&&velocity.z()<-cfg.downward_speed;count_=falling?count_+1:0;
    if(count_>=cfg.confirm_frames){const auto pv=fit(t,.10,true);double residual=0.;int samples=0;for(std::size_t k=0;k<history_.size();++k){const auto& a=history_[k];if(a.t<t-.10||a.p.z()>=anchor_->z()-.003)continue;const double dt=a.t-t;residual=std::max(residual,(a.p-pv.first-dt*pv.second-.5*dt*dt*gravity_).norm());++samples;}
      if(samples<3||residual>.004)fail("Release observations do not fit a free fall");if(!(pv.second.z()>-.9&&pv.second.z()<-.15)||pv.second.template head<2>().norm()>.25)fail("Release velocity outside the supported local drop");released_=Release{t,pv.first,pv.second,-drift.z(),frame};return released_;}
    return std::nullopt;
  }
 private:
  struct Observation{double t=0.;Vec3 p=Vec3::Zero();};Ring<Observation,1024> history_;Vec3 gravity_;std::optional<Vec3> anchor_;std::optional<Release> released_;bool have_frame_=false,armed_=false;std::uint64_t last_frame_=0;double last_time_=0.;int count_=0,prearm_motion_frames_=0;std::string fault_;
  [[noreturn]] void fail(const char* text){if(fault_.empty())fault_=text;throw std::runtime_error(fault_);}
  std::pair<Vec3,Vec3> fit(double t,double seconds,bool fallen)const {Eigen::Matrix2d normal=Eigen::Matrix2d::Zero();Eigen::Matrix<double,2,3> rhs=Eigen::Matrix<double,2,3>::Zero();int count=0;
    for(std::size_t k=0;k<history_.size();++k){const auto& a=history_[k];if(a.t<t-seconds||(fallen&&a.p.z()>=anchor_->z()-.003))continue;const double dt=a.t-t;const Eigen::Vector2d row(1.,dt);const Vec3 p=fallen?(a.p-.5*dt*dt*gravity_).eval():a.p;normal+=row*row.transpose();rhs+=row*p.transpose();++count;}
    require(count>=2&&normal.determinant()>1e-12,"Insufficient release fit samples");const Eigen::Matrix<double,2,3> pv=normal.ldlt().solve(rhs);return {pv.row(0).transpose(),pv.row(1).transpose()};}
};

inline Vec3 predicted_landing(const Vec6& state,const ObserverConfig& cfg) {
  require(state.allFinite(),"Invalid landing prediction state");if(state.z()<=cfg.radius)return state.template head<3>();
  Vec12 initial=Vec12::Zero();initial.template head<6>()=state;double lo=0.,hi=1.;while(BallObserver::free_state(initial,hi,cfg).z()>cfg.radius){hi*=2.;require(hi<=4.,"No finite near-term landing");}
  for(int k=0;k<40;++k){const double t=.5*(lo+hi);if(BallObserver::free_state(initial,t,cfg).z()>cfg.radius)lo=t;else hi=t;}return BallObserver::free_state(initial,.5*(lo+hi),cfg).template head<3>();
}
} // namespace juggling::ball

// OFFLINE ONLY. Same session reference/velocity adapter as hardware, with a
// 5 ms illustrative velocity-servo lag and impulse ball. No robot SDK/network.
#include "controller.hpp"
#include "session_control.hpp"
#include "joint_velocity_tracking.hpp"
#include <cstdio>
#include <deque>
#include <Eigen/Geometry>
#include "ball_gravity.hpp"
#ifndef JUGGLING_BALL_DRAG
#define JUGGLING_BALL_DRAG .02
#endif
namespace {
namespace g=juggling::generated;
using namespace juggling;
struct Plant {
 RobotModel model;SessionControl control;JointVelocityTracking servo;
 State x;Joint acceleration=Joint::Zero(),command_acceleration=Joint::Zero();std::array<double,7> command{};
 double t=0,release=2,servo_time=0;int impacts=0;bool active=false;
 Joint filtered_torque=Joint::Zero();bool filter_initialized=false;
 std::array<double,4> motion_ratios{}; // velocity, acceleration, inverse-dynamics torque, filtered rate
 // Empirical parameters belong only to this offline plant. The generated MPC,
 // observer, reference adapter and arm rigid-body model remain unchanged.
 bool empirical=false;double drag=JUGGLING_BALL_DRAG,quadratic_drag=0,restitution=g::restitution,friction=g::friction;
 double radius=g::radius,mass=g::mass,inertia=g::inertia_ratio,residual_time_constant=0,restitution_edge_loss=0,restitution_reference_radius=.075;
 Eigen::Vector3d gravity{ball_gravity[0],ball_gravity[1],ball_gravity[2]},center_offset=Eigen::Vector3d::Zero();
 Eigen::Vector3d acceleration_residual=Eigen::Vector3d::Zero(),acceleration_target=Eigen::Vector3d::Zero(),impact_residual=Eigen::Vector3d::Zero();
 Eigen::Matrix3d normal_rotation=Eigen::Matrix3d::Identity();
 Eigen::Vector2d axes{g::paddle_radius,g::paddle_radius};
 Joint lag=Joint::Constant(.005),dead_time=Joint::Zero();
 std::deque<std::pair<double,Joint>> command_history;
 std::array<double,36> last_contact{};int crossings=0;
 Plant(const char* factors,const double* q,const double* ball,double at):model(factors),control(joints(q)),servo({}),release(at){
  x.setZero();x.head<3>()=Eigen::Map<const Eigen::Vector3d>(ball);x.segment<7>(6)=Eigen::Map<const Joint>(q);
 }
 static std::array<double,7> joints(const double* x){std::array<double,7> out;std::copy(x,x+7,out.begin());return out;}
 static Eigen::Matrix3d skew(const Eigen::Vector3d& v){Eigen::Matrix3d s;s<<0,-v.z(),v.y(),v.z(),0,-v.x(),-v.y(),v.x(),0;return s;}
 struct Face {Eigen::Vector3d p,n,local;Eigen::Matrix3d R;Matrix J;double gap,speed;bool inside;};
 Face face(const State& at){
  auto f=model.contact_fk(at.segment<7>(6));Face out;out.R=f[4]*normal_rotation;out.n=out.R.col(2);
  Eigen::Vector3d offset=f[4]*center_offset;out.p=f[0].col(0)+offset;
  Eigen::Vector3d rel=at.head<3>()+Eigen::Map<const Eigen::Vector3d>(g::origin.data())-out.p;
  out.local=out.R.transpose()*rel;out.gap=out.local.z()-radius;
  Eigen::Vector3d tangent=rel-out.n*out.local.z();out.J=f[2]-skew(offset+tangent)*f[3];
  out.speed=out.n.dot(at.segment<3>(3)-out.J*at.segment<7>(13));
  out.inside=std::pow(out.local.x()/axes.x(),2)+std::pow(out.local.y()/axes.y(),2)<1.;return out;
 }
 Joint delayed_command(double at){
  Joint result=Eigen::Map<const Joint>(command.data());
  if(!empirical)return result;
  for(int j=0;j<7;++j){result[j]=0.;for(auto it=command_history.rbegin();it!=command_history.rend();++it)
    if(it->first<=at-dead_time[j]+1e-10){result[j]=it->second[j];break;}}
  return result;
 }
 double impact(State& at,const Face& geo){
  auto decomposition=model.mass(at.segment<7>(6)).eval().ldlt();Matrix response=decomposition.solve(geo.J.transpose());
  Eigen::Vector3d relative=at.segment<3>(3)-at.segment<3>(20).cross(radius*geo.n)-geo.J*at.segment<7>(13);
  Matrix K=Matrix::Identity(3,3)+(Matrix::Identity(3,3)-geo.n*geo.n.transpose())/inertia+mass*geo.J*response;
  Matrix T=geo.R.leftCols(2);auto tangent_mass=(T.transpose()*K*T).eval().ldlt();double vn=geo.n.dot(relative);
  auto impulse=[&](double j)->Eigen::Vector3d{Eigen::Vector2d desired=-tangent_mass.solve(T.transpose()*(relative+K*geo.n*j));double cap=friction*j;
    return j*geo.n+T*(desired*cap/std::sqrt(cap*cap+desired.squaredNorm()+1e-10));};
  double contact_restitution=std::clamp(restitution-restitution_edge_loss*std::pow(geo.local.head<2>().norm()/restitution_reference_radius,4),0.,1.5);
  double lo=0,hi=std::max(1.,-3*vn);for(int k=0;k<50;++k){double mid=.5*(lo+hi);if(geo.n.dot(K*impulse(mid))+(1+contact_restitution)*vn>0)hi=mid;else lo=mid;}
  double j=.5*(lo+hi);Eigen::Vector3d u=impulse(j);at.segment<3>(3)+=u+geo.R*impact_residual;
  at.segment<3>(20)-=geo.n.cross(u)/(inertia*radius);at.head<3>()+=(1e-10-geo.gap)*geo.n;return mass*j;
 }
 void configure(const double* p,int count){
  if(count!=35||t!=0)throw std::runtime_error("Empirical plant requires 35 parameters before stepping");
  for(int k=0;k<count;++k)if(!std::isfinite(p[k]))throw std::runtime_error("Nonfinite empirical plant parameter");
  gravity=Eigen::Map<const Eigen::Vector3d>(p);drag=p[3];quadratic_drag=p[4];restitution=p[5];friction=p[6];radius=p[7];mass=p[8];inertia=p[9];
  Eigen::Vector3d rv=Eigen::Map<const Eigen::Vector3d>(p+10);double angle=rv.norm();normal_rotation=Eigen::Matrix3d::Identity();
  if(angle>0)normal_rotation=Eigen::AngleAxisd(angle,rv/angle).toRotationMatrix();
  center_offset=Eigen::Map<const Eigen::Vector3d>(p+13);axes=Eigen::Map<const Eigen::Vector2d>(p+16);
  lag=Eigen::Map<const Joint>(p+18);dead_time=Eigen::Map<const Joint>(p+25);residual_time_constant=p[32];
  restitution_edge_loss=p[33];restitution_reference_radius=p[34];
  if(gravity.z()>=0||drag<0||quadratic_drag<0||restitution<0||restitution>1.5||friction<0||radius<=0||mass<=0||inertia<=0||axes.minCoeff()<=0
    ||lag.minCoeff()<0||lag.maxCoeff()>.1||dead_time.minCoeff()<0||dead_time.maxCoeff()>.1||residual_time_constant<0||restitution_edge_loss<0||restitution_reference_radius<=0)
    throw std::runtime_error("Invalid empirical plant parameter range");
  empirical=true;
 }
 State free(const State& before,double dt,double start){
  State out=before;Joint target=delayed_command(start),delta=before.segment<7>(13)-target;
  for(int j=0;j<7;++j){double decay=lag[j]>0?std::exp(-dt/lag[j]):0.;out[6+j]+=dt*target[j]+lag[j]*(1-decay)*delta[j];out[13+j]=target[j]+decay*delta[j];}
  if(start>=release-1e-10){
   double tm=start+.5*dt;Eigen::Vector3d wind=gravity+acceleration_residual;
   if(!empirical)wind+=Eigen::Vector3d(.12*std::cos(.8*tm)+.04*std::sin(1.7*tm),.10*std::sin(.8*tm)+.03*std::cos(1.3*tm),0.);
   if(quadratic_drag==0){double d=std::exp(-drag*dt),p1=drag>0?-std::expm1(-drag*dt)/drag:dt,p2=drag>0?(dt-p1)/drag:.5*dt*dt;
    out.head<3>()+=p1*before.segment<3>(3)+p2*wind;out.segment<3>(3)=d*before.segment<3>(3)+p1*wind;
   }else{Eigen::Vector3d v=before.segment<3>(3);auto a=[&](const Eigen::Vector3d& w)->Eigen::Vector3d{return wind-(drag+quadratic_drag*w.norm())*w;};
    Eigen::Vector3d k1=a(v),k2=a(v+.5*dt*k1),k3=a(v+.5*dt*k2),k4=a(v+dt*k3);
    out.head<3>()+=dt*v+dt*dt*(k1+k2+k3)/6.;out.segment<3>(3)+=dt*(k1+2*k2+2*k3+k4)/6.;}
  }return out;
 }
 void step(){
  constexpr double dt=.0005;
  if(t>=servo_time-1e-10){auto previous=command;auto ref=control.stopped()?std::array<double,21>{}:control.reference(t);command=servo.step(ref,joints(x.data()+6),joints(x.data()+13),.001,control.stopped(),JointVelocityTracking::normal_stop_gain);
   for(int j=0;j<7;++j)command_acceleration[j]=(command[j]-previous[j])/.001;
   command_history.emplace_back(t,Eigen::Map<const Joint>(command.data()));while(command_history.size()>150)command_history.pop_front();servo_time+=.001;}
  auto end=free(x,dt,t);
  if(empirical){auto a=face(x),b=face(end);
   if(t>=release&&a.gap>=-1e-10&&b.gap<=0&&b.speed<0){
    double lo=0,hi=dt;while(hi-lo>1e-10){double mid=.5*(lo+hi);if(face(free(x,mid,t)).gap>0)lo=mid;else hi=mid;}
    double hit=.5*(lo+hi);State at=free(x,hit,t);auto geo=face(at);Eigen::Vector3d before=at.segment<3>(3);double impulse=0;
    if(geo.inside&&geo.speed<0){impulse=impact(at,geo);end=free(at,dt-hit,t+hit);++impacts;}
    last_contact.fill(0.);last_contact[0]=t+hit;
    for(int j=0;j<3;++j){last_contact[1+j]=geo.local[j];last_contact[4+j]=before[j];last_contact[7+j]=at[3+j];last_contact[10+j]=geo.n[j];last_contact[13+j]=geo.p[j];last_contact[16+j]=(geo.J*at.segment<7>(13))[j];}
    for(int j=0;j<7;++j)last_contact[19+j]=at[6+j];last_contact[26]=impulse;last_contact[27]=geo.inside?1.:0.;last_contact[28]=axes.x();last_contact[29]=axes.y();++crossings;
   }
  }else{auto a=model.geometry(x),b=model.geometry(end);
  if(t>=release&&a.gap>=-1e-10&&b.gap<=0&&b.speed<0){
   double lo=0,hi=dt;while(hi-lo>1e-10){double mid=(lo+hi)/2;if(model.geometry(free(x,mid,t)).gap>0)lo=mid;else hi=mid;}
   double hit=(lo+hi)/2;State at=free(x,hit,t);auto geo=model.geometry(at);
   if(geo.radial<g::paddle_radius&&geo.speed<0){Joint dq=at.segment<7>(13);apply_ball_impact(model,at);at.segment<7>(13)=dq;end=free(at,dt-hit,t+hit);++impacts;}
  }}
  acceleration=(end.segment<7>(13)-x.segment<7>(13))/dt;x=end;t+=dt;
  double blend=residual_time_constant>0?-std::expm1(-dt/residual_time_constant):1.;acceleration_residual+=blend*(acceleration_target-acceleration_residual);
  // Diagnostic only: ideal inverse dynamics of the illustrative servo motion.
  // This excludes ball-impact torque and unmodeled physical servo/sensor effects.
  Joint torque=model.rnea(x.segment<7>(6),x.segment<7>(13),acceleration);
  if(!filter_initialized){filtered_torque=torque;filter_initialized=true;}
  Joint next=filtered_torque-std::expm1(-dt/.005)*(torque-filtered_torque);
  Joint rate=(next-filtered_torque)/dt;filtered_torque=next;
  if(control.mpc())for(int j=0;j<7;++j){
   motion_ratios[0]=std::max(motion_ratios[0],std::abs(x[13+j])/g::velocity_limit[j]);
   motion_ratios[1]=std::max(motion_ratios[1],std::abs(acceleration[j])/g::acceleration_limit[j]);
   motion_ratios[2]=std::max(motion_ratios[2],std::abs(torque[j])/g::torque_limit[j]);
   motion_ratios[3]=std::max(motion_ratios[3],std::abs(rate[j])/measured_torque_rate_guard);
  }
  // Once stopping, the intentionally unjuggled ball may fall out of view;
  // continue recording the complete arm braking trajectory for diagnostics.
  if(!x.allFinite()||(!control.stopped()&&(x[2]<-.3||x[2]>2.)))throw std::runtime_error("Offline ball left workspace");
 }
};
void message(char* out,size_t n,const char* text){if(out&&n)std::snprintf(out,n,"%s",text);}
}
extern "C" {
void* fixture_plant_create(const char* f,const double* q,const double* b,double release,char* error,size_t n){try{return new Plant(f,q,b,release);}catch(const std::exception& e){message(error,n,e.what());return nullptr;}}
void fixture_plant_destroy(void* p){delete static_cast<Plant*>(p);}
int fixture_plant_step(void* p,double* state,double* acceleration,int* impacts,char* error,size_t n){try{
 auto& m=*static_cast<Plant*>(p);m.step();std::copy(m.x.data(),m.x.data()+23,state);
 // Protocol ddq_d is desired acceleration, not lagged physical acceleration.
 std::copy(m.command_acceleration.data(),m.command_acceleration.data()+7,acceleration);*impacts=m.impacts;return 0;
 }catch(const std::exception& e){message(error,n,e.what());return -1;}}
int fixture_plant_plan(void* p,double time,const double* plan,char* error,size_t n){try{
 auto& m=*static_cast<Plant*>(p);juggling::SessionCommand c;c.kind=juggling::SessionCommand::Plan;c.source=time;std::copy(plan,plan+c.trajectory.size(),c.trajectory.begin());m.control.accept(c,Plant::joints(m.x.data()+6),m.t);return 0;
 }catch(const std::exception& e){message(error,n,e.what());return -1;}}
int fixture_plant_move(void* p,int id,double time,const double* goal,char* error,size_t n){try{
 auto& m=*static_cast<Plant*>(p);juggling::SessionCommand c;c.kind=juggling::SessionCommand::Move;c.id=id;c.source=time;c.goal=Plant::joints(goal);
 m.control.accept(c,Plant::joints(m.x.data()+6),m.t);return 0;
 }catch(const std::exception& e){message(error,n,e.what());return -1;}}
// Offline supervisor transport only: these functions have no SDK dependency.
void fixture_plant_status(void* p,int* id,int* moving,int* mpc){
 auto& m=*static_cast<Plant*>(p);*id=m.control.id();*moving=m.control.moving(m.t);*mpc=m.control.mpc();
}
void fixture_plant_release_at(void* p,double time){static_cast<Plant*>(p)->release=time;}
void fixture_plant_stop(void* p){auto& m=*static_cast<Plant*>(p);SessionCommand c;c.kind=SessionCommand::Stop;m.control.accept(c,Plant::joints(m.x.data()+6),m.t);}
int fixture_plant_delivery_limit(void* p,double seconds,char* error,size_t n){try{static_cast<Plant*>(p)->control.delivery_limit(seconds);return 0;}catch(const std::exception& e){message(error,n,e.what());return -1;}}
int fixture_plant_protocol(){return 2;}
void fixture_plant_motion_ratios(void* p,double* ratios){auto& m=*static_cast<Plant*>(p);std::copy(m.motion_ratios.begin(),m.motion_ratios.end(),ratios);}
// Versioned offline-only empirical extension. Units/order are specified by
// tracking.empirical_simulation.NativePlant; no hardware executable calls this.
int fixture_plant_empirical_version(){return 1;}
int fixture_plant_configure_execution(void* p,const double* steps,const double* gains,char* error,size_t n){try{
 auto& m=*static_cast<Plant*>(p);if(m.t!=0)throw std::runtime_error("Configure empirical execution before stepping");
 m.control.execution_steps({steps[0],steps[1],steps[2]});m.servo.execution_gains(gains[0],gains[1],gains[2]);return 0;
 }catch(const std::exception& e){message(error,n,e.what());return -1;}}
int fixture_plant_configure(void* p,const double* parameters,int count,char* error,size_t n){try{static_cast<Plant*>(p)->configure(parameters,count);return 0;}catch(const std::exception& e){message(error,n,e.what());return -1;}}
int fixture_plant_disturbance(void* p,const double* acceleration,const double* impact,int immediate,char* error,size_t n){try{
 auto& m=*static_cast<Plant*>(p);m.acceleration_target=Eigen::Map<const Eigen::Vector3d>(acceleration);m.impact_residual=Eigen::Map<const Eigen::Vector3d>(impact);
 if(!m.acceleration_target.allFinite()||!m.impact_residual.allFinite())throw std::runtime_error("Invalid empirical disturbance");
 if(immediate)m.acceleration_residual=m.acceleration_target;return 0;
 }catch(const std::exception& e){message(error,n,e.what());return -1;}}
int fixture_plant_gravity(void* p,const double* gravity,char* error,size_t n){try{auto& m=*static_cast<Plant*>(p);m.gravity=Eigen::Map<const Eigen::Vector3d>(gravity);if(!m.gravity.allFinite()||m.gravity.z()>=0)throw std::runtime_error("Invalid empirical gravity");return 0;}catch(const std::exception& e){message(error,n,e.what());return -1;}}
int fixture_plant_last_contact(void* p,double* data){auto& m=*static_cast<Plant*>(p);std::copy(m.last_contact.begin(),m.last_contact.end(),data);return m.crossings;}
}

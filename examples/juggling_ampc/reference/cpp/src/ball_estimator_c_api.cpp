// Offline adapter. No robot SDK, sockets, generated factors or control calls.
#include "ball_estimator.hpp"
#include <cstdio>
#include <memory>
namespace b=juggling::ball;
namespace {
template<class F> int call(char* error,int size,F fn){try{fn();return 0;}catch(const std::exception& e){if(error&&size>0)std::snprintf(error,size,"%s",e.what());return 1;}}
template<int N> auto vector(const double* p){return Eigen::Map<const Eigen::Matrix<double,N,1>>(p);}
using Row12=Eigen::Matrix<double,12,12,Eigen::RowMajor>;
using Row6=Eigen::Matrix<double,6,6,Eigen::RowMajor>;
b::ObserverConfig config(const double* p,int count){
 if(count!=32)throw std::runtime_error("Observer config ABI expects 32 doubles");
 b::ObserverConfig c;
 c.position_std=p[0];c.drag=p[1];c.quadratic_drag=p[2];c.gravity=vector<3>(p+3);
 c.acceleration_time_constant=p[6];c.acceleration_noise=p[7];c.velocity_noise=p[8];c.spin_noise=p[9];
 c.initial_velocity_std=p[10];c.initial_spin_std=p[11];c.initial_acceleration_std=p[12];
 c.restitution=p[13];c.friction=p[14];c.ball_mass=p[15];c.radius=p[16];c.paddle_radius=p[17];c.inertia_ratio=p[18];
 c.contact_delay=p[19];c.impact_time_std=p[20];c.impact_velocity_std=p[21];c.impact_tangent_std=p[22];c.impact_spin_std=p[23];
 c.contact_measurement_std=p[24];c.contact_measurement_window=p[25];c.impact_refractory=p[26];
 c.innovation_gate=p[27];c.max_measurement_age=p[28];c.history_seconds=p[29];
 if(p[30]!=1)throw std::runtime_error("Observer ABI version mismatch");
 // ABI1's previously reserved slot remains zero for all legacy callers.
 c.impact_independent_tangent_std=p[31];return c;
}
}
extern "C" {
int offline_estimator_abi(){return 1;}
// Feature bit0: ABI1 slot31 accepts independent tangential velocity noise.
unsigned int offline_observer_features(){return 1u;}
void* offline_observer_create(const double* p,int n,char* e,int z){void* result=nullptr;call(e,z,[&]{result=new b::BallObserver(config(p,n));});return result;}
void offline_observer_destroy(void* h){delete static_cast<b::BallObserver*>(h);}
int offline_observer_arm(void* h,double t,const double* v,char* e,int z){return call(e,z,[&]{b::ArmSample a;a.t=t;a.p=vector<3>(v);a.n=vector<3>(v+3);a.velocity=vector<3>(v+6);a.angular_velocity=vector<3>(v+9);a.inverse_spatial_inertia=Eigen::Map<const Row6>(v+12);static_cast<b::BallObserver*>(h)->record_arm(a);});}
int offline_observer_initialize(void* h,double t,const double* p,const double* v,char* e,int z){return call(e,z,[&]{static_cast<b::BallObserver*>(h)->initialize(vector<3>(p),t,vector<3>(v));});}
int offline_observer_estimate(void* h,double t,double* x,double* P,double* info,char* e,int z){return call(e,z,[&]{auto* o=static_cast<b::BallObserver*>(h);auto s=o->estimate(t);Eigen::Map<b::Vec12> ox(x);ox=s.x;Eigen::Map<Row12> op(P);op=s.P;info[0]=s.last_impact;info[1]=s.measurement_age;info[2]=s.stale;info[3]=o->posterior_time();info[4]=o->latest_arm_time();});}
int offline_observer_update(void* h,double t,const double* p,double arrival,double* info,char* reason,int reason_size,char* e,int z){return call(e,z,[&]{auto u=static_cast<b::BallObserver*>(h)->update_position(t,vector<3>(p),arrival);info[0]=u.accepted;info[1]=u.near_impact;info[2]=u.nis;for(int j=0;j<3;++j)info[3+j]=u.innovation[j];info[6]=static_cast<b::BallObserver*>(h)->posterior_time();std::snprintf(reason,reason_size,"%s",u.reason);});}
int offline_observer_pending(void* h,double t,int* value,char* e,int z){return call(e,z,[&]{*value=static_cast<b::BallObserver*>(h)->contact_pending(t);});}

void* offline_raw_create(char* e,int z){void* result=nullptr;call(e,z,[&]{result=new b::RawBall(b::RawBallConfig{3.,.5,.35,3.,.08,.001});});return result;}
void offline_raw_destroy(void* h){delete static_cast<b::RawBall*>(h);}
int offline_raw_update(void* h,std::uint64_t frame,double t,const double* xyz,const std::uint64_t* ids,int n,const double* face,const double* prediction,int motion,double* output,char* e,int z){return call(e,z,[&]{
 auto* raw=static_cast<b::RawBall*>(h);raw->allow_motion=motion;std::vector<b::Marker> points;for(int k=0;k<n;++k)points.push_back({ids[k],vector<3>(xyz+3*k)});
 std::optional<b::Vec3> f;if(face)f=vector<3>(face);std::optional<b::AssociationPrediction> p;
 if(prediction){b::AssociationPrediction a;a.x=vector<6>(prediction);a.P=Eigen::Map<const Row6>(prediction+6);a.position_std=prediction[42];p=a;}
 auto found=raw->update(frame,t,points,f,p);output[0]=bool(found);if(found)for(int j=0;j<3;++j)output[1+j]=(*found)[j];output[4]=raw->selected();output[5]=raw->seen_time();
 });}

void* offline_gate_create(const double* gravity,char* e,int z){void* result=nullptr;call(e,z,[&]{result=new b::ReleaseGate({},vector<3>(gravity));});return result;}
void offline_gate_destroy(void* h){delete static_cast<b::ReleaseGate*>(h);}
int offline_gate_update(void* h,double t,const double* p,std::uint64_t frame,double* out,char* e,int z){return call(e,z,[&]{auto* gate=static_cast<b::ReleaseGate*>(h);auto r=gate->update(t,vector<3>(p),frame);out[0]=bool(gate->anchor());if(gate->anchor())for(int j=0;j<3;++j)out[1+j]=(*gate->anchor())[j];out[4]=bool(r);if(r){out[5]=r->capture_time;for(int j=0;j<3;++j){out[6+j]=r->position[j];out[9+j]=r->velocity[j];}out[12]=r->observed_drop;out[13]=double(r->frame);}});}
int offline_gate_arm(void* h,char* e,int z){return call(e,z,[&]{static_cast<b::ReleaseGate*>(h)->arm();});}
int offline_gate_missing(void* h,double t,char* e,int z){return call(e,z,[&]{static_cast<b::ReleaseGate*>(h)->missing(t);});}
}

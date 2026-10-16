// Offline batched adapter around the unchanged hardware hybrid EKF.
#include "ball_estimator.hpp"
#include <memory>
#include <vector>
#include <cstring>
namespace ba=juggling::ball;
struct Batch {
  std::vector<std::unique_ptr<ba::BallObserver>> observers;
  std::vector<int> failed;
};
static ba::ArmSample arm(double t,const double* g){
  ba::ArmSample a;a.t=t;a.p=Eigen::Map<const ba::Vec3>(g);a.n=Eigen::Map<const ba::Vec3>(g+3);
  a.velocity=Eigen::Map<const ba::Vec3>(g+6);a.angular_velocity=Eigen::Map<const ba::Vec3>(g+9);
  a.inverse_spatial_inertia=Eigen::Map<const Eigen::Matrix<double,6,6,Eigen::RowMajor>>(g+12);return a;
}
extern "C" void* juggling_observers_create(int n,const double* config,const double* position,const double* velocity,const double* geometry){
  try{
    if(n<=0)return nullptr;
    ba::ObserverConfig c;c.gravity=Eigen::Map<const ba::Vec3>(config);c.drag=config[3];c.ball_mass=config[4];c.radius=config[5];c.restitution=config[6];c.friction=config[7];c.inertia_ratio=config[8];c.paddle_radius=config[9];
    c.contact_measurement_window=.015;c.max_measurement_age=.1;c.initial_velocity_std=.08;
    auto b=std::make_unique<Batch>();b->failed.resize(n,0);b->observers.reserve(n);
    for(int i=0;i<n;++i){auto o=std::make_unique<ba::BallObserver>(c);o->record_arm(arm(0.,geometry+i*48));o->initialize(Eigen::Map<const ba::Vec3>(position+i*3),0.,Eigen::Map<const ba::Vec3>(velocity+i*3));b->observers.push_back(std::move(o));}
    return b.release();
  }catch(...){return nullptr;}
}
extern "C" void juggling_observers_destroy(void* ptr){delete static_cast<Batch*>(ptr);}
extern "C" void juggling_observers_step(void* ptr,double t,const double* geometry,const double* capture,const double* position,int predict,double* output){
  auto& batch=*static_cast<Batch*>(ptr);const int n=batch.observers.size();
  #pragma omp parallel for schedule(static)
  for(int i=0;i<n;++i){
    if(!batch.failed[i])try{
      auto& o=*batch.observers[i];if(t>o.latest_arm_time())o.record_arm(arm(t,geometry+i*48));
      if(std::isfinite(capture[i]))o.update_position(capture[i],Eigen::Map<const ba::Vec3>(position+i*3),t);
      if(predict){auto e=o.estimate(t);Eigen::Map<Eigen::Matrix<double,9,1>>(output+12*i)=e.x.head<9>();output[12*i+9]=e.last_impact;output[12*i+10]=e.measurement_age;output[12*i+11]=e.stale?1.:0.;}
    }catch(...){batch.failed[i]=1;}
    if(batch.failed[i]){for(int j=0;j<11;++j)output[12*i+j]=std::numeric_limits<double>::quiet_NaN();output[12*i+11]=1.;}
  }
}

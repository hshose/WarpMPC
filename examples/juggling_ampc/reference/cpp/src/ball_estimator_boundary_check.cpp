// Build twice: with the optional conditioned Jacobian and with the legacy
// default. Neither executable opens hardware, solves an MPC, nor simulates an
// experiment. The legacy executable succeeds only if it reproduces the bug.
#include "ball_estimator.hpp"
#include <iomanip>
#include <iostream>
#include <memory>

using namespace juggling::ball;
namespace {
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
ArmSample arm(double t,double upward){ArmSample a;a.t=t;a.p.z()=upward*t;a.velocity.z()=upward;return a;}
struct Result {
  const char* name;
  double contact_delay,impact_time,position_error,velocity_error,position_variance,velocity_variance;
  double covariance_min_eigenvalue,covariance_max_eigenvalue,updated_vz,updated_position_error;
  double predicted_vz,expected_position_variance;
};
Result boundary_case(const char* name,double event,double contact_delay=0.,double penetration=0.){
  ObserverConfig cfg;cfg.drag=0.;cfg.quadratic_drag=0.;cfg.restitution=.557;
  cfg.contact_delay=contact_delay;cfg.impact_time_std=0.;
  cfg.initial_velocity_std=.02;cfg.initial_spin_std=.2;cfg.initial_acceleration_std=.02;
  cfg.impact_velocity_std=.02;cfg.impact_tangent_std=.005;cfg.impact_spin_std=.02;
  constexpr double dt=.002,incoming=-3.4,upward=1.1;
  const double gravity=cfg.gravity.z();
  const double before=incoming+gravity*event,relative_before=before-upward;
  const double initial_z=cfg.radius+(upward-incoming)*event-.5*gravity*event*event+relative_before*contact_delay-penetration;
  auto observer=std::make_unique<BallObserver>(cfg);
  observer->record_arm(arm(0.,upward));
  observer->initialize({0.,0.,initial_z},0.,{0.,0.,incoming});
  observer->record_arm(arm(dt,upward));
  const auto predicted=observer->estimate(dt);
  const double after=upward-cfg.restitution*(before-upward),remaining=dt-event;
  const double expected_z=cfg.radius+upward*event+relative_before*contact_delay-penetration+after*remaining+.5*gravity*remaining*remaining;
  const double expected_vz=after+gravity*remaining;
  check(std::isfinite(predicted.last_impact),"Expected nominal boundary impact missing");
  check(std::abs(predicted.last_impact-event)<1e-10,"Nominal impact time changed");
  check(std::abs(predicted.x.z()-expected_z)<1e-10,"Nominal postimpact position changed");
  check(std::abs(predicted.x[5]-expected_vz)<1e-9,"Nominal postimpact velocity changed");
  check(predicted.P.allFinite(),"Nonfinite boundary covariance");
  Eigen::SelfAdjointEigenSolver<Mat12> eigen(predicted.P);
  check(eigen.info()==Eigen::Success,"Boundary covariance eigensolver failed");
  // Position uncertainty dominates this entry. The closed-form saltation
  // derivative dp(end)/dp(0) for a translating plane is the expression below;
  // the deliberately small initial velocity uncertainty contributes <1%.
  const double d_position=1.-(1.+cfg.restitution)*(relative_before-gravity*remaining)/(relative_before-gravity*contact_delay);
  const double expected_position_variance=cfg.position_std*cfg.position_std*d_position*d_position;
  Vec3 measured=predicted.x.template head<3>();measured.z()+=.0005;
  const auto update=observer->update_position(dt,measured,dt);
  check(update.accepted,"Small postimpact position observation rejected");
  const auto posterior=observer->estimate(dt);
  check(posterior.x.allFinite()&&posterior.P.allFinite(),"Nonfinite updated boundary estimate");
#ifdef JUGGLING_CONDITIONED_CONTACT_JACOBIAN
  check(predicted.P(5,5)<1.,"Conditioned event produced excessive velocity covariance");
  check(eigen.eigenvalues().minCoeff()>-1e-9,"Conditioned covariance is not PSD");
  check(eigen.eigenvalues().maxCoeff()<10.,"Conditioned event produced excessive covariance");
  check(std::abs(predicted.P(2,2)/expected_position_variance-1.)<.03,"Boundary position covariance misses event-conditioned derivative");
  check(std::abs(posterior.x[5]-expected_vz)<.02,"Position update produced unphysical rebound velocity");
  Eigen::SelfAdjointEigenSolver<Mat12> posterior_eigen(posterior.P);
  check(posterior_eigen.info()==Eigen::Success&&posterior_eigen.eigenvalues().minCoeff()>-1e-9,"Updated covariance is not PSD");
#endif
  return {name,contact_delay,predicted.last_impact,std::abs(predicted.x.z()-expected_z),std::abs(predicted.x[5]-expected_vz),
          predicted.P(2,2),predicted.P(5,5),eigen.eigenvalues().minCoeff(),eigen.eigenvalues().maxCoeff(),
          posterior.x[5],std::abs(posterior.x.z()-expected_z),predicted.x[5],expected_position_variance};
}
void no_impact_case(){
  ObserverConfig cfg;cfg.drag=0.;cfg.contact_delay=0.;
  auto observer=std::make_unique<BallObserver>(cfg);observer->record_arm(arm(0.,0.));
  observer->initialize({0.,0.,.05},0.,{0.,0.,-1.});observer->record_arm(arm(.002,0.));
  Vec12 initial=Vec12::Zero();initial.z()=.05;initial[5]=-1.;
  const auto predicted=observer->estimate(.002);
  check(!std::isfinite(predicted.last_impact),"Jacobian option invented a nominal contact");
  check((predicted.x-observer->free(initial,.002)).norm()<1e-13,"No-impact mean changed");
}
void print(const Result& r){
  std::cout<<"{\"case\":\""<<r.name<<"\",\"contact_delay_s\":"<<r.contact_delay<<",\"impact_time_s\":"<<r.impact_time
    <<",\"mean_position_error_m\":"<<r.position_error<<",\"mean_velocity_error_m_s\":"<<r.velocity_error
    <<",\"position_variance_m2\":"<<r.position_variance<<",\"expected_position_variance_m2\":"<<r.expected_position_variance
    <<",\"velocity_variance_m2_s2\":"<<r.velocity_variance<<",\"covariance_min_eigenvalue\":"<<r.covariance_min_eigenvalue
    <<",\"covariance_max_eigenvalue\":"<<r.covariance_max_eigenvalue<<",\"predicted_vz_m_s\":"<<r.predicted_vz
    <<",\"updated_vz_m_s\":"<<r.updated_vz<<",\"updated_position_error_m\":"<<r.updated_position_error<<"}";
}
}
int main(){try{
  no_impact_case();
  const std::array<Result,7> results{boundary_case("near_beginning",.000000223),
      boundary_case("interior",.001),boundary_case("near_end",.002-.000000223),
      boundary_case("clamped_start_penetrated",0.,.001,.0004),
      boundary_case("near_beginning_delay",.000000223,.001),
      boundary_case("interior_delay",.001,.001),boundary_case("near_end_delay",.002-.000000223,.001)};
#ifdef JUGGLING_CONDITIONED_CONTACT_JACOBIAN
  constexpr bool conditioned=true;
#else
  constexpr bool conditioned=false;
  check(results.back().velocity_variance>1e4,"Legacy regression did not reproduce covariance explosion");
  check(std::abs(results.back().updated_vz-results.back().predicted_vz)>1.,"Legacy regression did not reproduce excessive velocity correction");
#endif
  std::cout<<std::setprecision(17)<<"{\"passed\":true,\"conditioned_contact_jacobian\":"<<(conditioned?"true":"false")
    <<",\"legacy_pathology_reproduced\":"<<(conditioned?"false":"true")<<",\"hardware_activity\":false,\"cases\":[";
  for(std::size_t k=0;k<results.size();++k){if(k)std::cout<<',';print(results[k]);}
  std::cout<<"]}\n";return 0;
}catch(const std::exception& e){std::cerr<<"ball_estimator_boundary_check: "<<e.what()<<'\n';return 1;}}

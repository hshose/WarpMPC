#include "ball_estimator.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
using namespace juggling::ball;
namespace {
void check(bool value,const char* text){if(!value)throw std::runtime_error(text);}
template<class F> void throws(F f,const char* text){bool caught=false;try{f();}catch(const std::exception&){caught=true;}check(caught,text);}
ArmSample arm(double t){ArmSample a;a.t=t;return a;}
void check_raw_selection(){
  // Stable near point wins even when farther clutter has perfect coverage.
  // A near intermittent/moving point must not displace a persistent fixture.
  const Vec3 near(.03,.01,1.2),far(.18,.01,1.2);
  for(int scenario=0;scenario<5;++scenario){auto raw=std::make_unique<RawBall>();
    for(int k=0;k<=300;++k){std::vector<Marker> points{{999,far},{1,{.5,0,1.2}},{2,{0,0,.1}}};
      if(scenario!=3&&(scenario!=1||k%25!=12)&&(scenario!=2||k%4!=2)){
        Vec3 p=near;if(scenario==4)p.z()+=(k%2?.0045:-.0045);points.push_back({std::uint64_t(k%7),p});}
      if(k%2)std::reverse(points.begin(),points.end());raw->update(k,k*.01,points,Vec3::Zero());
    }
    check(raw->selected(),"No stable ceiling marker selected");
    const Vec3 expected=scenario<2?near:far;
    check((raw->selection()->position-expected).norm()<1e-10,"Stationarity/visibility/nearest-XY selection failed");
    if(scenario<2)check(raw->selection()->eligible_candidates==2,"Stable distractor missing from selection diagnostics");
  }
  auto empty=std::make_unique<RawBall>();for(int k=0;k<=310;++k)empty->update(k,k*.01,{},Vec3::Zero());check(!empty->selected(),"Empty stream selected a ball");
}
}
int main(int argc,char** argv){try{
  ObserverConfig cfg;cfg.gravity={.0928822557147292,-.010164049455359378,-9.676561546699691};cfg.drag=.09093571620319228;cfg.quadratic_drag=.06386175735959494;
  auto observer=std::make_unique<BallObserver>(cfg);observer->record_arm(arm(0.));observer->initialize({0.,0.,1.},0.);
  Vec12 initial=Vec12::Zero();initial.z()=1.;initial.template segment<3>(3)=Vec3(.12,-.04,.3);initial.template tail<3>()=Vec3(.02,-.01,.04);
  const auto roundtrip=observer->free(observer->free(initial,.1),-.1);check((roundtrip-initial).norm()<2e-8,"Mixed flight reversibility");
  double max_error=0.;
  for(int k=1;k<=600;++k){const double t=k*.0005;observer->record_arm(arm(t));if(k%20==0){const double capture=t-.0045;const Vec3 exact=observer->free(Vec12((Vec12()<<0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.).finished()),capture).template head<3>();const auto update=observer->update_position(capture,exact,t);check(update.accepted,"Delayed exact flight measurement rejected");const auto estimated=observer->estimate(t);const Vec3 truth=observer->free(Vec12((Vec12()<<0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.).finished()),t).template head<3>();max_error=std::max(max_error,(estimated.x.template head<3>()-truth).norm());}}
  check(max_error<2e-8,"Delayed flight estimate error");check(!observer->update_position(.2955,{0.,0.,1.},.3).accepted,"Duplicate timestamp accepted");check(!observer->update_position(.299,{20.,0.,1.},.3).accepted,"Outlier accepted");
  for(int k=601;k<=780;++k)observer->record_arm(arm(k*.0005));check(observer->estimate(.39).measurement_age>.08,"Missing measurements did not coast");
  throws([&]{observer->estimate(.40);},"Future prediction allowed");
  auto bouncing=std::make_unique<BallObserver>(cfg);bouncing->record_arm(arm(0.));bouncing->initialize({0.,0.,.12},0.,{0.,0.,-1.});
  for(int k=1;k<=200;++k)bouncing->record_arm(arm(k*.001));const auto bounced=bouncing->estimate(.2);check(std::isfinite(bounced.last_impact)&&bounced.last_impact>=0.&&bounced.last_impact<=.2,"Causal impact missing");check(bounced.x.z()>.02,"Ball did not rebound");check(bounced.P.allFinite(),"Invalid impact covariance");Eigen::SelfAdjointEigenSolver<Mat12> eigen(bounced.P);check(eigen.eigenvalues().minCoeff()>-1e-8,"Impact covariance is not PSD");
  auto separating=std::make_unique<BallObserver>(cfg);separating->record_arm(arm(0.));separating->initialize({0.,0.,.019},0.,{0.,0.,1.});separating->record_arm(arm(.002));check(!std::isfinite(separating->estimate(.002).last_impact),"Separating ball received impulse");
  ObserverConfig finite_cfg;finite_cfg.drag=0.;finite_cfg.quadratic_drag=0.;auto finite_arm=std::make_unique<BallObserver>(finite_cfg);auto finite_a=arm(0.);finite_a.inverse_spatial_inertia.template topLeftCorner<3,3>()=2.*Mat3::Identity();finite_arm->record_arm(finite_a);finite_arm->initialize({0.,0.,.019},0.,{0.,0.,-1.});finite_a.t=1e-7;finite_arm->record_arm(finite_a);const auto finite_hit=finite_arm->estimate(1e-7);const double expected_velocity=-1.+(1.+finite_cfg.restitution)/(1.+2.*finite_cfg.ball_mass)+finite_cfg.gravity.z()*1e-7;check(std::abs(finite_hit.x[5]-expected_velocity)<1e-9,"Finite-arm inverse inertia missing from impulse");
  auto oblique=std::make_unique<BallObserver>(finite_cfg);ArmSample oblique_a=arm(0.);oblique_a.n=Vec3(.01,-.02,1.).normalized();oblique_a.velocity=Vec3(.03,-.04,.1);Mat3 inverse_mass;inverse_mass<<.7,.12,-.08,.12,.9,.07,-.08,.07,1.1;oblique_a.inverse_spatial_inertia.template topLeftCorner<3,3>()=inverse_mass;const Vec3 incoming(-.1,.2,-1.2),relative=incoming-oblique_a.velocity;oblique->record_arm(oblique_a);oblique->initialize((finite_cfg.radius+oblique_a.n.dot(relative)*finite_cfg.contact_delay)*oblique_a.n,0.,incoming);oblique_a.t=1e-7;oblique_a.p=oblique_a.velocity*oblique_a.t;oblique->record_arm(oblique_a);const auto oblique_est=oblique->estimate(1e-7);check(std::abs(oblique_est.last_impact)<1e-12,"Oblique event is not at initial contact");const auto oblique_at_impact=oblique->free(oblique_est.x,-1e-7);const Vec3 velocity_jump=oblique_at_impact.template segment<3>(3)-incoming,spin_jump=oblique_at_impact.template segment<3>(6);
  auto raw=std::make_unique<RawBall>();std::optional<Vec3> found;for(int k=0;k<=310;++k)found=raw->update(k,k*.01,{{std::uint64_t(k%7),{.03,.01,1.2}}},Vec3::Zero());check(raw->selected()&&found.has_value(),"Stationary unlabeled marker not acquired");raw->allow_motion=true;const double last=raw->seen_time();check(!raw->update(311,last+.01,{{9,{.08,.01,1.2}}},Vec3::Zero()),"Marker jumped 50 mm");check(!raw->update(312,last+.02,{},Vec3::Zero()),"Missing marker synthesized");check(raw->update(313,last+.03,{{9,{.031,.01,1.2}}},Vec3::Zero()).has_value(),"Marker did not bridge short gap");check(!raw->update(314,last+.04,{{9,{.031,.01,1.2}},{10,{.032,.01,1.2}}},Vec3::Zero()),"Ambiguous marker accepted");throws([&]{raw->update(315,last+.12,{{9,{.031,.01,1.2}}},Vec3::Zero());},"Lost raw marker reacquired");
  check_raw_selection();
  ReleaseGate gate({},cfg.gravity);for(int k=0;k<=100;++k)gate.update(k*.01,{.03,.01,1.2},k);check(gate.anchor().has_value(),"Fixture did not stabilize");gate.arm();std::optional<Release> release;for(int k=101;k<=109&&!release;++k){const double t=k*.01,dt=t-1.;release=gate.update(t,Vec3(.03,.01,1.2)+.5*dt*dt*cfg.gravity,k);}check(release.has_value(),"Falling ball not released");check((release->velocity-(release->capture_time-1.)*cfg.gravity).norm()<1e-10,"Release velocity fit incorrect");
  ReleaseGate lost;lost.update(0.,{0.,0.,1.},1);throws([&]{lost.missing(.081);},"Release allowed without tracking");
  std::ofstream dump;if(argc==2){dump.open(argv[1]);dump<<std::setprecision(17);for(int model=0;model<2;++model){ObserverConfig c=cfg;if(model==0)c.quadratic_drag=0.;BallObserver b(c);for(double dt:{-.015,0.,.0005,.01,.15}){const auto x=b.free(initial,dt);dump<<model<<' '<<dt;for(int j=0;j<12;++j)dump<<' '<<x[j];dump<<'\n';}}}
  if(argc==2){std::ofstream contact(std::string(argv[1])+".contact");contact<<std::setprecision(17);for(const Vec3* value:std::array<const Vec3*,4>{&oblique_a.n,&relative,&velocity_jump,&spin_jump})for(int j=0;j<3;++j)contact<<(*value)[j]<<" ";contact<<"\n";}
  std::vector<double> latency;latency.reserve(2000);for(int k=0;k<2000;++k){const auto start=std::chrono::steady_clock::now();const auto e=bouncing->estimate(.2);check(e.x.allFinite(),"Nonfinite benchmark");const auto stop=std::chrono::steady_clock::now();latency.push_back(std::chrono::duration<double,std::micro>(stop-start).count());}std::sort(latency.begin(),latency.end());
  std::cout<<std::setprecision(9)<<"{\"passed\":true,\"delayed_flight_max_error_m\":"<<max_error<<",\"impact_time_s\":"<<bounced.last_impact<<",\"release_time_s\":"<<release->capture_time<<",\"predict_200ms_with_impact_us_p50\":"<<latency[1000]<<",\"predict_200ms_with_impact_us_p99\":"<<latency[1980]<<",\"predict_200ms_with_impact_us_max\":"<<latency.back()<<"}\n";return 0;
}catch(const std::exception& e){std::cerr<<"ball_estimator_check: "<<e.what()<<'\n';return 1;}}

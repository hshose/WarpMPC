// Native callback state-machine checks, no SDK or network.
#include "session_control.hpp"
#include "joint_velocity_tracking.hpp"
#include <iostream>
#include <limits>
#include <functional>
using namespace juggling;
namespace g=juggling::generated;
void require(bool result){if(!result)throw std::runtime_error("Check failed");}
void rejects(const std::function<void()>& f){bool rejected=false;try{f();}catch(const std::runtime_error&){rejected=true;}require(rejected);}
SessionCommand plan(double time){SessionCommand c;c.kind=SessionCommand::Plan;c.source=time;
 for(int k=0;k<4;++k)for(int j=0;j<7;++j)c.trajectory[28*k+j]=g::home[j];return c;}
int main(){try{
 for(double dt:{0.,.001,.005,.006,.010}){
  require(valid_robot_callback_period(dt));JointVelocityTracking probe({});
  probe.step({}, {}, {},dt,false);
 }
 for(double dt:{-.001,.010001,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}){
  require(!valid_robot_callback_period(dt));JointVelocityTracking probe({});
  rejects([&]{probe.step({}, {}, {},dt,false);});
 }
 SessionControl control(g::home);SessionCommand move;move.kind=SessionCommand::Move;move.source=1;move.id=1;move.goal=g::home;move.goal[6]+=.1;
 control.accept(move,g::home,1);require(control.moving(1));rejects([&]{control.accept(plan(1.01),g::home,1.01);});
 auto end=control.reference(10);require(std::abs(end[6]-move.goal[6])<1e-12);require(!control.moving(10));
 SessionControl delivery(g::home);delivery.delivery_limit(.015);rejects([&]{delivery.accept(plan(1),g::home,1.016);});delivery.accept(plan(2),g::home,2.014);
 SessionControl clock(g::home);
 rejects([&]{clock.accept(plan(1),g::home,1.07);});rejects([&]{clock.accept(plan(1.01),g::home,1);});
 clock.accept(plan(2),g::home,2);rejects([&]{clock.reference(2.061);});rejects([&]{clock.accept(plan(2),g::home,2);});
 rejects([&]{move.source=3;clock.accept(move,g::home,3);});
 for(int type=0;type<3;++type){SessionControl bad(g::home);auto c=plan(1);
  if(type==0)c.trajectory[0]=std::numeric_limits<double>::quiet_NaN();
  if(type==1)c.trajectory[7]=g::velocity_limit[0]+.01;
  if(type==2)c.trajectory[14]=g::torque_limit[0]+.01;
  rejects([&]{bad.accept(c,g::home,1);});}
 if(g::stage_times[2]>.06){SessionControl unused(g::home);auto c=plan(1);c.trajectory[3*28+7]=2*g::velocity_limit[0];unused.accept(c,g::home,1);
  auto r=unused.reference(1.05);require(r[7]==0);rejects([&]{unused.reference(1.061);});}
 SessionControl bad_move(g::home);move.source=1;move.goal[0]+=1.01;rejects([&]{bad_move.accept(move,g::home,1);});
 SessionControl stopped(g::home);SessionCommand stop;stopped.accept(stop,g::home,1);require(stopped.stopped());rejects([&]{stopped.accept(plan(1),g::home,1);});
 SessionControl acceleration(g::home);auto p=plan(1);p.trajectory[28+21]=1.11*g::acceleration_limit[0];rejects([&]{acceleration.accept(p,g::home,1);});
 SessionControl future_acceleration(g::home);p=plan(1);p.trajectory[28+21]=1.08*g::acceleration_limit[0];future_acceleration.accept(p,g::home,1);
 future_acceleration.reference(1+.5*g::stage_steps[0]);
 rejects([&]{future_acceleration.reference(1+g::stage_steps[0]);});
 // Inexact shooting velocities must not introduce acceleration absent from
 // the torque-derived endpoint accelerations.
 SessionControl defect(g::home);p=plan(1);p.trajectory[28+7]=.9*g::acceleration_limit[0]*g::stage_steps[0];defect.accept(p,g::home,1);
 require(defect.reference(1.015)[14]==0&&defect.reference(1.015)[7]==0);
 // A constant-jerk trajectory must remain smooth, including knot crossings.
 SessionControl ramp(g::home);auto smooth=plan(1);
 for(int k=0;k<4;++k){double t=g::stage_times[k];smooth.trajectory[28*k]+=40*t*t*t/6;smooth.trajectory[28*k+7]=20*t*t;smooth.trajectory[28*k+21]=40*t;}
 ramp.accept(smooth,g::home,1);
 for(double t:{0.,.015,.029999,.030001,.055}){auto r=ramp.reference(1+t);require(std::abs(r[0]-g::home[0]-40*t*t*t/6)<1e-10);require(std::abs(r[7]-20*t*t)<1e-10);require(std::abs(r[14]-40*t)<1e-10);}
 // Tracking and controlled stop with exactly the execution-layer limiter.
 JointVelocityTracking servo({});std::array<double,7> q=g::home,dq{},last_v{},last_a{};std::array<double,21> reference{};
 double max_v=0,max_a=0,max_j=0;int shutdown_faults=0,first_lost_heartbeat=-1;
 for(int k=0;k<3000;++k){double t=k*.001;for(int j=0;j<7;++j){reference[j]=g::home[j]+(j==0?.03:0);reference[7+j]=reference[14+j]=0;}
  // Replay supervisor shutdown: input ceases while the callback decelerates.
  if(session_heartbeat_expired(t,std::min(t,.999),t>=1.))++shutdown_faults;
  // In active control an unannounced loss still trips at the original limit.
  if(first_lost_heartbeat<0&&session_heartbeat_expired(t,0,false))first_lost_heartbeat=k;
  auto v=servo.step(reference,q,dq,.001,t>=1.);for(int j=0;j<7;++j){double a=(v[j]-last_v[j])/.001;
   max_v=std::max(max_v,std::abs(v[j])/g::velocity_limit[j]);max_a=std::max(max_a,std::abs(a)/g::acceleration_limit[j]);
   max_j=std::max(max_j,std::abs(a-last_a[j])/.001/g::jerk_limit[j]);last_a[j]=a;last_v[j]=v[j];q[j]+=.001*v[j];dq[j]=v[j];}
 }
 require(max_v<=1&&max_a<=.9500001&&max_j<=.2000001&&std::abs(last_v[0])<.0001);
 require(shutdown_faults==0&&first_lost_heartbeat>=150&&first_lost_heartbeat<=151);
 // Finish only after finite-time zero hold and settled SDK states. Exercise
 // Both stop gains, signs and callbacks through the authorized 10 ms limit.
 for(double gain:{JointVelocityTracking::normal_stop_gain,12.})for(double sign:{-1.,1.})for(double dt:{.001,.002,.006,.010}){
  std::array<double,7> initial{};initial.fill(sign*1.5);JointVelocityTracking brake(initial);StopCompletion finish;
  auto v=initial;std::array<double,7> a{};bool complete=false;double time=0;
  for(;time<1.5&&!complete;time+=dt){auto next=brake.step({},g::home,v,dt,true,gain);std::array<double,7> next_a{};
   for(int j=0;j<7;++j){next_a[j]=(next[j]-v[j])/dt;
    require(std::abs(next_a[j])<=g::acceleration_limit[j]);
    require(std::abs(next_a[j]-a[j])/dt<=.200001*g::jerk_limit[j]);}
   complete=finish.update(brake.zero_hold(),v,v,a,dt);v=next;a=next_a;
  }
  require(complete&&time<1.5);for(double x:v)require(x==0.);
 }
 // Trial2-style abort while the wrist is still accelerating outward. Braking
 // must preserve measured-speed headroom without breaking jerk continuity.
 for(double sign:{-1.,1.})for(double dt:{.001,.002,.006,.010}){
  JointVelocityTracking brake({});std::array<double,7> v{},a{};std::array<double,21> ref{};
  bool stopping=false;double peak=0.;
  for(int k=0;k<int(2./dt);++k){
   ref.fill(0.);ref[7+5]=v[5];ref[14+5]=sign*8.1;
   if(std::abs(v[5])>=1.48)stopping=true;
   auto next=brake.step(ref,{},v,dt,stopping,JointVelocityTracking::normal_stop_gain);
   double acceleration=(next[5]-v[5])/dt;
   require(std::abs(acceleration-a[5])/dt<=.200001*g::jerk_limit[5]);
   a[5]=acceleration;v=next;peak=std::max(peak,std::abs(v[5]));
  }
  require(stopping&&peak<.975*g::velocity_limit[5]&&brake.zero_hold());
 }
 StopCompletion premature;std::array<double,7> residual{},zero{};residual[3]=-.01013382;
 for(int k=0;k<200;++k)require(!premature.update(true,zero,residual,zero,.001));
 residual[3]=.06117;for(int k=0;k<200;++k)require(!premature.update(true,zero,zero,residual,.001));
 JointHoming slow(g::home,move.goal,g::joint_lower,g::joint_upper),fast(g::home,move.goal,g::joint_lower,g::joint_upper,2.);
 require(std::abs(slow.duration()/fast.duration()-2.)<1e-12);
 std::cout<<"{\"passed\":true,\"physical_robot_used\":false,\"checks\":16,\"protocol\":2,\"max_callback_period_s\":"<<robot_callback_period_limit<<",\"shutdown_heartbeat_faults\":"<<shutdown_faults<<",\"active_heartbeat_expiry_ms\":"<<first_lost_heartbeat<<",\"maximum_velocity_ratio\":"<<max_v<<",\"maximum_acceleration_ratio\":"<<max_a<<",\"maximum_jerk_ratio\":"<<max_j<<"}\n";
 return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

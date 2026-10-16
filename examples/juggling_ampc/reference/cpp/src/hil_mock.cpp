// Wall-clock integration test of the SAME queues/workers used by libfranka.
// The physical arm is replaced here by the shared rigid-body dynamics; only the
// virtual ball receives impact impulses, exactly as in the intended no-ball HIL.
#include "controller.hpp"
#include "hil_c_api.h"
#include "motion_guard.hpp"
#include "joint_velocity_tracking.hpp"
#include <cmath>
#include "model_config.hpp"
#include <chrono>
#include <thread>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>

int main(int argc,char** argv){
 if(argc<3){std::cerr<<"Usage: juggling_hil_mock FACTORS OUTPUT_DIR [seconds=10] [max_command_age=0.06] [torque|velocity]\n";return 2;}
 double seconds=argc>3?std::stod(argv[3]):10.;double max_age=argc>4?std::stod(argv[4]):.06;if(seconds<=0||seconds>60)return 2;
 std::string execution=argc>5?argv[5]:"torque";if(execution!="torque"&&execution!="velocity")return 2;
 bool velocity_mode=execution=="velocity";
 namespace g=juggling::generated;using namespace juggling;using Clock=std::chrono::steady_clock;
 RobotModel robot(argv[1]);Joint q=Eigen::Map<const Joint>(g::home.data()),dq=Joint::Zero(),tau=robot.bias(q,dq);char error[4096]{};
 Joint servo_acceleration=Joint::Zero();
 void* hil=velocity_mode?juggling_hil_create_acceleration(argv[1],q.data(),dq.data(),servo_acceleration.data(),max_age,error,sizeof(error)):juggling_hil_create(argv[1],q.data(),dq.data(),tau.data(),max_age,error,sizeof(error));if(!hil){std::cerr<<error<<'\n';return 1;}
 if(juggling_hil_start(hil,error,sizeof(error))){std::cerr<<error<<'\n';juggling_hil_destroy(hil);return 1;}
 std::filesystem::create_directories(argv[2]);std::ofstream log(std::string(argv[2])+"/hil_callback.csv");log<<std::setprecision(17)<<"time,callback_us,command_age,lateness_us,fault\n";
 juggling::MotionGuard guard;JointVelocityTracking tracker({});std::array<double,7> velocity_command{};
 auto start=Clock::now();int fault=0;double max_callback=0;
 for(int k=0;k<=std::lround(seconds/.001);++k){double t=k*.001;auto deadline=start+std::chrono::microseconds(k*1000);std::this_thread::sleep_until(deadline);
  auto entered=Clock::now();Joint next;Eigen::Matrix<double,21,1> reference;double age=0;
  fault=velocity_mode?juggling_hil_tick_acceleration(hil,t,q.data(),dq.data(),servo_acceleration.data(),next.data(),reference.data(),&age):juggling_hil_tick(hil,t,q.data(),dq.data(),tau.data(),next.data(),reference.data(),&age);
  if(!fault&&velocity_mode){
    std::array<double,21> ref;std::array<double,7> position,velocity;
    std::copy(reference.data(),reference.data()+21,ref.begin());std::copy(q.data(),q.data()+7,position.begin());std::copy(dq.data(),dq.data()+7,velocity.begin());
    velocity_command=tracker.step(ref,position,velocity,.001,false);
  }else if(!fault){MotionGuard::M mass=robot.mass(q);Joint bias=robot.bias(q,dq);
    Joint acceleration=reference.tail<7>()+100.*(reference.head<7>()-q)+20.*(reference.segment<7>(7)-dq);
    next=guard.filter(mass,bias,mass*acceleration+bias,tau,.001);}
  double us=std::chrono::duration<double,std::micro>(Clock::now()-entered).count();max_callback=std::max(max_callback,us);
  log<<t<<','<<us<<','<<age<<','<<std::chrono::duration<double,std::micro>(entered-deadline).count()<<','<<fault<<'\n';if(fault)break;
  // Two 0.5 ms RK4 steps under the total torque applied during this 1 ms tick.
  if(velocity_mode){
    // Illustrative internal velocity-servo lag, not an identified SDK model.
    // Integrate its first-order response exactly at the same 0.5 ms plant step.
    constexpr double lag=.005,h=.0005;Joint desired=Eigen::Map<const Joint>(velocity_command.data());
    for(int sub=0;sub<2;++sub){double decay=std::exp(-h/lag);Joint delta=dq-desired;q+=h*desired+lag*(1-decay)*delta;dq=desired+decay*delta;}
    servo_acceleration=(desired-dq)/lag;tau=robot.mass(q)*servo_acceleration+robot.bias(q,dq);
    continue;
  }
  tau=next;
  auto acceleration=[&](const Joint& q0,const Joint& v){return robot.mass(q0).ldlt().solve(tau-robot.bias(q0,v)).eval();};
  for(int sub=0;sub<2;++sub){constexpr double h=.0005;Joint a1=acceleration(q,dq),v2=dq+.5*h*a1,q2=q+.5*h*dq;
   Joint a2=acceleration(q2,v2),v3=dq+.5*h*a2,q3=q+.5*h*v2,a3=acceleration(q3,v3),v4=dq+h*a3,q4=q+h*v3,a4=acceleration(q4,v4);
   q+=h/6*(dq+2*v2+2*v3+v4);dq+=h/6*(a1+2*a2+2*a3+a4);}
 }
 juggling_hil_stop(hil);if(juggling_hil_save(hil,argv[2],error,sizeof(error)))std::cerr<<error<<'\n';juggling_hil_destroy(hil);
 std::cout<<"{\"fault\":"<<fault<<",\"max_callback_us\":"<<max_callback<<"}\n";return fault?1:0;
}

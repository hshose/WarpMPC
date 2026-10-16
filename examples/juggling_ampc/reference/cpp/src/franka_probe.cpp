// Read-only FCI preflight: never calls control, unlock, recovery, or load setters.
#include <franka/robot.h>
#include <franka/exception.h>
#include "model_config.hpp"
#include "json_text.hpp"
#include <sys/resource.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc,char** argv){
  if(argc!=3){std::cout<<"Read-only probe. Usage: juggling_franka_probe ROBOT OUTPUT_DIR\n";return argc==1?0:2;}
  std::filesystem::create_directories(argv[2]);const std::string directory=argv[2];
  std::ofstream result(directory+"/probe.json");result<<std::setprecision(17);
  int realtime=0;std::ifstream("/sys/kernel/realtime")>>realtime;
  rlimit priority{},memory{};getrlimit(RLIMIT_RTPRIO,&priority);getrlimit(RLIMIT_MEMLOCK,&memory);
  result<<"{\"motion_commanded\":false,\"realtime_kernel\":"<<(realtime==1?"true":"false")
        <<",\"rtprio_soft\":"<<priority.rlim_cur<<",\"memlock_soft_bytes\":"<<memory.rlim_cur;
  try{
    franka::Robot robot(argv[1]);auto state=robot.readOnce();
    std::ofstream(directory+"/robot_state.json")<<std::setprecision(17)<<state<<'\n';
    double position_error=0,velocity=0;
    for(int j=0;j<7;++j){position_error=std::max(position_error,std::abs(state.q[j]-juggling::generated::home[j]));velocity=std::max(velocity,std::abs(state.dq[j]));}
    using Clock=std::chrono::steady_clock;
    std::vector<std::array<double,3>> samples;samples.reserve(4000);
    auto start=Clock::now();double first=-1;
    robot.read([&](const franka::RobotState& current){
      double timestamp=current.time.toSec();if(first<0)first=timestamp;
      samples.push_back({timestamp-first,std::chrono::duration<double>(Clock::now()-start).count(),current.control_command_success_rate});
      return timestamp-first<2. && samples.size()<4000;
    });
    std::ofstream timing(directory+"/read_samples.csv");timing<<std::setprecision(17)<<"robot_time_s,wall_time_s,control_command_success_rate\n";
    for(const auto& sample:samples)timing<<sample[0]<<','<<sample[1]<<','<<sample[2]<<'\n';
    result<<",\"fci_read_success\":true,\"server_version\":"<<robot.serverVersion()<<",\"robot_mode\":"<<state.robot_mode
          <<",\"max_home_error_rad\":"<<position_error<<",\"max_velocity_rad_s\":"<<velocity
          <<",\"near_home_and_stationary\":"<<(position_error<=.08&&velocity<=.03?"true":"false")
          <<",\"configured_total_load_kg\":"<<state.m_total<<",\"samples\":"<<samples.size()<<"}\n";
    std::cout<<"FCI read succeeded; protocol "<<robot.serverVersion()<<", mode "<<state.robot_mode<<", home error "<<position_error<<" rad. No motion commanded.\n";
    return 0;
  }catch(const std::exception& error){
    result<<",\"fci_read_success\":false,\"error\":"<<juggling::json_text(error.what())<<"}\n";
    std::cerr<<error.what()<<'\n';return 1;
  }
}

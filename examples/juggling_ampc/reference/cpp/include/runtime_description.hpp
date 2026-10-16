#pragma once
#include "model_config.hpp"
#include <sstream>
#include <iomanip>
namespace juggling {
inline std::string runtime_tuning_description(){
#ifdef JUGGLING_PARAMETRIC_FACTORS
  namespace g=generated;std::ostringstream s;s<<std::setprecision(17);
  s<<"{\"parametric\":true,\"contact_center_grip_offset\":"<<g::contact_center_grip_offset
   <<",\"contact_edge_margin\":"<<g::contact_edge_margin<<",\"apex\":"<<g::apex
   <<",\"mass\":"<<g::mass<<",\"drag\":"<<g::drag<<",\"gravity\":["<<g::gravity_x<<','<<g::gravity_y<<','<<g::gravity_z
   <<"],\"osqp_iters\":"<<g::osqp_iters<<",\"sqp_iters\":"<<g::sqp_iters<<",\"rho\":"<<g::rho
   <<",\"control_dt\":"<<g::control_dt<<",\"first_step\":"<<g::stage_steps[0]<<",\"horizon_s\":"<<g::stage_times.back()
   <<",\"servo_position_gain\":"<<g::servo_position_gain<<",\"servo_velocity_gain\":"<<g::servo_velocity_gain<<'}';return s.str();
#else
  return "{\"parametric\":false}";
#endif
}
}

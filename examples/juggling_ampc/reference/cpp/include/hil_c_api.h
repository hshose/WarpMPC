#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
void* juggling_hil_create(const char* factors,const double* q,const double* dq,const double* total_torque,
                         double max_command_age,char* error,size_t error_size);
/* For an internal position/velocity servo: map its reference acceleration
   through MPC-model RNEA outside the realtime callback. This is a modeled
   feedforward torque boundary, not a measured physical torque. */
void* juggling_hil_create_acceleration(const char* factors,const double* q,const double* dq,const double* reference_acceleration,
                                      double max_command_age,char* error,size_t error_size);
int juggling_hil_start(void* handle,char* error,size_t error_size);
/* Nonblocking, allocation-free callback entry. Returns 0 or a latched fault. */
int juggling_hil_tick(void* handle,double t,const double* q,const double* dq,const double* applied_total_torque,
                      double* planned_total_torque,double* reference_q_dq_ddq21,double* command_age);
int juggling_hil_tick_acceleration(void* handle,double t,const double* q,const double* dq,const double* reference_acceleration,
                                  double* planned_total_torque,double* reference_q_dq_ddq21,double* command_age);
void juggling_hil_stop(void* handle);
int juggling_hil_save(void* handle,const char* directory,char* error,size_t error_size);
void juggling_hil_destroy(void* handle);
#ifdef __cplusplus
}
#endif

#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* All arrays use doubles, column-major. MPC plan is 34 x 41. Error strings are
 * copied into the caller-owned buffer. Each handle is owned by one worker. */
void* juggling_create(const char* factor_directory,int compiled,char* error,size_t error_size);
void juggling_destroy(void* handle);
/* Optimized variants export this optional symbol. Set cost references once,
 * before initialization; xyz is relative to the compiled nominal face origin. */
int juggling_set_reference(void* handle,const double* xyz3,const double* posture7,char* error,size_t error_size);
/* Optional native feature: local rebound velocity bias per j/8, norm <= 0.3.
 * Updated between solves; neither the plant nor other controller handles change. */
int juggling_set_contact_bias(void* handle,const double* bias3,char* error,size_t error_size);
int juggling_step(void* handle,const double* state23,const double* torque7,double elapsed,
                  double* plan1394,double* target7,double* metrics18,char* error,size_t error_size);
int juggling_dimensions(void* handle,int* sizes4);
int juggling_pattern(void* handle,int* hp,int* hi,int* ap,int* ai);
int juggling_linearize(void* handle,const double* z,const double* parameter,double* h,double* gradient,
                       double* a,double* constraints,double* cost,char* error,size_t error_size);
/* Algebraic nonlinear objective/constraints only; does not solve or shift. */
int juggling_value(void* handle,const double* z,const double* parameter,
                   double* constraints,double* cost,char* error,size_t error_size);
int juggling_set_warmstart(void* handle,const double* z,const double* dual,char* error,size_t error_size);
int juggling_get_warmstart(void* handle,double* z,double* dual);
int juggling_initialize(void* handle,const double* state,const double* torque,char* error,size_t error_size);
int juggling_shift(void* handle,double dt,const double* state,const double* torque,char* error,size_t error_size);
#ifdef __cplusplus
}
#endif

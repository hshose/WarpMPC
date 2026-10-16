#include "hil_c_api.h"
#include "hil_runtime.hpp"
#include <cstdio>
#include <algorithm>
static void error_message(char* out,size_t n,const char* error){if(out&&n)std::snprintf(out,n,"%s",error);}
extern "C" {
void* juggling_hil_create(const char* factors,const double* q,const double* dq,const double* torque,double age,char* error,size_t n){try{juggling::ArmFrame initial;std::copy(q,q+7,initial.q.begin());std::copy(dq,dq+7,initial.dq.begin());std::copy(torque,torque+7,initial.torque.begin());return new juggling::HilRuntime(factors,initial,age);}catch(const std::exception& e){error_message(error,n,e.what());return nullptr;}}
void* juggling_hil_create_acceleration(const char* factors,const double* q,const double* dq,const double* acceleration,double age,char* error,size_t n){try{juggling::ArmFrame initial;std::copy(q,q+7,initial.q.begin());std::copy(dq,dq+7,initial.dq.begin());std::copy(acceleration,acceleration+7,initial.reference_acceleration.begin());initial.acceleration_input=true;return new juggling::HilRuntime(factors,initial,age);}catch(const std::exception& e){error_message(error,n,e.what());return nullptr;}}
int juggling_hil_start(void* p,char* error,size_t n){try{static_cast<juggling::HilRuntime*>(p)->start();return 0;}catch(const std::exception& e){error_message(error,n,e.what());return -1;}}
int juggling_hil_tick(void* p,double t,const double* q,const double* dq,const double* torque,double* target,double* reference,double* age){if(!p)return -1;juggling::ArmFrame frame;frame.t=t;std::copy(q,q+7,frame.q.begin());std::copy(dq,dq+7,frame.dq.begin());std::copy(torque,torque+7,frame.torque.begin());std::array<double,7> output;std::array<double,21> ref;int status=static_cast<juggling::HilRuntime*>(p)->tick(frame,output,ref,*age);std::copy(output.begin(),output.end(),target);std::copy(ref.begin(),ref.end(),reference);return status;}
int juggling_hil_tick_acceleration(void* p,double t,const double* q,const double* dq,const double* acceleration,double* target,double* reference,double* age){if(!p)return -1;juggling::ArmFrame frame;frame.t=t;std::copy(q,q+7,frame.q.begin());std::copy(dq,dq+7,frame.dq.begin());std::copy(acceleration,acceleration+7,frame.reference_acceleration.begin());frame.acceleration_input=true;std::array<double,7> output;std::array<double,21> ref;int status=static_cast<juggling::HilRuntime*>(p)->tick(frame,output,ref,*age);std::copy(output.begin(),output.end(),target);std::copy(ref.begin(),ref.end(),reference);return status;}
void juggling_hil_stop(void* p){if(p)static_cast<juggling::HilRuntime*>(p)->stop();}
int juggling_hil_save(void* p,const char* directory,char* error,size_t n){try{static_cast<juggling::HilRuntime*>(p)->save(directory);return 0;}catch(const std::exception& e){error_message(error,n,e.what());return -1;}}
void juggling_hil_destroy(void* p){delete static_cast<juggling::HilRuntime*>(p);}
}

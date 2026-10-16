#include "c_api.h"
#include "controller.hpp"
#include <cstdio>
#include <algorithm>
static void message(char* out,size_t n,const char* text){if(out&&n)std::snprintf(out,n,"%s",text);}
static juggling::Controller& controller(void* p){if(!p)throw std::runtime_error("Null controller handle");return *static_cast<juggling::Controller*>(p);}
extern "C" {
void* juggling_create(const char* directory,int compiled,char* error,size_t n){try{return new juggling::Controller(directory,compiled!=0);}catch(const std::exception& e){message(error,n,e.what());return nullptr;}}
void juggling_destroy(void* p){delete static_cast<juggling::Controller*>(p);}
#ifdef JUGGLING_RUNTIME_REFERENCE
int juggling_set_reference(void* p,const double* xyz,const double* q,char* error,size_t n){try{
 controller(p).set_reference(Eigen::Map<const Eigen::Vector3d>(xyz),Eigen::Map<const juggling::Joint>(q));return 0;
 }catch(const std::exception& e){message(error,n,e.what());return -1;}}
int juggling_set_contact_bias(void* p,const double* bias,char* error,size_t n){try{
 if(!bias)throw std::runtime_error("Null contact bias pointer");
 controller(p).set_contact_bias(Eigen::Map<const Eigen::Vector3d>(bias));return 0;
 }catch(const std::exception& e){message(error,n,e.what());return -1;}}
#endif
int juggling_step(void* p,const double* state,const double* torque,double dt,double* plan,double* target,double* metrics,char* error,size_t n){try{
 auto& c=controller(p);auto i=c.step(Eigen::Map<const juggling::State>(state),Eigen::Map<const juggling::Joint>(torque),dt);
 std::copy(c.physical_plan().data(),c.physical_plan().data()+c.variables(),plan);std::copy(c.target().data(),c.target().data()+7,target);
 double data[]={i.total_ms,i.shift_ms,i.data_ms,i.qp_ms,i.merit_ms,i.cost,i.violation,i.bound_violation,i.alpha,i.primal_residual,i.dual_residual,i.clipping,double(i.iterations),double(i.status),double(i.warm_dual_nonzeros),double(i.fallback_stage),double(i.evaluator_threads),double(i.sqp_iterations)};std::copy(data,data+18,metrics);return 0;
 }catch(const std::exception& e){message(error,n,e.what());return -1;}}
int juggling_dimensions(void* p,int* sizes){auto& c=controller(p);sizes[0]=c.variables();sizes[1]=c.nonlinear_rows();sizes[2]=c.hessian().values.size();sizes[3]=c.constraint_matrix().values.size();return 0;}
int juggling_pattern(void* p,int* hp,int* hi,int* ap,int* ai){auto& c=controller(p);auto& h=c.hessian();auto& a=c.constraint_matrix();std::copy(h.columns.begin(),h.columns.end(),hp);std::copy(h.rows.begin(),h.rows.end(),hi);std::copy(a.columns.begin(),a.columns.end(),ap);std::copy(a.rows.begin(),a.rows.end(),ai);return 0;}
int juggling_linearize(void* p,const double* z,const double* parameter,double* h,double* gradient,double* a,double* constraints,double* cost,char* error,size_t n){try{auto& c=controller(p);c.linearize(Eigen::Map<const juggling::Vector>(z,c.variables()),Eigen::Map<const Eigen::Matrix<double,30,1>>(parameter));std::copy(c.hessian().values.begin(),c.hessian().values.end(),h);std::copy(c.constraint_matrix().values.begin(),c.constraint_matrix().values.end(),a);std::copy(c.gradient().data(),c.gradient().data()+c.variables(),gradient);std::copy(c.constraints().data(),c.constraints().data()+c.nonlinear_rows(),constraints);*cost=c.cost();return 0;}catch(const std::exception& e){message(error,n,e.what());return -1;}}
int juggling_value(void* p,const double* z,const double* parameter,double* constraints,double* cost,char* error,size_t n){try{
 auto& c=controller(p);juggling::Vector values(c.nonlinear_rows());
 *cost=c.value(Eigen::Map<const juggling::Vector>(z,c.variables()),Eigen::Map<const Eigen::Matrix<double,30,1>>(parameter),values);
 std::copy(values.data(),values.data()+values.size(),constraints);return 0;
 }catch(const std::exception& e){message(error,n,e.what());return -1;}}
int juggling_set_warmstart(void* p,const double* z,const double* dual,char* error,size_t n){try{controller(p).set_warmstart(z,dual);return 0;}catch(const std::exception& e){message(error,n,e.what());return -1;}}
int juggling_get_warmstart(void* p,double* z,double* dual){auto& c=controller(p);std::copy(c.solution().data(),c.solution().data()+c.variables(),z);std::copy(c.dual().data(),c.dual().data()+c.dual().size(),dual);return 0;}
int juggling_initialize(void* p,const double* x,const double* tau,char* error,size_t n){try{controller(p).initialize(Eigen::Map<const juggling::State>(x),Eigen::Map<const juggling::Joint>(tau));return 0;}catch(const std::exception& e){message(error,n,e.what());return -1;}}
int juggling_shift(void* p,double dt,const double* x,const double* tau,char* error,size_t n){try{controller(p).shift(dt,Eigen::Map<const juggling::State>(x),Eigen::Map<const juggling::Joint>(tau));return 0;}catch(const std::exception& e){message(error,n,e.what());return -1;}}
}

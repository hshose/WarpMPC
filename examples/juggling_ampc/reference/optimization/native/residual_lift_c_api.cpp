#include "controller.hpp"
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
#include "residual_lift.hpp"
#endif
#include <cstdio>
#include <algorithm>
#include <stdexcept>
namespace {
juggling::Controller& controller(void* p){if(!p)throw std::runtime_error("Null controller handle");return *static_cast<juggling::Controller*>(p);}
void message(char* out,size_t n,const char* text){if(out&&n)std::snprintf(out,n,"%s",text);}
}
extern "C" {
// Twelve int64 entries; see residual_lift.md. L nnz is -1 before setup or
// without the separately verified private diagnostic bridge.
int juggling_qp_diagnostics(void* p,long long* counts,char* error,size_t n){try{
  if(!counts)throw std::runtime_error("Null QP diagnostic destination");const auto data=controller(p).qp_diagnostics();
  std::copy(data.begin(),data.end(),counts);return 0;
}catch(const std::exception& e){message(error,n,e.what());return -1;}}
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
int juggling_lift_pattern(void* p,int* pp,int* pi,int* ap,int* ai,char* error,size_t n){try{
  const auto& lift=controller(p).residual_lift();
  std::copy(lift.P.columns.begin(),lift.P.columns.end(),pp);std::copy(lift.P.rows.begin(),lift.P.rows.end(),pi);
  std::copy(lift.A.columns.begin(),lift.A.columns.end(),ap);std::copy(lift.A.rows.begin(),lift.A.rows.end(),ai);return 0;
}catch(const std::exception& e){message(error,n,e.what());return -1;}}
// Does not solve or mutate the accepted primal/dual, event cache or OSQP state.
// z/parameter use the existing physical API layout. P/A are CSC values in
// juggling_lift_pattern order; all remaining outputs are contiguous vectors.
int juggling_lift_prepare(void* p,const double* z,const double* parameter,double* pv,double* q,double* av,
    double* l,double* u,double* residual,double* warm_x,double* warm_y,double* cost,char* error,size_t n){try{
  auto& c=controller(p);c.prepare_lifted_qp(Eigen::Map<const juggling::Vector>(z,c.variables()),Eigen::Map<const Eigen::Matrix<double,30,1>>(parameter));
  const auto& lift=c.residual_lift();std::copy(lift.P.values.begin(),lift.P.values.end(),pv);std::copy(lift.A.values.begin(),lift.A.values.end(),av);
  auto copy=[](const juggling::Vector& from,double* to){std::copy(from.data(),from.data()+from.size(),to);};
  copy(lift.q,q);copy(lift.l,l);copy(lift.u,u);copy(lift.residual,residual);copy(lift.warm_primal,warm_x);copy(lift.warm_dual,warm_y);*cost=c.cost();return 0;
}catch(const std::exception& e){message(error,n,e.what());return -1;}}
#endif
}

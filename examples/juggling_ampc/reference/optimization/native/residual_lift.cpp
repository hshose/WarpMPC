#include "residual_lift.hpp"
#include <map>
#include <set>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#ifdef JUGGLING_LIFT_LINK_RHO_BRIDGE
#include "residual_lift_rho.h"
#endif

namespace juggling {
ResidualLift::ResidualLift(int n,int m,double proximal):physical_variables_(n),physical_rows_(m),proximal_(proximal){
  if(n<=0||m<n||!std::isfinite(proximal)||proximal<0.)throw std::runtime_error("Invalid residual-lift dimensions/proximal");
}
int ResidualLift::add_block(int rows,const std::vector<std::pair<int,int>>& entries){
  if(finalized_||rows<0)throw std::runtime_error("Invalid residual-lift block");
  std::set<std::pair<int,int>> seen;
  for(auto e:entries)if(e.first<0||e.first>=rows||e.second<0||e.second>=physical_variables_||!seen.insert(e).second)
    throw std::runtime_error("Invalid/duplicate residual-lift Jacobian entry");
  blocks_.push_back({residual_rows_,rows,entries,{}});residual_rows_+=rows;return blocks_.size()-1;
}
int ResidualLift::add_dense_block(int rows,const std::vector<int>& columns){
  std::vector<std::pair<int,int>> entries;
  for(int column:columns)for(int row=0;row<rows;++row)entries.emplace_back(row,column);
  return add_block(rows,entries);
}
void ResidualLift::finalize(const Sparse& physical_a){
  if(finalized_||int(physical_a.columns.size())!=physical_variables_+1)throw std::runtime_error("Invalid residual-lift finalization");
  using Key=std::pair<int,int>;std::set<Key> pattern;
  for(int c=0;c<physical_variables_;++c)for(int j=physical_a.columns[c];j<physical_a.columns[c+1];++j)
    pattern.emplace(c,physical_a.rows[j]);
  for(const auto& b:blocks_)for(auto e:b.entries)pattern.emplace(e.second,physical_rows_+b.row+e.first);
  for(int r=0;r<residual_rows_;++r)pattern.emplace(physical_variables_+r,physical_rows_+r);
  const int n=physical_variables_+residual_rows_,m=physical_rows_+residual_rows_;
  std::map<Key,int> lookup;A.columns.resize(n+1);int column=0;
  for(auto key:pattern){while(column<=key.first)A.columns[column++]=A.rows.size();lookup[key]=A.rows.size();A.rows.push_back(key.second);}
  while(column<=n)A.columns[column++]=A.rows.size();A.values.assign(A.rows.size(),0.);
  for(int c=0;c<physical_variables_;++c)for(int j=physical_a.columns[c];j<physical_a.columns[c+1];++j)
    physical_dest_.push_back(lookup.at({c,physical_a.rows[j]}));
  for(auto& b:blocks_)for(auto e:b.entries)b.destination.push_back(lookup.at({e.second,physical_rows_+b.row+e.first}));
  for(int r=0;r<residual_rows_;++r)A.values[lookup.at({physical_variables_+r,physical_rows_+r})]=1.;
  P.columns.resize(n+1);P.rows.resize(n);P.values.resize(n);
  for(int i=0;i<n;++i){P.columns[i]=i;P.rows[i]=i;P.values[i]=i<physical_variables_?proximal_:1.;}P.columns[n]=n;
  residual=Vector::Zero(residual_rows_);q=Vector::Zero(n);l=Vector::Zero(m);u=l;warm_primal=q;warm_dual=l;finalized_=true;
}
void ResidualLift::install_link_rho(OSQPSolver* solver,double multiplier){
  if(multiplier==1.)return; // Exact legacy policy: no allocation/workspace access.
  if(!finalized_||!std::isfinite(multiplier)||multiplier<1.||multiplier>1000.)
    throw std::runtime_error("Invalid residual-link rho multiplier");
#ifdef JUGGLING_LIFT_LINK_RHO_BRIDGE
  // Allocate at most once; ordinary data updates reconstruct the complete
  // policy in this scratch, while exact ADMM retries never call this method.
  rho_type_scratch_.resize(rows());rho_scratch_.resize(rows());
  if(juggling_osqp_install_lift_link_rho(solver,physical_rows_,rows(),multiplier,
       rho_type_scratch_.data(),rho_scratch_.data(),nullptr))
    throw std::runtime_error("Residual-link rho installation failed");
#else
  throw std::runtime_error("Residual-link rho requires the pinned private OSQP bridge");
#endif
}
void ResidualLift::clear(){
  residual.setZero();for(const auto& b:blocks_)for(int d:b.destination)A.values[d]=0.;
}
void ResidualLift::set_sparse(int id,const Vector& r,const std::vector<double>& j){
  if(id<0)return;auto& b=blocks_.at(id);
  if(r.size()!=b.size||j.size()!=b.destination.size())throw std::runtime_error("Residual-lift sparse block dimensions mismatch");
  residual.segment(b.row,b.size)=r;
  for(size_t k=0;k<j.size();++k)A.values[b.destination[k]]=-j[k];
}
void ResidualLift::set_dense(int id,const Vector& r,const Matrix& j){
  if(id<0)return;auto& b=blocks_.at(id);
  if(r.size()!=b.size||j.rows()!=b.size||j.size()!=int(b.destination.size()))throw std::runtime_error("Residual-lift dense block dimensions mismatch");
  residual.segment(b.row,b.size)=r;
  for(size_t k=0;k<b.destination.size();++k)A.values[b.destination[k]]=-j.data()[k];
}
void ResidualLift::prepare(const Sparse& physical_a,const Vector& physical_l,const Vector& physical_u,const Vector& z,const Vector& dual){
  if(physical_l.size()!=physical_rows_||physical_u.size()!=physical_rows_||z.size()!=physical_variables_||dual.size()!=physical_rows_)
    throw std::runtime_error("Residual-lift QP input dimensions mismatch");
  for(size_t j=0;j<physical_dest_.size();++j)A.values[physical_dest_[j]]=physical_a.values[j];
  q.head(physical_variables_)=-proximal_*z;
  l.head(physical_rows_)=physical_l;u.head(physical_rows_)=physical_u;
  l.tail(residual_rows_)=residual;
  // A's lower-left block is -J: equality rhs = r(z0)-J*z0.
  for(int c=0;c<physical_variables_;++c)for(int j=A.columns[c];j<A.columns[c+1];++j)
    if(A.rows[j]>=physical_rows_)l[A.rows[j]]+=A.values[j]*z[c];
  u.tail(residual_rows_)=l.tail(residual_rows_);
  warm_primal.head(physical_variables_)=z;warm_primal.tail(residual_rows_)=residual;
  warm_dual.head(physical_rows_)=dual;warm_dual.tail(residual_rows_)=-residual;
  if(!residual.allFinite()||!q.allFinite()||!l.allFinite()||!u.allFinite()||
      !std::all_of(A.values.begin(),A.values.end(),[](double v){return std::isfinite(v);}))
    throw std::runtime_error("Non-finite lifted QP data");
}
}

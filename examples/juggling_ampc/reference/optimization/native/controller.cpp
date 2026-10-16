#include "controller.hpp"
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
#include "residual_lift.hpp"
#endif
#ifdef JUGGLING_LIFT_PRIVATE_DIAGNOSTICS
extern "C" long long juggling_osqp_factor_nnz(const OSQPSolver*);
#endif
#include <cmath>
#include "model_config.hpp"
#include "ball_gravity.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <numeric>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <utility>
#ifdef JUGGLING_EXPERIMENTAL_GROUP_RHO
#include "osqp_group_rho.h"
#endif

namespace juggling {
namespace g=generated;
#ifdef JUGGLING_PREDICTION_CAP_NODES
static constexpr int restitution_cap_nodes=JUGGLING_PREDICTION_CAP_NODES;
static_assert(restitution_cap_nodes>=0&&restitution_cap_nodes<=g::N,"Invalid restitution cap prefix");
#else
static constexpr int restitution_cap_nodes=0;
#endif
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static Eigen::Matrix3d skew(const Eigen::Vector3d& v){Eigen::Matrix3d s;s<<0,-v.z(),v.y(),v.z(),0,-v.x(),-v.y(),v.x(),0;return s;}
template<size_t N> static Eigen::Map<const Eigen::Matrix<double,N,1>> map(const std::array<double,N>& a){return Eigen::Map<const Eigen::Matrix<double,N,1>>(a.data());}
static int source(double time){return std::clamp(int(std::upper_bound(g::stage_times.begin(),g::stage_times.end(),time+1e-12)-g::stage_times.begin())-1,0,g::N);}
#ifdef JUGGLING_KINEMATIC_DYNAMIC_PREFIX
static constexpr int kinematic_prefix=JUGGLING_KINEMATIC_DYNAMIC_PREFIX;
static_assert(kinematic_prefix>=1&&kinematic_prefix<g::N,"Kinematic tail requires at least one dynamic interval and a tail");
// Stage P converts tau[P] to a[P+1]; later stages contain only kinematics.
static bool kinematic_stage(int stage){return stage>=kinematic_prefix;}
static bool acceleration_node(int node){return node>kinematic_prefix;}
#endif
#if defined(JUGGLING_LEGACY_CONTACT_PLANE) && JUGGLING_LEGACY_CONTACT_PLANE
static constexpr std::array<int,2> contact_plane_residual_ids{{51,52}};
static constexpr std::array<int,18> contact_plane_constraint_ids{{0,1,2,3,4,5,6,7,8,16,17,18,19,20,21,22,24,25}};
#endif

Evaluation::Evaluation(const std::string& directory,const std::string& name,bool compiled){
#ifdef JUGGLING_PARAMETRIC_FACTORS
  if(g::loaded_runtime_values_path.empty())throw std::runtime_error("Set JUGGLING_RUNTIME_VALUES before loading a parametric controller");
  if(!std::filesystem::equivalent(directory+"/runtime_values.txt",g::loaded_runtime_values_path))
    throw std::runtime_error("One runtime tuning profile per process; start a fresh process with its JUGGLING_RUNTIME_VALUES");
#endif
  std::string individual=directory+"/"+name+".so";
  function=compiled?casadi::external(name,std::filesystem::exists(individual)?individual:directory+"/libjuggling_factors.so"):casadi::Function::load(directory+"/"+name+".casadi");
  args_.resize(function.sz_arg(),nullptr);results_.resize(function.sz_res(),nullptr);
  iw_.resize(function.sz_iw());work_.resize(function.sz_w());output.resize(function.n_out());
  for(int i=0;i<function.n_out();++i){output[i].resize(function.nnz_out(i));results_[i]=output[i].data();}
  memory_=function.checkout();
}
Evaluation::~Evaluation(){function.release(memory_);}
void Evaluation::call(const std::vector<const double*>& inputs){
  std::copy(inputs.begin(),inputs.end(),args_.begin());
#ifdef JUGGLING_PARAMETRIC_FACTORS
  if(function.n_in()==static_cast<int>(inputs.size())+1)args_[inputs.size()]=g::factor_parameters.data();
#endif
  if(function(args_.data(),results_.data(),iw_.data(),work_.data(),memory_))throw std::runtime_error("CasADi factor evaluation failed");
}
Matrix Evaluation::dense(int o) const {
  auto s=function.sparsity_out(o);Matrix out=Matrix::Zero(s.size1(),s.size2());
  for(casadi_int c=0;c<s.size2();++c)for(casadi_int i=s.colind(c);i<s.colind(c+1);++i)out(s.row(i),c)=output[o][i];
  return out;
}
RobotModel::RobotModel(const std::string& d,bool c):mass_(d,g::mass_name,c),bias_(d,g::bias_name,c),rnea_(d,g::rnea_name,c),fk_(d,g::fk_name,c),contact_(d,g::contact_fk_name,c){}
Matrix RobotModel::mass(const Joint& q){mass_.call({q.data()});return mass_.dense(0);}
Joint RobotModel::bias(const Joint& q,const Joint& dq){bias_.call({q.data(),dq.data()});return bias_.dense(0);}
Joint RobotModel::rnea(const Joint& q,const Joint& dq,const Joint& a){rnea_.call({q.data(),dq.data(),a.data()});return rnea_.dense(0);}
std::array<Matrix,5> RobotModel::contact_fk(const Joint& q){contact_.call({q.data()});return {contact_.dense(0),contact_.dense(1),contact_.dense(2),contact_.dense(3),contact_.dense(4)};}
Eigen::Vector3d RobotModel::normal(const Joint& q){fk_.call({q.data()});return fk_.dense(1);}
#if defined(JUGGLING_RAMP_PREDICTION) && JUGGLING_RAMP_PREDICTION
// Only the full-ramp prediction build uses this shifted surface. Keep
// contact_fk itself unchanged: empirical plant geometry owns its own offsets.
static void prediction_plane(std::array<Matrix,5>& f){
  Eigen::Vector3d offset=JUGGLING_PREDICTION_PLANE_OFFSET_M*f[1].col(0);
  f[0].col(0)+=offset;f[2]-=skew(offset)*f[3];
}
#endif
Geometry RobotModel::geometry(const State& x){
#if defined(JUGGLING_RAMP_PREDICTION) && JUGGLING_RAMP_PREDICTION
  auto f=contact_fk(x.segment<7>(6));prediction_plane(f);
  Eigen::Vector3d n=f[1].col(0),rel=x.head<3>()-f[0].col(0)+map(g::origin);
  Matrix N=-skew(n)*f[3];Eigen::Vector3d tangent=rel-n*(n.dot(rel));
  return {n.dot(rel)-g::radius,tangent.norm(),n.dot(x.segment<3>(3))-(n.transpose()*f[2]-rel.transpose()*N).dot(x.segment<7>(13)),n};
#else
  fk_.call({x.data()+6});Eigen::Vector3d p=fk_.dense(0),n=fk_.dense(1),rel=x.head<3>()-p+map(g::origin);
  Matrix J=fk_.dense(2),N=fk_.dense(3);Eigen::Vector3d tangent=rel-n*(n.dot(rel));
  return {n.dot(rel)-g::radius,tangent.norm(),n.dot(x.segment<3>(3))-(n.transpose()*J-rel.transpose()*N).dot(x.segment<7>(13)),n};
#endif
}


// Persistent worker pool. Factor outputs/workspaces are distinct, and sparse
// reduction remains in stage order, so thread count does not change arithmetic.
class EvaluationPool {
 public:
  struct Job {Evaluation* evaluator;int column;double step;};
  EvaluationPool(int count,const std::vector<Job>& jobs):jobs_(jobs){
    for(int i=0;i<count;++i)workers_.emplace_back([this,i,count]{run(i,count);});
  }
  ~EvaluationPool(){{std::lock_guard<std::mutex> lock(mutex_);stop_=true;}wake_.notify_all();for(auto& w:workers_)w.join();}
  void evaluate(const double* z,const double* parameter){
    std::unique_lock<std::mutex> lock(mutex_);z_=z;p_=parameter;left_=workers_.size();error_=nullptr;++generation_;wake_.notify_all();
    done_.wait(lock,[this]{return left_.load(std::memory_order_acquire)==0;});if(error_)std::rethrow_exception(error_);
  }
 private:
  std::vector<Job> jobs_;std::vector<std::thread> workers_;std::mutex mutex_;std::condition_variable wake_,done_;
  std::atomic<size_t> left_{0};size_t generation_=0;bool stop_=false;const double *z_=nullptr,*p_=nullptr;std::exception_ptr error_;
  void run(int worker,int count){size_t seen=0;for(;;){
    {std::unique_lock<std::mutex> lock(mutex_);wake_.wait(lock,[&]{return stop_||generation_!=seen;});if(stop_)return;seen=generation_;}
    for(size_t j=worker;j<jobs_.size();j+=count){
      try{jobs_[j].evaluator->call({z_+jobs_[j].column,p_});}catch(...){std::lock_guard<std::mutex> lock(mutex_);if(!error_)error_=std::current_exception();}
    }
    if(left_.fetch_sub(1,std::memory_order_acq_rel)==1){std::lock_guard<std::mutex> lock(mutex_);done_.notify_one();}
  }}
};

Controller::Controller(const std::string& d,bool c):robot(d,c),first_data_(d,"first_data",c),stage_data_(d,"stage_data",c),last_data_(d,"last_data",c),first_value_(d,"first_value",c),stage_value_(d,"stage_value",c),last_value_(d,"last_value",c),n_(g::NZ*(g::N+1)),m_(g::CONSTRAINTS+restitution_cap_nodes){
#ifdef JUGGLING_KINEMATIC_DYNAMIC_PREFIX
#if defined(JUGGLING_RAMP_PREDICTION) && JUGGLING_RAMP_PREDICTION
  throw std::runtime_error("Kinematic tail does not support ramp prediction");
#endif
#ifdef JUGGLING_PREDICTION_PLANE_OFFSET_M
  if(JUGGLING_PREDICTION_PLANE_OFFSET_M!=0.)throw std::runtime_error("Kinematic tail currently requires zero prediction-plane offset");
#endif
#ifdef JUGGLING_SOLVER_WARMSTART_ARM_SHIFT
  if(JUGGLING_SOLVER_WARMSTART_ARM_SHIFT!=0.)throw std::runtime_error("Kinematic tail does not support arm-shift repair");
#endif
#ifdef JUGGLING_SOLVER_WARMSTART_REPAIR
  if(JUGGLING_SOLVER_WARMSTART_REPAIR!=0.)throw std::runtime_error("Kinematic tail currently requires warmstart_repair=0");
#endif
#ifdef JUGGLING_TAIL_ENDPOINT_EXTRA
  if(JUGGLING_TAIL_ENDPOINT_EXTRA!=0.)throw std::runtime_error("Kinematic tail does not support endpoint elasticity");
#endif
#ifdef JUGGLING_BOUNDED_MULTISTART_MODE
  if(JUGGLING_BOUNDED_MULTISTART_MODE!=0.)throw std::runtime_error("Kinematic tail does not support bounded multistart");
#endif
#endif
#ifdef JUGGLING_FIRST_EVENT_CENTER_WEIGHT
  configure_first_event_center();
#endif
#ifdef JUGGLING_SOLVER_LANDING_WEIGHT
  const double landing_weight=JUGGLING_SOLVER_LANDING_WEIGHT;
#ifdef JUGGLING_SOLVER_LANDING_WINDOW
  const double landing_window=JUGGLING_SOLVER_LANDING_WINDOW;
#else
  constexpr double landing_window=.12;
#endif
  if(!std::isfinite(landing_weight)||landing_weight<0.||!std::isfinite(landing_window)||landing_window<.02||landing_window>.3)
    throw std::runtime_error("Invalid landing guidance weight/window");
#ifndef JUGGLING_PARAMETRIC_FACTORS
  if(landing_weight>0.)throw std::runtime_error("Landing guidance requires parametric contact geometry");
#endif
#endif
  z_=Vector::Zero(n_);dual_=Vector::Zero(m_+n_);lower_.resize(n_);upper_.resize(n_);cl_.resize(m_);cu_.resize(m_);
  gradient_.resize(n_);constraints_.resize(m_);q_.resize(n_);l_.resize(m_+n_);u_.resize(m_+n_);trial_.resize(n_);trial_constraints_.resize(m_);
  lower_=map(g::all_lower);upper_=map(g::all_upper);
#ifdef JUGGLING_KINEMATIC_DYNAMIC_PREFIX
  // Catch a stale torque-box export before a mixed-unit seed can be used.
  for(int k=kinematic_prefix+1;k<=g::N;++k)for(int joint=0;joint<7;++joint){
    const int index=k*g::NZ+g::NX+joint;
    const double low=lower_[index]*g::scale[g::NX+joint]+g::offset[g::NX+joint];
    const double high=upper_[index]*g::scale[g::NX+joint]+g::offset[g::NX+joint];
    if(!std::isfinite(low)||!std::isfinite(high)||low>=0.||high<=0.
       ||low < -g::acceleration_limit[joint]-1e-12||high > g::acceleration_limit[joint]+1e-12)
      throw std::runtime_error("Kinematic tail requires physical acceleration boxes on nodes P+1..N");
  }
#endif
  cl_.head(g::CONSTRAINTS)=map(g::constraint_lower);cu_.head(g::CONSTRAINTS)=map(g::constraint_upper);
#ifdef JUGGLING_PREDICTION_CAP_NODES
  if(restitution_cap_nodes>0){
    restitution_cap_normalization_=std::sqrt(g::objective_scale)*g::comp_weight;
    if(!std::isfinite(restitution_cap_normalization_)||restitution_cap_normalization_<=0.
       ||!std::isfinite(JUGGLING_PREDICTION_CAP_VALUE)||JUGGLING_PREDICTION_CAP_VALUE<0.)
      throw std::runtime_error("Invalid restitution cap normalization/bound");
    cl_.tail(restitution_cap_nodes).setConstant(-std::numeric_limits<double>::infinity());
    cu_.tail(restitution_cap_nodes).setConstant(JUGGLING_PREDICTION_CAP_VALUE);
  }
#endif
#ifdef JUGGLING_SOLVER_CONTACT_RHO_MULTIPLIER
  const double contact_rho_multiplier=JUGGLING_SOLVER_CONTACT_RHO_MULTIPLIER;
  if(!std::isfinite(contact_rho_multiplier)||contact_rho_multiplier<1.||contact_rho_multiplier>1000.)
    throw std::runtime_error("Contact rho multiplier must be within [1,1000]");
  if(contact_rho_multiplier!=1.){
#if !defined(JUGGLING_EXPERIMENTAL_GROUP_RHO) || !defined(JUGGLING_GROUP_RHO_PROFILE)
    throw std::runtime_error("Contact rho multiplier requires an isolated experimental_group_rho build/profile");
#else
    if(JUGGLING_SOLVER_ADAPTIVE_RHO!=0.||JUGGLING_SOLVER_RHO_IS_VEC!=1.)
      throw std::runtime_error("Group rho requires adaptive_rho=0 and rho_is_vec=1");
    contact_rho_multiplier_=contact_rho_multiplier;
    group_rho_selected_.assign(m_+n_,0);group_rho_types_.resize(m_+n_);group_rho_values_.resize(m_+n_);
    for(int row=0;row<g::CONSTRAINTS;++row)
      if(g::constraint_ids[row]==g::contact_begin+2)group_rho_selected_[row]=1;
    for(int row=g::CONSTRAINTS;row<m_;++row)group_rho_selected_[row]=1;
    const char* path=std::getenv("JUGGLING_GROUP_RHO_TRACE");
    if(path&&*path){group_rho_trace_=std::make_unique<std::ofstream>(path,std::ios::out|std::ios::app);
      if(!*group_rho_trace_)throw std::runtime_error("Cannot open group-rho trace");group_rho_trace_->precision(17);}
#endif
  }
#endif
  layouts_.push_back({&first_data_,&first_value_,0,0,g::stage_steps[0],{},{}});
  for(int k=0;k<g::N;++k){
    stage_evaluators_.push_back(std::make_unique<Evaluation>(d,g::stage_data_names[k],c));
    stage_values_.push_back(std::make_unique<Evaluation>(d,g::stage_value_names[k],c));
    layouts_.push_back({stage_evaluators_.back().get(),stage_values_.back().get(),k*g::NZ,g::stage_offsets[k],g::stage_steps[k],{},{}});
    if(k<restitution_cap_nodes)layouts_.back().restitution_cap_row=g::CONSTRAINTS+k;
  }
  layouts_.push_back({&last_data_,&last_value_,g::N*g::NZ,g::stage_offsets[g::N],g::stage_steps.back(),{},{}});
#if defined(JUGGLING_LEGACY_CONTACT_PLANE) && JUGGLING_LEGACY_CONTACT_PLANE
#ifndef JUGGLING_PARAMETRIC_FACTORS
#error Legacy contact-plane prediction requires parametric factors
#endif
#if defined(JUGGLING_RAMP_PREDICTION) && JUGGLING_RAMP_PREDICTION
#error Legacy contact-plane prediction cannot be combined with full-ramp factors
#endif
  if(!std::isfinite(JUGGLING_PREDICTION_PLANE_OFFSET_M)||std::abs(JUGGLING_PREDICTION_PLANE_OFFSET_M)>.02)
    throw std::runtime_error("Invalid legacy prediction-only contact-plane offset");
  contact_plane_data_=std::make_unique<Evaluation>(d,"contact_plane_prediction_data",c);
  contact_plane_value_=std::make_unique<Evaluation>(d,"contact_plane_prediction_value",c);
  for(auto* evaluator:{contact_plane_data_.get(),contact_plane_value_.get()}){
    const auto& fn=evaluator->function;
    const int sizes[]={2*g::NZ,g::NP,static_cast<int>(g::factor_parameters.size()),1};
    if(fn.n_in()!=4)throw std::runtime_error("Contact-plane helper requires four explicit inputs");
    for(int i=0;i<4;++i)if(fn.size1_in(i)!=sizes[i]||fn.size2_in(i)!=1)
      throw std::runtime_error("Contact-plane helper input dimensions mismatch");
  }
  const auto& plane_data=contact_plane_data_->function;const auto& plane_value=contact_plane_value_->function;
  if(plane_data.n_out()!=4||plane_value.n_out()!=2)
    throw std::runtime_error("Contact-plane helper output count mismatch");
  if(plane_data.size1_out(0)!=2||plane_data.size2_out(0)!=1||plane_data.size1_out(1)!=2||plane_data.size2_out(1)!=2*g::NZ
     ||plane_data.size1_out(2)!=18||plane_data.size2_out(2)!=1||plane_data.size1_out(3)!=18||plane_data.size2_out(3)!=2*g::NZ)
    throw std::runtime_error("Contact-plane helper output dimensions mismatch");
  if(plane_data.sparsity_out(0)!=plane_value.sparsity_out(0)||plane_data.sparsity_out(2)!=plane_value.sparsity_out(1))
    throw std::runtime_error("Contact-plane data/value correction sparsity differs");
  for(auto& f:layouts_)if(f.data->function.size1_in(0)==2*g::NZ){
#ifdef JUGGLING_KINEMATIC_DYNAMIC_PREFIX
    // The finite-arm helper contains mass-dependent dynamics corrections that
    // do not exist in these factors. The constructor enforces offset==0.
    if(kinematic_stage(f.column/g::NZ))continue;
#endif
    const auto residual=f.data->function.sparsity_out(0),jr=f.data->function.sparsity_out(1);
    const auto constraints=f.data->function.sparsity_out(2),jc=f.data->function.sparsity_out(3);
    std::array<int,18> rows;rows.fill(-1);
    for(int row=0;row<constraints.size1();++row)for(size_t i=0;i<rows.size();++i)
      if(g::constraint_ids[f.row+row]==contact_plane_constraint_ids[i])rows[i]=row;
    if(std::any_of(rows.begin(),rows.end(),[](int i){return i<0;}))
      throw std::runtime_error("Contact-plane correction requires all ball/spin/dynamics/gap rows");
    const auto dr=plane_data.sparsity_out(0),dc=plane_data.sparsity_out(2);
    for(int source=0;source<dr.nnz();++source){
      const int row=contact_plane_residual_ids[dr.row(source)];int dest=-1;
      for(int i=0;i<residual.nnz();++i)if(residual.row(i)==row){dest=i;break;}
      if(dest<0)throw std::runtime_error("Contact-plane residual extends factor output sparsity");
      f.plane_residual_dest.push_back({dest,source});f.plane_residual_rows.push_back({row,source});
    }
    for(int source=0;source<dc.nnz();++source){
      const int row=rows[dc.row(source)];int dest=-1;
      for(int i=0;i<constraints.nnz();++i)if(constraints.row(i)==row){dest=i;break;}
      if(dest<0)throw std::runtime_error("Contact-plane constraint extends factor output sparsity");
      f.plane_constraint_dest.push_back({dest,source});f.plane_constraint_rows.push_back({row,source});
    }
    const auto drj=plane_data.sparsity_out(1),dcj=plane_data.sparsity_out(3);
    for(int column=0;column<drj.size2();++column)for(int source=drj.colind(column);source<drj.colind(column+1);++source){
      const int row=contact_plane_residual_ids[drj.row(source)];int dest=-1;
      for(int i=jr.colind(column);i<jr.colind(column+1);++i)if(jr.row(i)==row){dest=i;break;}
      if(dest<0)throw std::runtime_error("Contact-plane correction extends residual Jacobian sparsity");
      f.plane_residual_jacobian_dest.push_back({dest,source});
    }
    for(int column=0;column<dcj.size2();++column)for(int source=dcj.colind(column);source<dcj.colind(column+1);++source){
      const int row=rows[dcj.row(source)];int dest=-1;
      for(int i=jc.colind(column);i<jc.colind(column+1);++i)if(jc.row(i)==row){dest=i;break;}
      if(dest<0)throw std::runtime_error("Contact-plane correction extends constraint Jacobian sparsity");
      f.plane_constraint_jacobian_dest.push_back({dest,source});
    }
  }
#endif
#if defined(JUGGLING_LEGACY_EDGE_PREDICTION) && JUGGLING_LEGACY_EDGE_PREDICTION
#ifndef JUGGLING_PARAMETRIC_FACTORS
#error Legacy edge prediction requires parametric factors
#endif
#if defined(JUGGLING_RAMP_PREDICTION) && JUGGLING_RAMP_PREDICTION
#error Legacy edge prediction cannot be combined with full-ramp factors
#endif
  const double edge_loss=JUGGLING_PREDICTION_EDGE_LOSS;
  if(!std::isfinite(edge_loss)||edge_loss<0.||edge_loss>1.||g::restitution<0.||g::restitution>1.5)
    throw std::runtime_error("Invalid legacy edge restitution parameters");
  edge_prediction_data_=std::make_unique<Evaluation>(d,"edge_prediction_data",c);
  edge_prediction_value_=std::make_unique<Evaluation>(d,"edge_prediction_value",c);
  for(auto* evaluator:{edge_prediction_data_.get(),edge_prediction_value_.get()}){
    const auto& fn=evaluator->function;
    const int sizes[]={2*g::NZ,g::NP,static_cast<int>(g::factor_parameters.size()),1};
    if(fn.n_in()!=4)throw std::runtime_error("Legacy edge prediction requires four explicit inputs");
    for(int i=0;i<4;++i)if(fn.size1_in(i)!=sizes[i]||fn.size2_in(i)!=1)
      throw std::runtime_error("Legacy edge prediction input dimensions mismatch");
    if(fn.size1_out(0)!=1||fn.size2_out(0)!=1||fn.nnz_out(0)!=1)
      throw std::runtime_error("Legacy edge prediction correction must be scalar");
  }
  const auto& edge_data=edge_prediction_data_->function;
  if(edge_data.n_out()!=2||edge_prediction_value_->function.n_out()!=1)
    throw std::runtime_error("Legacy edge prediction output count mismatch");
  const auto edge_jr=edge_data.sparsity_out(1);
  if(edge_jr.size1()!=1||edge_jr.size2()!=2*g::NZ)
    throw std::runtime_error("Legacy edge prediction Jacobian dimensions mismatch");
  for(auto& f:layouts_)if(f.data->function.size1_in(0)==2*g::NZ){
    const auto rs=f.data->function.sparsity_out(0),jr=f.data->function.sparsity_out(1);
    for(int i=0;i<rs.nnz();++i)if(rs.row(i)==52)f.edge_residual_dest=i;
    if(f.edge_residual_dest<0)throw std::runtime_error("Legacy edge prediction requires residual row 52");
    for(int column=0;column<edge_jr.size2();++column)
      for(int source=edge_jr.colind(column);source<edge_jr.colind(column+1);++source){
        int dest=-1;
        for(int i=jr.colind(column);i<jr.colind(column+1);++i)if(jr.row(i)==52){dest=i;break;}
        if(dest<0)throw std::runtime_error("Legacy edge correction extends existing row-52 Jacobian sparsity");
        f.edge_jacobian_dest.push_back({dest,source});
      }
  }
#endif
#if defined(JUGGLING_LEGACY_CONTACT_BIAS) && JUGGLING_LEGACY_CONTACT_BIAS
#ifndef JUGGLING_PARAMETRIC_FACTORS
#error Legacy contact bias requires parametric factors
#endif
#if defined(JUGGLING_RAMP_PREDICTION) && JUGGLING_RAMP_PREDICTION
#error Legacy contact bias cannot be combined with full-ramp factors
#endif
  contact_bias_data_=std::make_unique<Evaluation>(d,"contact_bias_prediction_data",c);
  contact_bias_value_=std::make_unique<Evaluation>(d,"contact_bias_prediction_value",c);
  for(auto* evaluator:{contact_bias_data_.get(),contact_bias_value_.get()}){
    const auto& fn=evaluator->function;
    const int sizes[]={2*g::NZ,g::NP,static_cast<int>(g::factor_parameters.size()),3};
    if(fn.n_in()!=4)throw std::runtime_error("Contact bias prediction requires four explicit inputs");
    for(int i=0;i<4;++i)if(fn.size1_in(i)!=sizes[i]||fn.size2_in(i)!=1)
      throw std::runtime_error("Contact bias prediction input dimensions mismatch");
    if(fn.size1_out(0)!=6||fn.size2_out(0)!=1||fn.nnz_out(0)!=6)
      throw std::runtime_error("Contact bias prediction correction must be a dense six-vector");
  }
  const auto& bias_data=contact_bias_data_->function;
  if(bias_data.n_out()!=2||contact_bias_value_->function.n_out()!=1)
    throw std::runtime_error("Contact bias prediction output count mismatch");
  const auto bias_jc=bias_data.sparsity_out(1);
  if(bias_jc.size1()!=6||bias_jc.size2()!=2*g::NZ)
    throw std::runtime_error("Contact bias prediction Jacobian dimensions mismatch");
  for(auto& f:layouts_)if(f.data->function.size1_in(0)==2*g::NZ){
    const auto cs=f.data->function.sparsity_out(2),jc=f.data->function.sparsity_out(3);
    for(int i=0;i<cs.nnz();++i)if(cs.row(i)<6)f.bias_constraint_dest[cs.row(i)]=i;
    if(std::any_of(f.bias_constraint_dest.begin(),f.bias_constraint_dest.end(),[](int i){return i<0;}))
      throw std::runtime_error("Contact bias requires ball constraint rows 0..5");
    for(int column=0;column<bias_jc.size2();++column)
      for(int source=bias_jc.colind(column);source<bias_jc.colind(column+1);++source){
        int dest=-1;
        for(int i=jc.colind(column);i<jc.colind(column+1);++i)if(jc.row(i)==bias_jc.row(source)){dest=i;break;}
        if(dest<0)throw std::runtime_error("Contact bias correction extends existing ball Jacobian sparsity");
        f.bias_jacobian_dest.push_back({dest,source});
      }
  }
#endif
#ifdef JUGGLING_TAIL_ENDPOINT_START_S
  configure_tail_endpoint_elasticity();
#endif
#ifdef JUGGLING_FIXED_MIDPOINT_POSTURE
  configure_midpoint_preconditioner();
#endif
#ifdef JUGGLING_RADIAL_CONTACT_J0_M_S
  configure_radial_contact_scaling();
#endif
#ifdef JUGGLING_BOUNDED_MULTISTART_MODE
  configure_multistart();
#endif
  build_pattern();
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
  configure_residual_lift();
#endif
  const char* threads=std::getenv("JUGGLING_EVAL_THREADS");evaluator_threads_=threads?std::stoi(threads):2;
  if(evaluator_threads_<1||evaluator_threads_>16)throw std::runtime_error("JUGGLING_EVAL_THREADS must be 1..16");
  if(evaluator_threads_>1){std::vector<EvaluationPool::Job> jobs;for(auto& f:layouts_)jobs.push_back({f.data,f.column,f.step});pool_=std::make_unique<EvaluationPool>(evaluator_threads_,jobs);}
#ifdef JUGGLING_SOLVER_WARMSTART_SEED_RADIUS
  const double seed_radius=JUGGLING_SOLVER_WARMSTART_SEED_RADIUS;
#ifdef JUGGLING_SOLVER_WARMSTART_SEED_LEAD
  const double seed_lead=JUGGLING_SOLVER_WARMSTART_SEED_LEAD;
#else
  constexpr double seed_lead=.3;
#endif
  if(!std::isfinite(seed_radius)||seed_radius<0.||(seed_radius>0.&&seed_radius<=g::paddle_radius))throw std::runtime_error("Virtual seed radius must be zero or exceed nominal paddle radius");
  if(!std::isfinite(seed_lead)||seed_lead<.3)throw std::runtime_error("Virtual seed lead must be at least 0.3 seconds");
#ifdef JUGGLING_SOLVER_WARMSTART_SEED_MAX_CONTACTS
  const double seed_max_contacts=JUGGLING_SOLVER_WARMSTART_SEED_MAX_CONTACTS;
  if(!std::isfinite(seed_max_contacts)||seed_max_contacts<1.||seed_max_contacts>4.||seed_max_contacts!=std::floor(seed_max_contacts))
    throw std::runtime_error("Virtual seed maximum contacts must be an integer in [1,4]");
#endif
  if(seed_radius>0.){const char* path=std::getenv("JUGGLING_SEED_TRACE");if(path&&*path){
    seed_trace_=std::make_unique<std::ofstream>(path,std::ios::out|std::ios::app);
    if(!*seed_trace_)throw std::runtime_error("Cannot open virtual-seed trace");seed_trace_->precision(17);
  }}
#endif
#ifdef JUGGLING_SOLVER_WARMSTART_ARM_SHIFT
  if(JUGGLING_SOLVER_WARMSTART_ARM_SHIFT!=0.&&JUGGLING_SOLVER_WARMSTART_ARM_SHIFT!=1.)
    throw std::runtime_error("Arm-shift warm start must be 0 or 1");
  if(JUGGLING_SOLVER_WARMSTART_ARM_SHIFT>0.){
#if !defined(JUGGLING_PARAMETRIC_FACTORS) || (defined(JUGGLING_RAMP_PREDICTION) && JUGGLING_RAMP_PREDICTION)
    throw std::runtime_error("Arm-shift warm start requires parametric legacy factors");
#else
    if(JUGGLING_SOLVER_WARMSTART_SEED_RADIUS<=0.||g::stage_offsets[1]-g::stage_offsets[0]!=int(g::seed_arm_lower.size()))
      throw std::runtime_error("Arm-shift warm start requires virtual seeds and a full first stage");
#endif
  }
#endif
}
#ifdef JUGGLING_FIXED_MIDPOINT_POSTURE
void Controller::configure_midpoint_preconditioner(){
#if !defined(JUGGLING_PARAMETRIC_FACTORS) || defined(JUGGLING_KINEMATIC_DYNAMIC_PREFIX) || (defined(JUGGLING_RAMP_PREDICTION) && JUGGLING_RAMP_PREDICTION)
  throw std::runtime_error("Fixed midpoint posture requires constant-preconditioned legacy dynamics");
#endif
#ifdef JUGGLING_SOLVER_WARMSTART_ARM_SHIFT
  if(JUGGLING_SOLVER_WARMSTART_ARM_SHIFT!=0.)throw std::runtime_error("Fixed midpoint posture does not support arm-shift repair");
#endif
#ifdef JUGGLING_SOLVER_WARMSTART_REPAIR
  if(JUGGLING_SOLVER_WARMSTART_REPAIR!=0.)throw std::runtime_error("Fixed midpoint posture does not support warmstart repair");
#endif
#ifdef JUGGLING_BOUNDED_MULTISTART_MODE
  if(JUGGLING_BOUNDED_MULTISTART_MODE!=0.)throw std::runtime_error("Fixed midpoint posture does not support multistart");
#endif
  const Joint reference=map(g::midpoint_preconditioner_posture);
  if(!reference.allFinite()||(reference.array()<map(g::joint_lower).array()).any()
     ||(reference.array()>map(g::joint_upper).array()).any())
    throw std::runtime_error("Invalid fixed midpoint posture");
  const Matrix nominal=robot.mass(reference),home=robot.mass(map(g::midpoint_preconditioner_home));
  if(!nominal.allFinite()||!home.allFinite()||!nominal.isApprox(nominal.transpose(),1e-12)
     ||!home.isApprox(home.transpose(),1e-12))throw std::runtime_error("Invalid fixed midpoint mass matrix");
  Eigen::LLT<Matrix> nominal_factor(nominal),home_factor(home);
  if(nominal_factor.info()!=Eigen::Success||home_factor.info()!=Eigen::Success)
    throw std::runtime_error("Fixed midpoint mass matrices must be positive definite");
  midpoint_transform_=nominal_factor.solve(home);
  if(!midpoint_transform_.allFinite()||!(nominal*midpoint_transform_).isApprox(home,1e-11))
    throw std::runtime_error("Failed fixed midpoint coordinate transform");
  int stages=0;
  for(auto& f:layouts_)if(f.data->function.size1_in(0)==2*g::NZ){
    const auto cs=f.data->function.sparsity_out(2),vs=f.value->function.sparsity_out(1),jc=f.data->function.sparsity_out(3);
    if(cs.size2()!=1||vs!=cs||f.row+cs.size1()>g::CONSTRAINTS)
      throw std::runtime_error("Fixed midpoint constraint layout mismatch");
    f.midpoint_constraint_rows.fill(-1);f.midpoint_constraint_dest.fill(-1);
    for(int row=0;row<cs.size1();++row){const int joint=g::constraint_ids[f.row+row]-g::midpoint_dynamics_begin;
      if(joint<0||joint>=7)continue;
      if(f.midpoint_constraint_rows[joint]>=0||cl_[f.row+row]!=0.||cu_[f.row+row]!=0.)
        throw std::runtime_error("Fixed midpoint transform requires unique zero-equality dynamics rows");
      f.midpoint_constraint_rows[joint]=row;
    }
    for(int slot=0;slot<cs.nnz();++slot)for(int joint=0;joint<7;++joint)
      if(cs.row(slot)==f.midpoint_constraint_rows[joint])f.midpoint_constraint_dest[joint]=slot;
    if(std::any_of(f.midpoint_constraint_rows.begin(),f.midpoint_constraint_rows.end(),[](int v){return v<0;})
       ||std::any_of(f.midpoint_constraint_dest.begin(),f.midpoint_constraint_dest.end(),[](int v){return v<0;}))
      throw std::runtime_error("Fixed midpoint dynamics rows are missing from factor output");
    for(int column=0;column<jc.size2();++column){std::array<int,7> slots;slots.fill(-1);int count=0;
      for(int slot=jc.colind(column);slot<jc.colind(column+1);++slot)for(int joint=0;joint<7;++joint)
        if(jc.row(slot)==f.midpoint_constraint_rows[joint]){slots[joint]=slot;++count;}
      if(count==0)continue;
      if(count!=7)throw std::runtime_error("Fixed midpoint transform would extend Jacobian sparsity");
      f.midpoint_jacobian_dest.push_back(slots);
    }
    if(f.midpoint_jacobian_dest.empty())throw std::runtime_error("Empty fixed midpoint dynamics Jacobian");
    ++stages;
  }
  if(stages!=g::N)throw std::runtime_error("Fixed midpoint transform must cover every shooting interval");
}
void Controller::transform_midpoint_data(Layout& f){
  if(f.midpoint_jacobian_dest.empty())return;
  auto& out=f.data->output;Joint original;
  for(int joint=0;joint<7;++joint)original[joint]=out[2][f.midpoint_constraint_dest[joint]];
  Joint transformed=midpoint_transform_*original;
  for(int joint=0;joint<7;++joint)out[2][f.midpoint_constraint_dest[joint]]=transformed[joint];
  for(const auto& slots:f.midpoint_jacobian_dest){
    for(int joint=0;joint<7;++joint)original[joint]=out[3][slots[joint]];
    transformed.noalias()=midpoint_transform_*original;
    for(int joint=0;joint<7;++joint)out[3][slots[joint]]=transformed[joint];
  }
}
void Controller::transform_midpoint_value(const Layout& f,Vector& constraints) const {
  if(f.midpoint_jacobian_dest.empty())return;
  Joint original;for(int joint=0;joint<7;++joint)original[joint]=constraints[f.midpoint_constraint_rows[joint]];
  const Joint transformed=midpoint_transform_*original;
  for(int joint=0;joint<7;++joint)constraints[f.midpoint_constraint_rows[joint]]=transformed[joint];
}
#endif
#ifdef JUGGLING_TAIL_ENDPOINT_START_S
void Controller::configure_tail_endpoint_elasticity(){
  const double start=JUGGLING_TAIL_ENDPOINT_START_S,extra=JUGGLING_TAIL_ENDPOINT_EXTRA,weight=JUGGLING_TAIL_ENDPOINT_WEIGHT;
  if(!std::isfinite(start)||start<.30||start>=g::stage_times.back()||!std::isfinite(extra)||extra<0.||extra>.20
     ||!std::isfinite(weight)||weight<=0.)throw std::runtime_error("Invalid bounded tail endpoint elasticity");
  if(extra==0.)return; // Exact disabled profile: no row selection, cost or pattern changes.
  tail_endpoint_boundary_=std::max(5,int(std::lower_bound(g::stage_times.begin(),g::stage_times.end(),start)-g::stage_times.begin()));
  if(tail_endpoint_boundary_>=g::N)throw std::runtime_error("Tail endpoint elasticity has no unprotected endpoint");
  std::map<std::pair<int,int>,size_t> unique;
  for(int k=0;k<g::N;++k){
    const int layout=k+1;const auto& f=layouts_[layout];
    const int rows=f.data->function.size1_out(2);
    if(f.row!=g::stage_offsets[k]||rows!=g::stage_offsets[k+1]-f.row)
      throw std::runtime_error("Tail endpoint stage layout mismatch");
    for(int local=0;local<rows;++local){
      const int row=f.row+local,id=g::constraint_ids[row]-g::endpoint_acceleration_begin;
      if(id<0||id>=14)continue;
      const int node=k+id/7,joint=id%7;
      if(node<=tail_endpoint_boundary_){tail_endpoint_protected_rows_.push_back(row);continue;}
      const double lo=cl_[row],hi=cu_[row];
      if(!std::isfinite(lo)||!std::isfinite(hi)||lo>=hi||lo-extra<-.95-1e-12||hi+extra>.95+1e-12)
        throw std::runtime_error("Tail endpoint original/extra bounds exceed the finite 0.95 research cap");
      const auto key=std::make_pair(node,joint);auto found=unique.find(key);
      if(found==unique.end()){
        TailEndpointPenalty term;term.layout=layout;term.local_row=local;term.row=row;term.node=node;term.joint=joint;
        term.original_lower=lo;term.original_upper=hi;
        term.time_weight=.5*(g::stage_steps[node-1]+(node<g::N?g::stage_steps[node]:0.));
        term.gain_squared=g::objective_scale*weight*weight*(term.time_weight/.03)/(.05*.05);
        if(!std::isfinite(term.gain_squared)||term.gain_squared<=0.)throw std::runtime_error("Invalid tail endpoint penalty scale");
        term.duplicate_rows.push_back(row);unique.emplace(key,tail_endpoint_penalties_.size());tail_endpoint_penalties_.push_back(std::move(term));
      }else{
        auto& term=tail_endpoint_penalties_[found->second];
        if(std::abs(lo-term.original_lower)>1e-12||std::abs(hi-term.original_upper)>1e-12)
          throw std::runtime_error("Duplicate endpoint occurrences have inconsistent original bounds");
        term.duplicate_rows.push_back(row);
      }
    }
  }
  if(tail_endpoint_penalties_.empty())throw std::runtime_error("Tail endpoint elasticity selected no retained endpoint rows");
  // Validate and capture ALL original bounds before touching the private copies.
  // Never rescale by the optionally enlarged execution acceleration array.
  for(const auto& term:tail_endpoint_penalties_)for(int row:term.duplicate_rows){
    cl_[row]=std::max(-.95,term.original_lower-extra);cu_[row]=std::min(.95,term.original_upper+extra);
  }
  const char* path=std::getenv("JUGGLING_TAIL_ENDPOINT_TRACE");
  if(path&&*path){
    tail_endpoint_trace_=std::make_unique<std::ofstream>(path,std::ios::out|std::ios::app);
    if(!*tail_endpoint_trace_)throw std::runtime_error("Cannot open tail endpoint elasticity trace");tail_endpoint_trace_->precision(17);
    auto& out=*tail_endpoint_trace_;
    out<<"{\"event\":\"tail_endpoint_policy\",\"boundary_node\":"<<tail_endpoint_boundary_
       <<",\"boundary_time_s\":"<<g::stage_times[tail_endpoint_boundary_]<<",\"requested_start_s\":"<<start
       <<",\"extra_fraction\":"<<extra<<",\"weight\":"<<weight<<",\"fixed_excess_scale\":0.05,\"research_hard_ceiling\":0.95,\"protected_endpoint_rows\":[";
    for(size_t i=0;i<tail_endpoint_protected_rows_.size();++i){if(i)out<<',';out<<tail_endpoint_protected_rows_[i];}
    out<<"],\"terms\":[";
    for(size_t i=0;i<tail_endpoint_penalties_.size();++i){const auto& term=tail_endpoint_penalties_[i];if(i)out<<',';
      out<<"{\"node\":"<<term.node<<",\"joint\":"<<term.joint<<",\"row\":"<<term.row
         <<",\"original_lower\":"<<term.original_lower<<",\"original_upper\":"<<term.original_upper
         <<",\"relaxed_lower\":"<<cl_[term.row]<<",\"relaxed_upper\":"<<cu_[term.row]
         <<",\"dt_weight_s\":"<<term.time_weight<<",\"gain_squared\":"<<term.gain_squared<<",\"duplicate_rows\":[";
      for(size_t j=0;j<term.duplicate_rows.size();++j){if(j)out<<',';out<<term.duplicate_rows[j];}out<<"]}";
    }
    out<<"]}\n";out.flush();
  }
}
double Controller::tail_endpoint_regularization(const Vector& constraints,bool assemble){
  double cost=0.;
  for(const auto& term:tail_endpoint_penalties_){
    const double a=constraints[term.row];
    const double excess=a>term.original_upper?a-term.original_upper:(a<term.original_lower?a-term.original_lower:0.);
    if(excess==0.)continue; // Squared hinge: zero gradient/GN contribution at a bound.
    cost+=.5*term.gain_squared*excess*excess;
    if(assemble){const auto& jc=layouts_[term.layout].data->output[3];
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
      Matrix jacobian(1,term.jacobian.size());const double gain=std::sqrt(term.gain_squared);
      for(size_t k=0;k<term.jacobian.size();++k)jacobian(0,k)=gain*jc[term.jacobian[k][1]];
      lift_->set_dense(term.lift_block,Vector::Constant(1,gain*excess),jacobian);
      if(assemble_normal_){
#endif
      for(const auto& entry:term.jacobian)gradient_[entry[0]]+=term.gain_squared*excess*jc[entry[1]];
      for(const auto& product:term.hessian)H_.values[product.destination]+=term.gain_squared*jc[product.left]*jc[product.right];
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
      }
#endif
    }
  }
  return cost;
}
void Controller::trace_tail_endpoint_elasticity(const Vector& constraints,const StepInfo& info){
  if(!tail_endpoint_trace_||tail_endpoint_trace_records_>=1024)return;
  const size_t step=tail_endpoint_trace_step_++;
  auto violation=[](double a,double l,double u){return std::max({0.,l-a,a-u});};
  double original=0.,hard=0.,prefix=0.,integrated=0.;int near_cap=0;std::vector<double> debts;
  for(const auto& term:tail_endpoint_penalties_){
    double debt=0.;
    for(int row:term.duplicate_rows){debt=std::max(debt,violation(constraints[row],term.original_lower,term.original_upper));
      hard=std::max(hard,violation(constraints[row],cl_[row],cu_[row]));}
    original=std::max(original,debt);integrated+=term.time_weight*debt;debts.push_back(debt);
    near_cap+=debt>=.9*JUGGLING_TAIL_ENDPOINT_EXTRA;
  }
  for(int row:tail_endpoint_protected_rows_)prefix=std::max(prefix,violation(constraints[row],cl_[row],cu_[row]));
  if(step>=8&&step%25!=0&&original<=tail_endpoint_logged_debt_+.005)return;
  tail_endpoint_logged_debt_=std::max(tail_endpoint_logged_debt_,original);++tail_endpoint_trace_records_;
  std::sort(debts.begin(),debts.end());
  auto& out=*tail_endpoint_trace_;
  out<<"{\"event\":\"tail_endpoint_debt\",\"step\":"<<step<<",\"controller_time_s\":"<<tail_endpoint_elapsed_
     <<",\"original_max_excess\":"<<original<<",\"original_p50_excess\":"<<debts[debts.size()/2]
     <<",\"original_p95_excess\":"<<debts[size_t(.95*(debts.size()-1))]
     <<",\"integrated_original_excess_s\":"<<integrated<<",\"relaxed_tail_hard_violation\":"<<hard
     <<",\"protected_endpoint_violation\":"<<prefix<<",\"all_hard_constraint_violation\":"<<info.violation
     <<",\"bound_violation\":"<<info.bound_violation<<",\"penalty_cost\":"<<tail_endpoint_regularization(constraints,false)
     <<",\"unique_endpoints_near_extra_cap\":"<<near_cap<<"}\n";out.flush();
}
#endif
#ifdef JUGGLING_EXPERIMENTAL_GROUP_RHO
void Controller::install_group_rho(int sqp){
  if(contact_rho_multiplier_==1.)return; // Default parity: no private call/refactor.
  const bool record=group_rho_trace_&&(group_rho_updates_<8||group_rho_updates_%100==0);
  JugglingGroupRhoReport report{};
  const int status=juggling_osqp_install_group_rho(solver_,m_+n_,group_rho_selected_.data(),contact_rho_multiplier_,
      group_rho_types_.data(),group_rho_values_.data(),record?&report:nullptr);
  if(status)throw std::runtime_error("OSQP group-rho synchronization failed ("+std::to_string(status)+")");
  if(record){*group_rho_trace_<<"{\"update\":"<<group_rho_updates_<<",\"time\":"<<seed_controller_time_
      <<",\"sqp\":"<<sqp<<",\"multiplier\":"<<contact_rho_multiplier_<<",\"base_rho\":"<<report.base_rho
      <<",\"selected_inequalities\":"<<report.selected_inequalities<<",\"selected_equalities\":"<<report.selected_equalities
      <<",\"selected_free\":"<<report.selected_free<<",\"selected_rho_min\":"<<report.selected_rho_min
      <<",\"selected_rho_max\":"<<report.selected_rho_max<<",\"effective_rho_min\":"<<report.effective_rho_min
      <<",\"effective_rho_max\":"<<report.effective_rho_max<<"}\n";}
  ++group_rho_updates_;
}
#endif
#ifndef JUGGLING_NULLSPACE_POSITION_WEIGHT
#define JUGGLING_NULLSPACE_POSITION_WEIGHT 0.
#define JUGGLING_NULLSPACE_VELOCITY_WEIGHT 0.
#endif
void Controller::update_posture_projector(){
  if(JUGGLING_NULLSPACE_POSITION_WEIGHT==0. && JUGGLING_NULLSPACE_VELOCITY_WEIGHT==0.)return;
  auto f=robot.contact_fk(map(g::home)+reference_posture_delta_);
  Eigen::Vector3d n=f[1];Eigen::Matrix3d cross;
  cross<<0.,-n[2],n[1],n[2],0.,-n[0],-n[1],n[0],0.;
  Matrix task(6,7);task.topRows(3)=f[2];task.bottomRows(3)=-cross*f[3];
  Eigen::JacobiSVD<Matrix> svd(task,Eigen::ComputeFullV);
  posture_projector_.setIdentity();
  for(int k=0;k<svd.singularValues().size();++k)if(svd.singularValues()[k]>1e-8)
    posture_projector_-=svd.matrixV().col(k)*svd.matrixV().col(k).transpose();
}
double Controller::posture_regularization(const Vector& at,bool assemble){
  if(JUGGLING_NULLSPACE_POSITION_WEIGHT==0. && JUGGLING_NULLSPACE_VELOCITY_WEIGHT==0.)return 0.;
  double cost=0.;size_t dest=0;
  // Frozen at the session reference: a local task-nullspace preference, not an
  // exact nullspace guarantee far from that posture. QP and merit costs match.
  for(int k=0;k<g::N;++k)for(int block=0;block<2;++block){
    int offset=k*g::NZ+(block?13:6);double scale=block?2.:1.;
    double gain=block?JUGGLING_NULLSPACE_VELOCITY_WEIGHT:JUGGLING_NULLSPACE_POSITION_WEIGHT;
    double weight=g::objective_scale*(g::quadrature?g::stage_steps[k]/.03:1.)*gain*gain;
    Joint delta=scale*at.segment<7>(offset);if(!block)delta-=reference_posture_delta_;
    Joint projected=posture_projector_*delta;cost+=.5*weight*projected.squaredNorm();
    if(assemble){
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
      lift_->set_dense(posture_lift_blocks_[2*k+block],std::sqrt(weight)*projected,
          std::sqrt(weight)*scale*posture_projector_);
      if(assemble_normal_){
#endif
      gradient_.segment<7>(offset)+=weight*scale*posture_projector_*projected;
      auto h=(posture_projector_.transpose()*posture_projector_).eval();
      for(int c=0;c<7;++c)for(int r=0;r<=c;++r)
        H_.values[posture_hessian_dest_[dest++]]+=weight*scale*scale*h(r,c);
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
      }
#endif
    }
  }
  return cost;
}
double Controller::terminal_regularization(const Vector& at,bool assemble){
#ifdef JUGGLING_SOLVER_TERMINAL_POSITION_WEIGHT
  const double gains[]={JUGGLING_SOLVER_TERMINAL_POSITION_WEIGHT,JUGGLING_SOLVER_TERMINAL_VELOCITY_WEIGHT};
  double cost=0.;
  for(int block=0;block<2;++block){
    const double gain=gains[block];
    if(!std::isfinite(gain)||gain<0.)throw std::runtime_error("Invalid terminal arm regularization");
    if(gain==0.)continue;
    const double weight=g::objective_scale*gain*gain;
    for(int j=0;j<7;++j){
      const int index=g::N*g::NZ+(block?13:6)+j;
      const double scale=block?g::scale[13+j]/g::velocity_limit[j]:g::scale[6+j];
      const double residual=scale*at[index]-(block?0.:reference_posture_delta_[j]);
      cost+=.5*weight*residual*residual;
      if(assemble){
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
        lift_->set_dense(terminal_lift_blocks_[block*7+j],Vector::Constant(1,std::sqrt(weight)*residual),Matrix::Constant(1,1,std::sqrt(weight)*scale));
        if(assemble_normal_){
#endif
        gradient_[index]+=weight*scale*residual;H_.values[diagonal_[index]]+=weight*scale*scale;
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
        }
#endif
      }
    }
  }
  return cost;
#else
  return 0.;
#endif
}
#ifdef JUGGLING_FIRST_EVENT_CENTER_WEIGHT
void Controller::configure_first_event_center(){
  const double weight=JUGGLING_FIRST_EVENT_CENTER_WEIGHT,lead=JUGGLING_FIRST_EVENT_CENTER_ACQUIRE_LEAD_S;
  if(!std::isfinite(weight)||weight<0.||!std::isfinite(lead)||lead<.05||lead>.5)
    throw std::runtime_error("Invalid first-event centering weight/acquisition lead");
  if(weight==0.)return; // No geometry, trace, Hessian pattern or lifecycle work.
#ifndef JUGGLING_PARAMETRIC_FACTORS
  throw std::runtime_error("First-event centering requires parametric legacy factors");
#else
#if defined(JUGGLING_RAMP_PREDICTION) && JUGGLING_RAMP_PREDICTION
  throw std::runtime_error("First-event centering requires legacy arm interpolation");
#endif
  const Eigen::Map<const Eigen::Matrix3d> normal_rotation(g::factor_parameters.data()+26);
  Eigen::Vector3d grip=normal_rotation.transpose()*Eigen::Vector3d(-1.,0.,0.);grip.z()=0.;
  if(!grip.allFinite()||grip.norm()<=1e-12)throw std::runtime_error("Invalid first-event preferred contact direction");
  first_event_center_.preferred_local=g::factor_parameters[5]*grip.normalized();
  const char* path=std::getenv("JUGGLING_FIRST_EVENT_CENTER_TRACE");
  if(path&&*path){first_event_center_trace_=std::make_unique<std::ofstream>(path,std::ios::out|std::ios::app);
    if(!*first_event_center_trace_)throw std::runtime_error("Cannot open first-event centering trace");
    first_event_center_trace_->precision(17);}
#endif
}
void Controller::trace_first_event_center(const char* action,const char* reason,bool force){
  if(!first_event_center_trace_||first_event_center_trace_records_>=8192)return;
  if(!force&&first_event_center_updates_>=8&&first_event_center_updates_%25!=0)return;
  ++first_event_center_trace_records_;const auto& event=first_event_center_;auto& out=*first_event_center_trace_;
  out<<"{\"event\":\"first_event_center\",\"action\":\""<<action<<"\",\"reason\":\""<<reason
     <<"\",\"update\":"<<first_event_center_updates_<<",\"generation\":"<<event.generation
     <<",\"controller_time_s\":"<<event.clock<<",\"active\":"<<(event.active?"true":"false")
     <<",\"absolute_event_time_s\":"<<event.event_time<<",\"relative_event_time_s\":"<<event.lead
     <<",\"stage\":"<<event.stage<<",\"interval_tau_s\":"<<event.tau<<",\"seed_impulse\":"<<event.seed_impulse
     <<",\"has_seed_match\":"<<(event.has_seed_match?"true":"false")<<",\"raw_matched_absolute_event_time_s\":"<<event.matched_event_time
     <<",\"applied_time_refresh_s\":"<<event.applied_refresh
     <<",\"weight\":"<<JUGGLING_FIRST_EVENT_CENTER_WEIGHT<<",\"acquire_lead_s\":"<<JUGGLING_FIRST_EVENT_CENTER_ACQUIRE_LEAD_S
     <<",\"target\":["<<event.target[0]<<','<<event.target[1]<<','<<event.target[2]
     <<"],\"preferred_local\":["<<event.preferred_local[0]<<','<<event.preferred_local[1]<<','<<event.preferred_local[2]<<"]}\n";out.flush();
}
void Controller::freeze_first_event_center(const State& state,double elapsed){
  if(JUGGLING_FIRST_EVENT_CENTER_WEIGHT==0.)return;
  auto& event=first_event_center_;event.clock+=elapsed;++first_event_center_updates_;
  const int old_stage=event.stage;const double old_impulse=event.seed_impulse;
  event.matched_event_time=0.;event.applied_refresh=0.;event.has_seed_match=false;
  // The observer is the only source of rebound evidence. An upward innovation
  // relative to its preceding nominal flight retires the pending event; no
  // simulator hit, true state, or optimized impulse can certify a rebound.
  const bool observed_rebound=event.have_observation&&elapsed>0.&&state[5]>.15
    &&state[5]-free(event.previous_observation,elapsed,Joint::Zero())[5]>.5;
  event.lead=event.event_time-event.clock;
  if(event.active&&(observed_rebound||event.lead<-.06)){
    event.active=false;trace_first_event_center("retire",observed_rebound?"observed_upward_velocity_innovation":"event_expired",true);
  }
  // z_ is the prior accepted plan after ordinary time/arm shifting and the
  // measured-node replacement, BEFORE ball regeneration or any merit test.
  // Select the first credible event, or match the retained event locally.
  double selected_lead=0.,selected_impulse=0.,closest=.060000000001;bool found=false;
  for(int k=0;k<g::N;++k){
    const Row y=z_.segment<g::NZ>(k*g::NZ).cwiseProduct(map(g::scale))+map(g::offset);
    const Row yn=z_.segment<g::NZ>((k+1)*g::NZ).cwiseProduct(map(g::scale))+map(g::offset);
    const double j=y[30],fraction=y[31],h=g::stage_steps[k],lead=g::stage_times[k]+fraction*h;
    if(!std::isfinite(j)||j<(event.active?2.:.25)||!std::isfinite(fraction)||fraction<0.||fraction>1.||lead<0.||lead>=g::stage_times.back())continue;
    if(event.active&&(lead<=0.||std::abs(lead-event.lead)>closest))continue;
    const Joint acc=(yn.segment<7>(13)-y.segment<7>(13))/h;
    const State before=free(y.head<g::NX>(),fraction*h,acc);
    if(!before.allFinite()||before[5]>=-.05)continue;
    const State observed_flight=free(state,lead,Joint::Zero());
    const auto geometry=prediction_geometry(before);
    const bool credible=j>=2.&&(event.active||lead>JUGGLING_FIRST_EVENT_CENTER_ACQUIRE_LEAD_S)
      &&observed_flight[5]<-.05&&std::isfinite(geometry.gap)&&std::isfinite(geometry.speed)
      &&std::abs(geometry.gap)<=.03&&geometry.speed<-.05;
    if(!credible){
      if(event.active)continue;
      // The first material descending impulse is a barrier. Free flight from
      // the observation cannot legitimately target a LATER recovery impact
      // after skipping this weak, too-close or otherwise unsuitable contact.
      event.lead=lead;event.stage=k;event.tau=fraction*h;event.seed_impulse=j;
      event.matched_event_time=event.clock+lead;event.target=observed_flight.head<3>();
      trace_first_event_center("skip","skipped_first_contact",old_stage!=k||(old_impulse>=2.)!=(j>=2.));break;
    }
    selected_lead=lead;selected_impulse=j;found=true;
    if(!event.active)break;closest=std::abs(lead-event.lead);
  }
  const bool acquired=!event.active&&found;
  if(found){event.matched_event_time=event.clock+selected_lead;event.seed_impulse=selected_impulse;event.has_seed_match=true;
    if(acquired){event.event_time=event.matched_event_time;event.active=true;++event.generation;}
    else{
      // A credible same-event match may correct its time, but cannot keep
      // deferring interception by the full elapsed interval on every update.
      const double max_refresh=std::min(.005,.25*elapsed);
      event.applied_refresh=std::clamp(event.matched_event_time-event.event_time,-max_refresh,max_refresh);
      event.event_time+=event.applied_refresh;}
    event.lead=event.event_time-event.clock;}
  else if(event.active)event.seed_impulse=0.; // Missing/weak impulse does NOT disable or retime the event.
  if(event.active){
    // A short overdue grace interval retains identity until observation/expiry.
    // Its target and q interpolation are clamped to NOW, hence only measured
    // node-zero q contributes; never extrapolate the arm behind the horizon.
    const double prediction_time=std::max(0.,event.lead);
    event.stage=std::min(source(prediction_time),g::N-1);
    event.tau=std::clamp(prediction_time-g::stage_times[event.stage],0.,g::stage_steps[event.stage]);
    event.target=free(state,prediction_time,Joint::Zero()).head<3>();
    trace_first_event_center(acquired?"acquire":"freeze",acquired?"first_descending_seed_event":(found?"matched_seed_event":"retained_without_strong_seed"),
      acquired||event.stage!=old_stage||(old_impulse>=2.)!=(event.seed_impulse>=2.));
  }
  event.previous_observation=state;event.have_observation=true;
}
double Controller::first_event_center_cost(const Vector& at,bool assemble){
  const auto& event=first_event_center_;if(JUGGLING_FIRST_EVENT_CENTER_WEIGHT==0.||!event.active)return 0.;
  const int k=event.stage;const double h=g::stage_steps[k],tau=event.tau;
  const double next_coefficient=.5*tau*tau/h,current_coefficient=tau-next_coefficient;
  // Fixed numeric h and tau define this auxiliary interception objective.
  // Neither the impulse nor optimized contact fraction can weaken its cost.
  Joint contact_q;std::array<int,21> columns;std::array<double,21> chain;
  for(int j=0;j<7;++j){columns[j]=k*g::NZ+6+j;columns[7+j]=k*g::NZ+13+j;columns[14+j]=(k+1)*g::NZ+13+j;
    chain[j]=g::scale[6+j];chain[7+j]=current_coefficient*g::scale[13+j];chain[14+j]=next_coefficient*g::scale[13+j];
    contact_q[j]=g::offset[6+j]+chain[j]*at[columns[j]]
      +current_coefficient*g::offset[13+j]+chain[7+j]*at[columns[7+j]]
      +next_coefficient*g::offset[13+j]+chain[14+j]*at[columns[14+j]];}
  // A normal-only plane translation cancels in this tangent projection, so
  // use the shared unshifted FK and keep physical/model geometry unchanged.
  const auto f=robot.contact_fk(contact_q);const Eigen::Vector3d n=f[1].col(0);
  const Eigen::Vector3d u=event.target-f[0].col(0)+map(g::origin),preferred=f[4]*event.preferred_local;
  const Eigen::Matrix3d projector=Eigen::Matrix3d::Identity()-n*n.transpose();
  const double scale=std::sqrt(g::objective_scale)*JUGGLING_FIRST_EVENT_CENTER_WEIGHT/g::paddle_radius;
  const Eigen::Vector3d residual=scale*(projector*u-preferred);
  if(assemble){
    const Eigen::Matrix<double,3,7> jq=scale*(-projector*f[2]
      +((n.dot(u))*Eigen::Matrix3d::Identity()+n*u.transpose())*skew(n)*f[3]+skew(preferred)*f[3]);
    Eigen::Matrix<double,3,21> jacobian;for(int c=0;c<21;++c)jacobian.col(c)=chain[c]*jq.col(c%7);
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
    lift_->set_dense(center_lift_blocks_[k],residual,jacobian);
    if(assemble_normal_){
#endif
    const Eigen::Matrix<double,21,1> gradient=jacobian.transpose()*residual;
    const Eigen::Matrix<double,21,21> hessian=jacobian.transpose()*jacobian;
    for(int c=0;c<21;++c)gradient_[columns[c]]+=gradient[c];
    size_t destination=0;for(int c=0;c<21;++c)for(int r=0;r<=c;++r)
      H_.values[first_event_center_hessian_dest_[k][destination++]]+=hessian(r,c);
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
    }
#endif
  }
  return .5*residual.squaredNorm();
}
#endif
void Controller::freeze_landing_guidance(const State& state){
#if defined(JUGGLING_SOLVER_LANDING_WEIGHT) && defined(JUGGLING_PARAMETRIC_FACTORS)
  if(JUGGLING_SOLVER_LANDING_WEIGHT==0.)return;
  landing_node_weights_.assign(g::N+1,0.);
#ifdef JUGGLING_SOLVER_LANDING_WINDOW
  const double window=JUGGLING_SOLVER_LANDING_WINDOW;
#else
  constexpr double window=.12;
#endif
#ifdef JUGGLING_BALL_DRAG
  const double drag=JUGGLING_BALL_DRAG;
#else
  constexpr double drag=0.;
#endif
  const Eigen::Vector3d gravity=Eigen::Map<const Eigen::Vector3d>(ball_gravity.data());
  if(!gravity.allFinite()||gravity.z()>=0.||!std::isfinite(drag)||drag<0.)return;
  // Only the observed initial ball state defines the free-flight intercept.
  // Freeze it before shift repair and every inner SQP/merit evaluation: the
  // optimizer cannot move this target by changing its ball or impact variables.
  auto flight=[&](double t){
    const double a=drag*t;
    double velocity_integral,position_integral;
    if(std::abs(a)<1e-4){
      velocity_integral=t*(1.-a/2.+a*a/6.-a*a*a/24.+a*a*a*a/120.);
      position_integral=t*t*(.5-a/6.+a*a/24.-a*a*a/120.+a*a*a*a/720.);
    }else{
      velocity_integral=-std::expm1(-a)/drag;
      position_integral=(t-velocity_integral)/drag;
    }
    Eigen::Matrix<double,6,1> out;
    out.head<3>()=state.head<3>()+velocity_integral*state.segment<3>(3)+position_integral*gravity;
    out.tail<3>()=std::exp(-a)*state.segment<3>(3)+velocity_integral*gravity;
    return out;
  };
  double plane=reference_offset_.z()+g::radius;
#if defined(JUGGLING_RAMP_PREDICTION) && JUGGLING_RAMP_PREDICTION
  plane+=JUGGLING_PREDICTION_PLANE_OFFSET_M;
#elif defined(JUGGLING_LEGACY_CONTACT_PLANE) && JUGGLING_LEGACY_CONTACT_PLANE
  if(JUGGLING_PREDICTION_PLANE_OFFSET_M!=0.)plane+=JUGGLING_PREDICTION_PLANE_OFFSET_M;
#endif
  double lo=0.,hi=g::stage_times.back()+window;
  if(state[5]>0.)lo=drag>0.?std::log1p(-drag*state[5]/gravity.z())/drag:-state[5]/gravity.z();
  // An initially rising ball below the plane can still have a valid future
  // descending crossing, so bracket from its apex instead of rejecting b0.z.
  if(!std::isfinite(lo)||lo>hi||flight(lo)[2]<plane||flight(hi)[2]>plane)return;
  for(int i=0;i<48;++i){double mid=.5*(lo+hi);if(flight(mid)[2]>plane)lo=mid;else hi=mid;}
  const double landing_time=.5*(lo+hi);const auto landing=flight(landing_time);
  if(!landing.allFinite()||landing[5]>=0.)return;
  landing_target_=landing.head<2>();
  // Stable runtime-factor ABI: scalar 5 is the grip offset; entries 26..34
  // are the column-major normal correction. Project CAD -X into that plane,
  // exactly as the contact-center residual in the generated factors does.
  const Eigen::Map<const Eigen::Matrix3d> normal_rotation(g::factor_parameters.data()+26);
  Eigen::Vector3d grip=normal_rotation.transpose()*Eigen::Vector3d(-1.,0.,0.);grip.z()=0.;
  if(!grip.allFinite()||grip.norm()<=1e-12)return;
  landing_preferred_local_=g::factor_parameters[5]*grip.normalized();
  double normalization=0.;
  for(int k=1;k<=g::N;++k){
    const double relative=(g::stage_times[k]-landing_time)/window;
    const double kernel=std::max(0.,1.-relative*relative);
    const double width=.5*(g::stage_steps[k-1]+(k<g::N?g::stage_steps[k]:0.));
    landing_node_weights_[k]=width*kernel*kernel;normalization+=landing_node_weights_[k];
  }
  if(!std::isfinite(normalization)||normalization<=1e-14){std::fill(landing_node_weights_.begin(),landing_node_weights_.end(),0.);return;}
  for(double& weight:landing_node_weights_)weight/=normalization;
#endif
}
double Controller::landing_guidance(const Vector& at,bool assemble){
#if defined(JUGGLING_SOLVER_LANDING_WEIGHT) && defined(JUGGLING_PARAMETRIC_FACTORS)
  const double weight=JUGGLING_SOLVER_LANDING_WEIGHT;
  if(weight==0.||landing_node_weights_.empty())return 0.;
  double cost=0.,normal_offset=g::radius;
#if defined(JUGGLING_RAMP_PREDICTION) && JUGGLING_RAMP_PREDICTION
  normal_offset+=JUGGLING_PREDICTION_PLANE_OFFSET_M;
#elif defined(JUGGLING_LEGACY_CONTACT_PLANE) && JUGGLING_LEGACY_CONTACT_PLANE
  if(JUGGLING_PREDICTION_PLANE_OFFSET_M!=0.)normal_offset+=JUGGLING_PREDICTION_PLANE_OFFSET_M;
#endif
  for(int k=1;k<=g::N;++k){
    const double gate=landing_node_weights_[k];if(gate==0.)continue;
    const int offset=k*g::NZ+6;Joint q;
    for(int j=0;j<7;++j)q[j]=g::offset[6+j]+g::scale[6+j]*at[offset+j];
    auto f=robot.contact_fk(q);
    const Eigen::Vector3d shift=f[4]*landing_preferred_local_+normal_offset*f[1].col(0);
    const Eigen::Vector3d point=f[0].col(0)-map(g::origin)+shift;
    const double scale=std::sqrt(g::objective_scale)*weight*std::sqrt(gate)/g::paddle_radius;
    const Eigen::Vector2d residual=scale*(point.head<2>()-landing_target_);
    cost+=.5*residual.squaredNorm();
    if(assemble){
      Eigen::Matrix<double,2,7> jacobian=(f[2]-skew(shift)*f[3]).topRows(2);
      for(int j=0;j<7;++j)jacobian.col(j)*=scale*g::scale[6+j];
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
      lift_->set_dense(landing_lift_blocks_[k-1],residual,jacobian);
      if(assemble_normal_){
#endif
      gradient_.segment<7>(offset)+=jacobian.transpose()*residual;
      const Eigen::Matrix<double,7,7> hessian=jacobian.transpose()*jacobian;
      size_t destination=(k-1)*28;
      for(int c=0;c<7;++c)for(int r=0;r<=c;++r)
        H_.values[landing_hessian_dest_[destination++]]+=hessian(r,c);
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
      }
#endif
    }
  }
  return cost;
#else
  return 0.;
#endif
}
Controller::~Controller(){if(solver_)osqp_cleanup(solver_);if(osqp_h_)OSQPCscMatrix_free(osqp_h_);if(osqp_a_)OSQPCscMatrix_free(osqp_a_);}
void Controller::set_reference(const Eigen::Vector3d& offset,const Joint& posture){
  if(!offset.allFinite()||!posture.allFinite()||offset.head<2>().norm()>.32||std::abs(offset[2])>.050001)
    throw std::runtime_error("Fixture reference outside local workspace");
  for(int j=0;j<7;++j)if(posture[j]<g::joint_lower[j]||posture[j]>g::joint_upper[j]||std::abs(posture[j]-g::home[j])>1.0)
    throw std::runtime_error("Fixture posture outside local joint envelope");
  reference_offset_=offset;reference_posture_delta_=posture-map(g::home);
  update_posture_projector();
}
void Controller::set_contact_bias(const Eigen::Vector3d& bias){
  if(!bias.allFinite()||bias.norm()>.3)throw std::runtime_error("Contact bias must be finite with norm at most 0.3 m/s per j/8");
#if defined(JUGGLING_LEGACY_CONTACT_BIAS) && JUGGLING_LEGACY_CONTACT_BIAS
  bias_per_j8_local_=bias;
  contact_bias_active_=(bias.array()!=0.).any();
#else
  if((bias.array()!=0.).any())throw std::runtime_error("Controller was built without contact bias prediction");
#endif
}
void Controller::reference_residual(Vector& r,double w) const {
  double s=std::sqrt(g::objective_scale)*w;
  double energy_shift=reference_offset_[2];
#ifdef JUGGLING_FACTOR_APEX
  energy_shift+=g::apex-JUGGLING_FACTOR_APEX;
#endif
  if(r.size()>=54){
    r[0]-=s*g::energy_weight*energy_shift;
    r.segment<2>(1)-=s*g::lateral_weight*reference_offset_.head<2>()/.2;
    Eigen::Vector3d pose_shift=s*g::pose_weight*reference_offset_.cwiseQuotient(Eigen::Vector3d(.3,.3,.15));
#ifdef JUGGLING_PADDLE_LATERAL_WEIGHT_MULTIPLIER
    pose_shift.head<2>()*=g::paddle_lateral_weight;
#endif
    r.segment<3>(5)-=pose_shift;
    r.segment<7>(37)-=s*g::posture_weight*reference_posture_delta_;
  }else if(r.size()==5){
    r[0]-=2*s*g::energy_weight*energy_shift;
    r.segment<2>(1)-=2*s*g::lateral_weight*reference_offset_.head<2>()/.2;
  }else if(r.size()!=0)throw std::runtime_error("Unknown residual layout for fixture reference");
}
void Controller::build_pattern(){
  using Key=std::pair<int,int>;std::set<Key> hp,ap;
  for(int i=0;i<n_;++i){hp.emplace(i,i);ap.emplace(i,m_+i);}
  for(const auto& f:layouts_){
    auto jr=f.data->function.sparsity_out(1),jc=f.data->function.sparsity_out(3);
    std::vector<std::vector<int>> rows(jr.size1());
    for(int c=0;c<jr.size2();++c)for(int j=jr.colind(c);j<jr.colind(c+1);++j)rows[jr.row(j)].push_back(c);
    for(auto& row:rows)for(int a:row)for(int b:row)if(a<=b)hp.emplace(f.column+b,f.column+a);
    for(int c=0;c<jc.size2();++c)for(int j=jc.colind(c);j<jc.colind(c+1);++j)ap.emplace(f.column+c,f.row+jc.row(j));
    if(f.restitution_cap_row>=0){
      if(jr.size1()<54)throw std::runtime_error("Restitution cap requires stage residual row 52");
      for(int c=0;c<jr.size2();++c)for(int j=jr.colind(c);j<jr.colind(c+1);++j)
        if(jr.row(j)==52)ap.emplace(f.column+c,f.restitution_cap_row);
    }
  }
  if(JUGGLING_NULLSPACE_POSITION_WEIGHT!=0. || JUGGLING_NULLSPACE_VELOCITY_WEIGHT!=0.)
    for(int k=0;k<g::N;++k)for(int block:{6,13})for(int c=0;c<7;++c)for(int r=0;r<=c;++r)
      hp.emplace(k*g::NZ+block+c,k*g::NZ+block+r);
#ifdef JUGGLING_SOLVER_LANDING_WEIGHT
  if(JUGGLING_SOLVER_LANDING_WEIGHT>0.)
    for(int k=1;k<=g::N;++k)for(int c=0;c<7;++c)for(int r=0;r<=c;++r)
      hp.emplace(k*g::NZ+6+c,k*g::NZ+6+r);
#endif
#ifdef JUGGLING_FIRST_EVENT_CENTER_WEIGHT
  if(JUGGLING_FIRST_EVENT_CENTER_WEIGHT>0.)
    for(int k=0;k<g::N;++k){std::array<int,21> columns;
      for(int j=0;j<7;++j){columns[j]=k*g::NZ+6+j;columns[7+j]=k*g::NZ+13+j;columns[14+j]=(k+1)*g::NZ+13+j;}
      for(int c=0;c<21;++c)for(int r=0;r<=c;++r)hp.emplace(columns[c],columns[r]);}
#endif
#ifdef JUGGLING_TAIL_ENDPOINT_START_S
  // Endpoint acceleration lives in JC, not JR. Add its pairwise support to
  // the Hessian even when the squared hinge is inactive at this NLP point.
  for(auto& term:tail_endpoint_penalties_){
    const auto& f=layouts_[term.layout];const auto jc=f.data->function.sparsity_out(3);
    for(int c=0;c<jc.size2();++c)for(int j=jc.colind(c);j<jc.colind(c+1);++j)
      if(jc.row(j)==term.local_row)term.jacobian.push_back({f.column+c,j});
    if(term.jacobian.empty())throw std::runtime_error("Tail endpoint row has no constraint Jacobian support");
    for(size_t a=0;a<term.jacobian.size();++a)for(size_t b=a;b<term.jacobian.size();++b)
      hp.emplace(term.jacobian[b][0],term.jacobian[a][0]);
  }
#endif
  auto make=[&](const std::set<Key>& pattern,Sparse& s){
    std::map<Key,int> lookup;s.columns.resize(n_+1);int column=0;
    for(auto key:pattern){while(column<=key.first)s.columns[column++]=s.rows.size();lookup[key]=s.rows.size();s.rows.push_back(key.second);}
    while(column<=n_)s.columns[column++]=s.rows.size();s.values.resize(s.rows.size());return lookup;
  };
  auto hm=make(hp,H_),am=make(ap,A_);diagonal_.resize(n_);
  for(int i=0;i<n_;++i)diagonal_[i]=hm.at({i,i});
#ifdef JUGGLING_FIRST_EVENT_CENTER_WEIGHT
  if(JUGGLING_FIRST_EVENT_CENTER_WEIGHT>0.){
    first_event_center_hessian_dest_.resize(g::N);
    for(int k=0;k<g::N;++k){std::array<int,21> columns;size_t destination=0;
      for(int j=0;j<7;++j){columns[j]=k*g::NZ+6+j;columns[7+j]=k*g::NZ+13+j;columns[14+j]=(k+1)*g::NZ+13+j;}
      for(int c=0;c<21;++c)for(int r=0;r<=c;++r)
        first_event_center_hessian_dest_[k][destination++]=hm.at({columns[c],columns[r]});}
  }
#endif
#ifdef JUGGLING_TAIL_ENDPOINT_START_S
  for(auto& term:tail_endpoint_penalties_)
    for(size_t a=0;a<term.jacobian.size();++a)for(size_t b=a;b<term.jacobian.size();++b)
      term.hessian.push_back({hm.at({term.jacobian[b][0],term.jacobian[a][0]}),term.jacobian[a][1],term.jacobian[b][1]});
#endif
  if(JUGGLING_NULLSPACE_POSITION_WEIGHT!=0. || JUGGLING_NULLSPACE_VELOCITY_WEIGHT!=0.){
    update_posture_projector();
    for(int k=0;k<g::N;++k)for(int block:{6,13})for(int c=0;c<7;++c)for(int r=0;r<=c;++r)
      posture_hessian_dest_.push_back(hm.at({k*g::NZ+block+c,k*g::NZ+block+r}));
  }
#ifdef JUGGLING_SOLVER_LANDING_WEIGHT
  if(JUGGLING_SOLVER_LANDING_WEIGHT>0.)
    for(int k=1;k<=g::N;++k)for(int c=0;c<7;++c)for(int r=0;r<=c;++r)
      landing_hessian_dest_.push_back(hm.at({k*g::NZ+6+c,k*g::NZ+6+r}));
#endif
  for(auto& f:layouts_){
    auto jr=f.data->function.sparsity_out(1),jc=f.data->function.sparsity_out(3);
    std::vector<std::vector<std::pair<int,int>>> rows(jr.size1());
    for(int c=0;c<jr.size2();++c)for(int j=jr.colind(c);j<jr.colind(c+1);++j)rows[jr.row(j)].emplace_back(c,j);
    for(auto& row:rows)for(auto a:row)for(auto b:row)if(a.first<=b.first)f.hessian.push_back({hm.at({f.column+b.first,f.column+a.first}),a.second,b.second});
    for(int c=0;c<jc.size2();++c)for(int j=jc.colind(c);j<jc.colind(c+1);++j)f.a_dest.push_back(am.at({f.column+c,f.row+jc.row(j)}));
    if(f.restitution_cap_row>=0)
      for(int c=0;c<jr.size2();++c)for(int j=jr.colind(c);j<jr.colind(c+1);++j)
        if(jr.row(j)==52)f.restitution_cap_a.push_back({am.at({f.column+c,f.restitution_cap_row}),j});
  }
  for(int c=0;c<n_;++c)A_.values[am.at({c,m_+c})]=1.;
}
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
void Controller::configure_residual_lift(){
  if(multistart_mode_!=0)throw std::runtime_error("Residual lift does not yet support bounded multistart");
#ifdef JUGGLING_SOLVER_CONTACT_RHO_MULTIPLIER
  if(JUGGLING_SOLVER_CONTACT_RHO_MULTIPLIER!=1.)throw std::runtime_error("Residual lift requires unit contact rho multiplier");
#endif
  lift_=std::make_unique<ResidualLift>(n_,m_+n_,g::proximal);
  for(auto& f:layouts_){auto jr=f.data->function.sparsity_out(1);std::vector<std::pair<int,int>> entries;
    for(int c=0;c<jr.size2();++c)for(int j=jr.colind(c);j<jr.colind(c+1);++j)entries.emplace_back(jr.row(j),f.column+c);
    f.lift_block=lift_->add_block(jr.size1(),entries);
  }
  auto columns=[](int offset,int count){std::vector<int> out(count);std::iota(out.begin(),out.end(),offset);return out;};
  if(JUGGLING_NULLSPACE_POSITION_WEIGHT!=0.||JUGGLING_NULLSPACE_VELOCITY_WEIGHT!=0.)
    for(int k=0;k<g::N;++k)for(int block:{6,13})posture_lift_blocks_.push_back(lift_->add_dense_block(7,columns(k*g::NZ+block,7)));
#ifdef JUGGLING_SOLVER_TERMINAL_POSITION_WEIGHT
  terminal_lift_blocks_.assign(14,-1);
  for(int block=0;block<2;++block)if((block?JUGGLING_SOLVER_TERMINAL_VELOCITY_WEIGHT:JUGGLING_SOLVER_TERMINAL_POSITION_WEIGHT)>0.)
    for(int j=0;j<7;++j)terminal_lift_blocks_[block*7+j]=lift_->add_dense_block(1,{g::N*g::NZ+(block?13:6)+j});
#endif
#ifdef JUGGLING_FIRST_EVENT_CENTER_WEIGHT
  if(JUGGLING_FIRST_EVENT_CENTER_WEIGHT>0.)for(int k=0;k<g::N;++k){auto cols=columns(k*g::NZ+6,14);auto next=columns((k+1)*g::NZ+13,7);
    cols.insert(cols.end(),next.begin(),next.end());center_lift_blocks_.push_back(lift_->add_dense_block(3,cols));}
#endif
#ifdef JUGGLING_SOLVER_LANDING_WEIGHT
  if(JUGGLING_SOLVER_LANDING_WEIGHT>0.)for(int k=1;k<=g::N;++k)landing_lift_blocks_.push_back(lift_->add_dense_block(2,columns(k*g::NZ+6,7)));
#endif
#ifdef JUGGLING_TAIL_ENDPOINT_START_S
  for(auto& term:tail_endpoint_penalties_){std::vector<int> cols;for(auto entry:term.jacobian)cols.push_back(entry[0]);
    term.lift_block=lift_->add_dense_block(1,cols);}
#endif
  lift_->finalize(A_);
}
void Controller::linearize_lifted(const Vector& at,const Eigen::Matrix<double,30,1>& parameter){
  assemble_normal_=false;
  try{linearize(at,parameter);}catch(...){assemble_normal_=true;throw;}
  assemble_normal_=true;
}
void Controller::prepare_lifted_qp(const Vector& at,const Eigen::Matrix<double,30,1>& parameter){
  // Diagnostic-only assembly: also retain the original H/gradient for a direct
  // Schur comparison. No solve, seed update, guidance mutation or OSQP update.
  linearize(at,parameter);Vector az=Vector::Zero(m_);
  for(int c=0;c<n_;++c)for(int j=A_.columns[c];j<A_.columns[c+1];++j)
    if(A_.rows[j]<m_)az[A_.rows[j]]+=A_.values[j]*at[c];
  l_.head(m_)=cl_-constraints_+az;u_.head(m_)=cu_-constraints_+az;
  for(int i=0;i<n_;++i){const double center=std::clamp(at[i],lower_[i],upper_[i]);const int j=i%g::NZ;
    const bool unlimited=(j>=13&&j<20)||(j>=g::NX&&j<g::NX+7);
    l_[m_+i]=unlimited?lower_[i]:std::max(lower_[i],center-g::trust);
    u_[m_+i]=unlimited?upper_[i]:std::min(upper_[i],center+g::trust);}
  for(int i=0;i<g::NP;++i){double measured=(parameter[i]-g::offset[i])/g::scale[i];
    l_[m_+i]=std::min(l_[m_+i],measured);u_[m_+i]=std::max(u_[m_+i],measured);}
  for(int i=0;i<l_.size();++i){if(std::isinf(l_[i]))l_[i]=-OSQP_INFTY;if(std::isinf(u_[i]))u_[i]=OSQP_INFTY;}
  lift_->prepare(A_,l_,u_,at,dual_);
}
#endif
std::array<long long,12> Controller::qp_diagnostics() const {
  long long products=0;for(const auto& f:layouts_)products+=f.hessian.size();
  for(const auto& term:tail_endpoint_penalties_)products+=term.hessian.size();
  products+=7*posture_hessian_dest_.size()+2*landing_hessian_dest_.size()+3*231*first_event_center_hessian_dest_.size();
#ifdef JUGGLING_SOLVER_TERMINAL_POSITION_WEIGHT
  products+=7*(JUGGLING_SOLVER_TERMINAL_POSITION_WEIGHT>0.)+7*(JUGGLING_SOLVER_TERMINAL_VELOCITY_WEIGHT>0.);
#endif
  long long factor_l=-1;
#ifdef JUGGLING_LIFT_PRIVATE_DIAGNOSTICS
  factor_l=juggling_osqp_factor_nnz(solver_);
#endif
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
  const long long n=lift_->variables(),m=lift_->rows();
  // [physical n, physical rows incl boxes, residuals, lifted n/m, P/A nnz,
  // upper KKT nnz, original H/A nnz, native GN products omitted, actual L nnz].
  return {{n_,m_+n_,lift_->residual.size(),n,m,(long long)lift_->P.values.size(),(long long)lift_->A.values.size(),
    (long long)lift_->P.values.size()+(long long)lift_->A.values.size()+m,
    (long long)H_.values.size(),(long long)A_.values.size(),products,factor_l}};
#else
  return {{n_,m_+n_,0,n_,m_+n_,(long long)H_.values.size(),(long long)A_.values.size(),
    (long long)H_.values.size()+(long long)A_.values.size()+m_+n_,
    (long long)H_.values.size(),(long long)A_.values.size(),products,factor_l}};
#endif
}
// Native-export residual normalization: no sparsity or constraint changes.
static double identified_cost_scale(int rows,int row){
#ifdef JUGGLING_CONTACT_CENTER_MULTIPLIER
  if(rows==57&&row>=54)return JUGGLING_CONTACT_CENTER_MULTIPLIER;
#endif
#ifdef JUGGLING_ENERGY_MULTIPLIER
  if((rows>=54||rows==5)&&row==0)return JUGGLING_ENERGY_MULTIPLIER;
  if(rows>=54&&(row==51||row==52))return JUGGLING_CONTACT_MULTIPLIER;
#endif
#ifdef JUGGLING_POSE_WEIGHT_MULTIPLIER
  if(rows>=54){
    #ifdef JUGGLING_PADDLE_LATERAL_WEIGHT_MULTIPLIER
    if(row==5||row==6)return JUGGLING_POSE_WEIGHT_MULTIPLIER*JUGGLING_PADDLE_LATERAL_WEIGHT_MULTIPLIER;
#endif
    if(row>=5&&row<8)return JUGGLING_POSE_WEIGHT_MULTIPLIER;
    if(row>=10&&row<16)return JUGGLING_VELOCITY_WEIGHT_MULTIPLIER;
    if(row>=16&&row<23)return JUGGLING_ACCEL_WEIGHT_MULTIPLIER;
    if(row>=30&&row<37)return JUGGLING_RATE_WEIGHT_MULTIPLIER;
    if(row>=37&&row<44)return JUGGLING_POSTURE_WEIGHT_MULTIPLIER;
  }
#endif
#ifdef JUGGLING_LATERAL_WEIGHT_MULTIPLIER
  if((rows>=54||rows==5)&&row>=1&&row<3)return JUGGLING_LATERAL_WEIGHT_MULTIPLIER;
#endif
#ifdef JUGGLING_BALL_VELOCITY_WEIGHT_MULTIPLIER
  if((rows>=54||rows==5)&&row>=3&&row<5)return JUGGLING_BALL_VELOCITY_WEIGHT_MULTIPLIER;
#endif
  return 1.;
}
void Controller::configure_radial_contact_scaling(){
#ifdef JUGGLING_RADIAL_CONTACT_J0_M_S
  const double j0=JUGGLING_RADIAL_CONTACT_J0_M_S;
  if(!std::isfinite(j0)||j0<0.||j0>8.)throw std::runtime_error("Radial contact impulse scale must be within 0..8 m/s");
  if(j0==0.)return;
  if(g::NZ!=34||g::scale[30]!=8.||g::offset[30]!=0.)
    throw std::runtime_error("Radial contact scaling requires the normalized impulse column 30");
  int matched=0;
  for(auto& f:layouts_)if(f.data->function.size1_in(0)==2*g::NZ){
    const auto cs=f.data->function.sparsity_out(2),vs=f.value->function.sparsity_out(1),jc=f.data->function.sparsity_out(3);
    if(f.row+cs.size1()>g::CONSTRAINTS||vs.size1()!=cs.size1())
      throw std::runtime_error("Radial contact constraint layout mismatch");
    for(int row=0;row<cs.size1();++row)if(g::constraint_ids[f.row+row]==g::contact_begin+2){
      if(f.radial_constraint_row>=0||!std::isinf(cl_[f.row+row])||cl_[f.row+row]>0.||cu_[f.row+row]!=0.)
        throw std::runtime_error("Radial scaling requires one unchanged zero-upper-bound inequality");
      f.radial_constraint_row=row;
    }
    if(f.radial_constraint_row<0)throw std::runtime_error("Missing retained radial contact constraint");
    for(int slot=0;slot<cs.nnz();++slot)if(cs.row(slot)==f.radial_constraint_row)f.radial_constraint_dest=slot;
    bool value_row=false;for(int slot=0;slot<vs.nnz();++slot)if(vs.row(slot)==f.radial_constraint_row)value_row=true;
    for(int column=0;column<jc.size2();++column)
      for(int slot=jc.colind(column);slot<jc.colind(column+1);++slot)if(jc.row(slot)==f.radial_constraint_row){
        f.radial_jacobian_dest.push_back(slot);
        if(column==30)f.radial_impulse_jacobian_dest=slot;
      }
    if(f.radial_constraint_dest<0||f.radial_impulse_jacobian_dest<0||!value_row)
      throw std::runtime_error("Radial contact scaling extends existing constraint sparsity");
    ++matched;
  }
  if(matched!=g::N)throw std::runtime_error("Radial scaling must cover every shooting interval");
#endif
}
void Controller::linearize(const Vector& at,const Eigen::Matrix<double,30,1>& parameter){
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
  lift_->clear();
#endif
  std::fill(H_.values.begin(),H_.values.end(),0.);for(int d:diagonal_)H_.values[d]=g::proximal;
  gradient_.setZero();cost_=0.;
  if(pool_)pool_->evaluate(at.data(),parameter.data());else for(auto& f:layouts_)f.data->call({at.data()+f.column,parameter.data()});
  for(auto& f:layouts_){
    auto& out=f.data->output;
    {auto rs=f.data->function.sparsity_out(0),js=f.data->function.sparsity_out(1);
     for(size_t j=0;j<out[0].size();++j)out[0][j]*=identified_cost_scale(rs.size1(),rs.row(j));
     for(size_t j=0;j<out[1].size();++j)out[1][j]*=identified_cost_scale(js.size1(),js.row(j));}
#if defined(JUGGLING_LEGACY_CONTACT_PLANE) && JUGGLING_LEGACY_CONTACT_PLANE
    if(JUGGLING_PREDICTION_PLANE_OFFSET_M!=0.&&!f.plane_constraint_dest.empty()){
      // Already normalized/weighted. The shared correction evaluator runs
      // after the worker join, before GN, merit and restitution-cap assembly.
      const double offset=JUGGLING_PREDICTION_PLANE_OFFSET_M;
      contact_plane_data_->call({at.data()+f.column,parameter.data(),g::factor_parameters.data(),&offset});
      for(const auto& entry:f.plane_residual_dest)out[0][entry[0]]+=contact_plane_data_->output[0][entry[1]];
      for(const auto& entry:f.plane_residual_jacobian_dest)out[1][entry[0]]+=contact_plane_data_->output[1][entry[1]];
      for(const auto& entry:f.plane_constraint_dest)out[2][entry[0]]+=contact_plane_data_->output[2][entry[1]];
      for(const auto& entry:f.plane_constraint_jacobian_dest)out[3][entry[0]]+=contact_plane_data_->output[3][entry[1]];
    }
#endif
#if defined(JUGGLING_LEGACY_EDGE_PREDICTION) && JUGGLING_LEGACY_EDGE_PREDICTION
    if(JUGGLING_PREDICTION_EDGE_LOSS!=0.&&f.edge_residual_dest>=0){
      // The shared helper runs after the factor pool has joined. Its scalar
      // and Jacobian already carry the actual objective/contact weights.
      const double edge_loss=JUGGLING_PREDICTION_EDGE_LOSS;
      edge_prediction_data_->call({at.data()+f.column,parameter.data(),g::factor_parameters.data(),&edge_loss});
      out[0][f.edge_residual_dest]+=edge_prediction_data_->output[0][0];
      for(const auto& entry:f.edge_jacobian_dest)
        out[1][entry[0]]+=edge_prediction_data_->output[1][entry[1]];
    }
#endif
#if defined(JUGGLING_LEGACY_CONTACT_BIAS) && JUGGLING_LEGACY_CONTACT_BIAS
    if(contact_bias_active_&&f.bias_constraint_dest[0]>=0){
      contact_bias_data_->call({at.data()+f.column,parameter.data(),g::factor_parameters.data(),bias_per_j8_local_.data()});
      for(int row=0;row<6;++row)out[2][f.bias_constraint_dest[row]]+=contact_bias_data_->output[0][row];
      for(const auto& entry:f.bias_jacobian_dest)out[3][entry[0]]+=contact_bias_data_->output[1][entry[1]];
    }
#endif
#ifdef JUGGLING_RADIAL_CONTACT_J0_M_S
    if(f.radial_constraint_row>=0){
      // Positive nonlinear row scaling preserves the exact zero-bound feasible
      // set. Differentiate the scale as well; this is not QP-only preconditioning.
      const double j0=JUGGLING_RADIAL_CONTACT_J0_M_S,j=at[f.column+30]*g::scale[30]+g::offset[30];
      const double denominator=std::hypot(j,j0),s=std::hypot(8.,j0)/denominator;
      const double original=out[2][f.radial_constraint_dest];
      out[2][f.radial_constraint_dest]=s*original;
      for(int slot:f.radial_jacobian_dest)out[3][slot]*=s;
      out[3][f.radial_impulse_jacobian_dest]-=s*(j/denominator)/denominator*original*g::scale[30];
    }
#endif
#ifdef JUGGLING_FIXED_MIDPOINT_POSTURE
    transform_midpoint_data(f);
#endif
    if(g::quadrature && f.data->function.size1_out(0)>=54){
      double weight=std::sqrt(f.step/.03);auto sparsity=f.data->function.sparsity_out(1);
      auto rs=f.data->function.sparsity_out(0);for(size_t j=0;j<out[0].size();++j){int r=rs.row(j);if(r<51)out[0][j]*=weight*((r>=30&&r<37)?.03/f.step:1.);}
      for(size_t j=0;j<out[1].size();++j){int r=sparsity.row(j);if(r<51)out[1][j]*=weight*((r>=30&&r<37)?.03/f.step:1.);}
    }
    Vector residual=f.data->dense(0);reference_residual(residual,g::quadrature&&residual.size()>=54?std::sqrt(f.step/.03):1.);
    Vector con=f.data->dense(2);constraints_.segment(f.row,con.size())=con;
    cost_+=.5*residual.squaredNorm();auto jr=f.data->function.sparsity_out(1);
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
    lift_->set_sparse(f.lift_block,residual,out[1]);
    if(assemble_normal_){
#endif
    for(int c=0;c<jr.size2();++c)for(int j=jr.colind(c);j<jr.colind(c+1);++j)gradient_[f.column+c]+=out[1][j]*residual[jr.row(j)];
    for(auto product:f.hessian)H_.values[product.destination]+=out[1][product.left]*out[1][product.right];
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
    }
#endif
    for(size_t j=0;j<f.a_dest.size();++j)A_.values[f.a_dest[j]]=out[3][j];
    if(f.restitution_cap_row>=0){
      // Row 52 is an event residual, unaffected by mesh quadrature/reference
      // shifts. Remove its cost scaling to obtain the exact NLP expression
      // (j/8)*(restitution/8), and use the same scaled Jacobian row.
      constraints_[f.restitution_cap_row]=residual[52]/restitution_cap_normalization_;
      for(const auto& entry:f.restitution_cap_a)
        A_.values[entry[0]]=out[1][entry[1]]/restitution_cap_normalization_;
    }
  }
  cost_+=posture_regularization(at,true)+terminal_regularization(at,true);
#ifdef JUGGLING_TAIL_ENDPOINT_START_S
  if(!tail_endpoint_penalties_.empty())cost_+=tail_endpoint_regularization(constraints_,true);
#endif
#ifdef JUGGLING_SOLVER_LANDING_WEIGHT
  if(JUGGLING_SOLVER_LANDING_WEIGHT>0.)cost_+=landing_guidance(at,true);
#endif
#ifdef JUGGLING_FIRST_EVENT_CENTER_WEIGHT
  if(JUGGLING_FIRST_EVENT_CENTER_WEIGHT>0.)cost_+=first_event_center_cost(at,true);
#endif
  if(!gradient_.allFinite()||!constraints_.allFinite()||!std::isfinite(cost_)||
     !std::all_of(H_.values.begin(),H_.values.end(),[](double v){return std::isfinite(v);})||
     !std::all_of(A_.values.begin(),A_.values.end(),[](double v){return std::isfinite(v);}))throw std::runtime_error("Non-finite SQP data");
}
double Controller::value(const Vector& at,const Eigen::Matrix<double,30,1>& p,Vector& constraints){
  double cost=0.;
  for(auto& f:layouts_){f.value->call({at.data()+f.column,p.data()});Vector residual=f.value->dense(0);for(int j=0;j<residual.size();++j)residual[j]*=identified_cost_scale(residual.size(),j);
#if defined(JUGGLING_LEGACY_CONTACT_PLANE) && JUGGLING_LEGACY_CONTACT_PLANE
    if(JUGGLING_PREDICTION_PLANE_OFFSET_M!=0.&&!f.plane_constraint_dest.empty()){
      const double offset=JUGGLING_PREDICTION_PLANE_OFFSET_M;
      contact_plane_value_->call({at.data()+f.column,p.data(),g::factor_parameters.data(),&offset});
      for(const auto& entry:f.plane_residual_rows)residual[entry[0]]+=contact_plane_value_->output[0][entry[1]];
    }
#endif
#if defined(JUGGLING_LEGACY_EDGE_PREDICTION) && JUGGLING_LEGACY_EDGE_PREDICTION
    if(JUGGLING_PREDICTION_EDGE_LOSS!=0.&&f.edge_residual_dest>=0){
      const double edge_loss=JUGGLING_PREDICTION_EDGE_LOSS;
      edge_prediction_value_->call({at.data()+f.column,p.data(),g::factor_parameters.data(),&edge_loss});
      residual[52]+=edge_prediction_value_->output[0][0];
    }
#endif
    reference_residual(residual);if(g::quadrature&&residual.size()>=54){residual.head(51)*=std::sqrt(f.step/.03);residual.segment(30,7)*=.03/f.step;}cost+=.5*residual.squaredNorm();Vector c=f.value->dense(1);
#if defined(JUGGLING_LEGACY_CONTACT_PLANE) && JUGGLING_LEGACY_CONTACT_PLANE
    if(JUGGLING_PREDICTION_PLANE_OFFSET_M!=0.&&!f.plane_constraint_dest.empty())
      for(const auto& entry:f.plane_constraint_rows)c[entry[0]]+=contact_plane_value_->output[1][entry[1]];
#endif
#if defined(JUGGLING_LEGACY_CONTACT_BIAS) && JUGGLING_LEGACY_CONTACT_BIAS
    if(contact_bias_active_&&f.bias_constraint_dest[0]>=0){
      contact_bias_value_->call({at.data()+f.column,p.data(),g::factor_parameters.data(),bias_per_j8_local_.data()});
      for(int row=0;row<6;++row)c[row]+=contact_bias_value_->output[0][row];
    }
#endif
#ifdef JUGGLING_RADIAL_CONTACT_J0_M_S
    if(f.radial_constraint_row>=0){
      const double j0=JUGGLING_RADIAL_CONTACT_J0_M_S,j=at[f.column+30]*g::scale[30]+g::offset[30];
      c[f.radial_constraint_row]*=std::hypot(8.,j0)/std::hypot(j,j0);
    }
#endif
#ifdef JUGGLING_FIXED_MIDPOINT_POSTURE
    transform_midpoint_value(f,c);
#endif
    constraints.segment(f.row,c.size())=c;
    if(f.restitution_cap_row>=0)constraints[f.restitution_cap_row]=residual[52]/restitution_cap_normalization_;
  }
#ifdef JUGGLING_SOLVER_LANDING_WEIGHT
  if(JUGGLING_SOLVER_LANDING_WEIGHT>0.)cost+=landing_guidance(at,false);
#endif
#ifdef JUGGLING_TAIL_ENDPOINT_START_S
  if(!tail_endpoint_penalties_.empty())cost+=tail_endpoint_regularization(constraints,false);
#endif
#ifdef JUGGLING_FIRST_EVENT_CENTER_WEIGHT
  if(JUGGLING_FIRST_EVENT_CENTER_WEIGHT>0.)cost+=first_event_center_cost(at,false);
#endif
  return cost+posture_regularization(at,false)+terminal_regularization(at,false);
}
Geometry Controller::prediction_geometry(const State& x){
  auto result=robot.geometry(x);
#if defined(JUGGLING_LEGACY_CONTACT_PLANE) && JUGGLING_LEGACY_CONTACT_PLANE
  // Translating by delta*n leaves tangent, radial distance and d(gap)/dt
  // unchanged: n'*(delta*n_dot)=0. Keep shared plant geometry untouched.
  if(JUGGLING_PREDICTION_PLANE_OFFSET_M!=0.)result.gap-=JUGGLING_PREDICTION_PLANE_OFFSET_M;
#endif
  return result;
}
std::array<Matrix,5> Controller::prediction_contact_fk(const Joint& q){
  auto f=robot.contact_fk(q);
#if defined(JUGGLING_LEGACY_CONTACT_PLANE) && JUGGLING_LEGACY_CONTACT_PLANE
  if(JUGGLING_PREDICTION_PLANE_OFFSET_M!=0.){
    const Eigen::Vector3d offset=JUGGLING_PREDICTION_PLANE_OFFSET_M*f[1].col(0);
    f[0].col(0)+=offset;f[2]-=skew(offset)*f[3];
  }
#endif
  return f;
}
Vector Controller::prediction_stage_constraints(const double* stage,const double* parameter){
  stage_value_.call({stage,parameter});Vector constraints=stage_value_.dense(1);
#if defined(JUGGLING_LEGACY_CONTACT_PLANE) && JUGGLING_LEGACY_CONTACT_PLANE
  if(JUGGLING_PREDICTION_PLANE_OFFSET_M!=0.){
    const double offset=JUGGLING_PREDICTION_PLANE_OFFSET_M;
    contact_plane_value_->call({stage,parameter,g::factor_parameters.data(),&offset});
    const auto support=contact_plane_value_->function.sparsity_out(1);
    for(int i=0;i<support.nnz();++i)
      constraints[contact_plane_constraint_ids[support.row(i)]]+=contact_plane_value_->output[1][i];
  }
#endif
  return constraints;
}
State Controller::free(const State& x,double dt,const Joint& acc) const {
  State out=x;Eigen::Vector3d gravity=Eigen::Map<const Eigen::Vector3d>(ball_gravity.data());
  #ifdef JUGGLING_BALL_DRAG
  const double drag=JUGGLING_BALL_DRAG;
#else
  constexpr double drag=0.;
#endif
  const double decay=std::exp(-drag*dt), vd=drag>0?-std::expm1(-drag*dt)/drag:dt;
  const double pd=drag>0?(dt-vd)/drag:.5*dt*dt;
  out.head<3>()+=vd*x.segment<3>(3)+pd*gravity;out.segment<3>(3)=decay*x.segment<3>(3)+vd*gravity;
  out.segment<7>(6)+=dt*x.segment<7>(13)+.5*dt*dt*acc;out.segment<7>(13)+=dt*acc;return out;
}
#if defined(JUGGLING_RAMP_PREDICTION) && JUGGLING_RAMP_PREDICTION
State Controller::free_ramp(const State& x,double dt,const Joint& acc,const Joint& jerk) const {
  State out=free(x,dt,acc);out.segment<7>(6)+=(dt*dt*dt/6.)*jerk;
  out.segment<7>(13)+=(.5*dt*dt)*jerk;return out;
}
#endif
static double apply_ball_impact_prediction(RobotModel& robot,State& x,bool virtual_proposal,double legacy_plane_offset=0.,bool neglect_arm_reaction=false){
#if defined(JUGGLING_RAMP_PREDICTION) && JUGGLING_RAMP_PREDICTION
  auto f=robot.contact_fk(x.segment<7>(6));prediction_plane(f);
  Eigen::Vector3d n=f[1],rel=x.head<3>()-f[0].col(0)+map(g::origin),tangent=rel-n*(n.dot(rel));
  const double restitution=std::clamp(g::restitution-JUGGLING_PREDICTION_EDGE_LOSS*std::pow(tangent.squaredNorm()/(.075*.075),2),0.,1.5);
#else
  auto f=robot.contact_fk(x.segment<7>(6));
#if defined(JUGGLING_LEGACY_CONTACT_PLANE) && JUGGLING_LEGACY_CONTACT_PLANE
  // Only Controller::impact supplies a nonzero offset. The public
  // apply_ball_impact API and physical plant retain their original geometry.
  if(legacy_plane_offset!=0.){
    const Eigen::Vector3d offset=legacy_plane_offset*f[1].col(0);
    f[0].col(0)+=offset;f[2]-=skew(offset)*f[3];
  }
#endif
  Eigen::Vector3d n=f[1],rel=x.head<3>()-f[0].col(0)+map(g::origin),tangent=rel-n*(n.dot(rel));
#if defined(JUGGLING_LEGACY_EDGE_PREDICTION) && JUGGLING_LEGACY_EDGE_PREDICTION
  double restitution=g::restitution;
  // A virtual mode presumes the optimizer will move the paddle underneath it.
  // Its deliberately inflated radius must not seed a zero-restitution edge hit.
  // Ordinary support contacts and all exact NLP predictions use the edge law.
  if(!virtual_proposal&&JUGGLING_PREDICTION_EDGE_LOSS!=0.)
    restitution=std::clamp(g::restitution-JUGGLING_PREDICTION_EDGE_LOSS*std::pow(tangent.squaredNorm()/(.075*.075),2),0.,1.5);
#endif
#endif
  Matrix J=f[2]-skew(tangent)*f[3];Matrix response;
  if(!neglect_arm_reaction){auto M=robot.mass(x.segment<7>(6)).ldlt();response=M.solve(J.transpose());}
  Eigen::Vector3d relative=x.segment<3>(3)-x.segment<3>(20).cross(g::radius*n)-J*x.segment<7>(13);
  Matrix K;
  if(neglect_arm_reaction)K=Matrix::Identity(3,3)+(Matrix::Identity(3,3)-n*n.transpose())/g::inertia_ratio;
  else K=Matrix::Identity(3,3)+(Matrix::Identity(3,3)-n*n.transpose())/g::inertia_ratio+g::mass*J*response;
  Matrix T=f[4].leftCols(2);auto tangent_mass=(T.transpose()*K*T).eval().ldlt();double vn=n.dot(relative);
  auto impulse=[&](double j)->Eigen::Vector3d{Eigen::Vector2d desired=-tangent_mass.solve(T.transpose()*(relative+K*n*j));double cap=g::friction*j;return j*n+T*(desired*cap/std::sqrt(cap*cap+desired.squaredNorm()+1e-10));};
  double lo=0.,hi=std::max(1.,-3*vn);
  // Monotone normal equation; bisection tolerance is tighter than the root
  // tolerance of the reference NumPy/Brent warmstart.
#if (defined(JUGGLING_RAMP_PREDICTION) && JUGGLING_RAMP_PREDICTION) || (defined(JUGGLING_LEGACY_EDGE_PREDICTION) && JUGGLING_LEGACY_EDGE_PREDICTION)
  for(int k=0;k<50;++k){double mid=.5*(lo+hi);if(n.dot(K*impulse(mid))+(1+restitution)*vn>0)hi=mid;else lo=mid;}
#else
  for(int k=0;k<50;++k){double mid=.5*(lo+hi);if(n.dot(K*impulse(mid))+(1+g::restitution)*vn>0)hi=mid;else lo=mid;}
#endif
  double j=.5*(lo+hi);Eigen::Vector3d u=impulse(j);x.segment<3>(3)+=u;x.segment<3>(20)-=n.cross(u)/(g::inertia_ratio*g::radius);
#if defined(JUGGLING_RAMP_PREDICTION) && JUGGLING_RAMP_PREDICTION
  // The executed joint-velocity servo prescribes the arm arc; an impulsive
  // synthetic dq change here would disagree with the ramp shooting equations.
  double gap=n.dot(rel)-g::radius;x.head<3>()+=(1e-10-gap)*n;return g::mass*j;
#else
  if(!neglect_arm_reaction)x.segment<7>(13)-=g::mass*response*u;
  double gap=n.dot(rel)-g::radius;x.head<3>()+=(1e-10-gap)*n;return g::mass*j;
#endif
}
double apply_ball_impact(RobotModel& robot,State& x){return apply_ball_impact_prediction(robot,x,false);}
void Controller::impact(State& x,Contact& contact,bool virtual_proposal,bool kinematic_tail){
#if defined(JUGGLING_NEGLECT_BALL_REACTION) && JUGGLING_NEGLECT_BALL_REACTION
  constexpr bool neglect_arm_reaction=true;
#elif defined(JUGGLING_KINEMATIC_DYNAMIC_PREFIX)
  const bool neglect_arm_reaction=kinematic_tail;
#else
  constexpr bool neglect_arm_reaction=false;
#endif
#if defined(JUGGLING_LEGACY_CONTACT_PLANE) && JUGGLING_LEGACY_CONTACT_PLANE
  contact.impulse=apply_ball_impact_prediction(robot,x,virtual_proposal,JUGGLING_PREDICTION_PLANE_OFFSET_M,neglect_arm_reaction);
#else
  contact.impulse=apply_ball_impact_prediction(robot,x,virtual_proposal,0.,neglect_arm_reaction);
#endif
#if defined(JUGGLING_LEGACY_CONTACT_BIAS) && JUGGLING_LEGACY_CONTACT_BIAS
  if(contact_bias_active_){
    const auto f=robot.contact_fk(x.segment<7>(6));
    // A phenomenological ball kick only; nominal arm/spin impulse is unchanged.
    // q is unchanged by the nominal impact, so this is the same contact frame.
    x.segment<3>(3)+=(contact.impulse/(8.*g::mass))*(f[4]*bias_per_j8_local_);
  }
#endif
}
#ifdef JUGGLING_SOLVER_WARMSTART_SEED_RADIUS
void Controller::trace_virtual_seed(const SeedContext& seed,bool applied,const char* outcome,const SeedMeritScreen* merit){
  if(seed.contacts==0||!seed_trace_)return;
  // Each proposed mode gets a row. Applied/outcome describe the complete
  // regeneration, so a later failure marks every preceding proposal discarded.
  for(int i=0;i<seed.contacts;++i){const auto& event=seed.events[i];
    *seed_trace_<<"{\"event\":\"warmstart_only_virtual_contact\",\"controller_time_s\":"<<seed_controller_time_
      <<",\"stage\":"<<event.stage<<",\"lead_s\":"<<event.lead<<",\"radial_m\":"<<event.radius
      <<",\"seed_radius_m\":"<<seed.radius<<",\"nominal_seed_radius_m\":"<<g::paddle_radius
      <<",\"minimum_lead_s\":"<<seed.minimum_lead<<",\"normal_impulse_Ns\":"<<event.impulse
      <<",\"applied\":"<<(applied?"true":"false")<<",\"fallback_stage\":"<<fallback_stage_
      <<",\"outcome\":\""<<outcome<<"\",\"event_index\":"<<i+1<<",\"max_contacts\":"<<seed.max_contacts
      <<",\"edge_correction_bypassed_for_virtual_seed\":"<<(event.edge_correction_bypassed?"true":"false");
#ifdef JUGGLING_SOLVER_WARMSTART_REPAIR
    if(JUGGLING_SOLVER_WARMSTART_REPAIR==3.){
      *seed_trace_<<",\"seed_selection_mode\":3,\"contact_ordinal\":"<<event.contact_ordinal
        <<",\"first_virtual_contact_ordinal\":"<<seed.events[0].contact_ordinal
        <<",\"candidate_contact_count\":"<<seed.all_contacts
        <<",\"candidate_first_contact_lead_s\":"<<seed.first_contact_lead
        <<",\"first_upcoming_contact_virtual\":"<<(seed.events[0].contact_ordinal==1?"true":"false");
    }
#endif
    if(merit){
      *seed_trace_<<",\"merit_screen_mode\":"<<merit->mode<<",\"merit_penalty\":"<<merit->penalty;
      auto point=[&](const char* name,const SeedMeritScreen::Point& p){
        *seed_trace_<<",\""<<name<<"\":{\"evaluated\":"<<(p.evaluated?"true":"false");
        auto number=[&](const char* key,double v){*seed_trace_<<",\""<<key<<"\":";
          if(p.evaluated&&std::isfinite(v))*seed_trace_<<v;else *seed_trace_<<"null";};
        number("cost",p.cost);number("merit",p.merit);
        number("nonlinear_l1",p.constraint_l1);number("nonlinear_max",p.constraint_max);
        number("bounds_l1",p.bound_l1);number("bounds_max",p.bound_max);*seed_trace_<<"}";
      };
      point("shifted",merit->shifted);point("regenerated",merit->regenerated);
    }
    *seed_trace_<<"}\n";
  }
  seed_trace_->flush();
}
#endif
std::pair<State,std::vector<Contact>> Controller::rollout(const State& x,double duration,const Joint& acc,double max_step,const Joint* end_acc,SeedContext* seed,int prediction_stage){
#ifdef JUGGLING_KINEMATIC_DYNAMIC_PREFIX
  const bool tail=kinematic_stage(prediction_stage);
#else
  constexpr bool tail=false;
#endif
#if defined(JUGGLING_RAMP_PREDICTION) && JUGGLING_RAMP_PREDICTION
  Joint jerk=Joint::Zero();if(end_acc&&duration>0)jerk=(*end_acc-acc)/duration;
  State state=x;std::vector<Contact> contacts;double elapsed=0.;
  while(elapsed<duration-1e-14){
    Joint local_acc=acc+elapsed*jerk;
    auto advance=[&](double delta){return free_ramp(state,delta,local_acc,jerk);};
    double dt=std::min(max_step,duration-elapsed);State end=advance(dt);auto a=prediction_geometry(state),b=prediction_geometry(end);
    if(a.gap>=-1e-10&&b.gap<=0&&(a.speed<0||b.speed<0)){
      double lo=0.,hi=dt;while(hi-lo>1e-10){double mid=.5*(lo+hi);if(prediction_geometry(advance(mid)).gap>0)lo=mid;else hi=mid;}
      double hit=.5*(lo+hi);State collision=advance(hit);auto geo=prediction_geometry(collision);
#ifdef JUGGLING_SOLVER_WARMSTART_SEED_RADIUS
      // A mode-discovery seed only: the NLP/observer/plant keep their original
      // contact support. The context is never supplied during initialization.
      const double lead=seed?seed->stage_start+elapsed+hit:0.;
      const bool virtual_contact=seed&&seed->contacts<seed->max_contacts&&lead>seed->minimum_lead&&lead>.3
        &&geo.radial>g::paddle_radius&&geo.radial<=seed->radius&&geo.speed<0;
      if((geo.radial<=g::paddle_radius||virtual_contact)&&geo.speed<0){Contact c;impact(collision,c,virtual_contact,tail);
        // Count completed impact resets, including ordinary contacts, across
        // every shooting interval. Crossing attempts without a reset do not
        // consume an ordinal; multiple contacts in one interval remain ordered.
        if(seed&&++seed->all_contacts==1)seed->first_contact_lead=lead;
        if(virtual_contact){auto& event=seed->events[seed->contacts++];event.stage=seed->stage;event.lead=lead;event.radius=geo.radial;event.impulse=c.impulse;event.contact_ordinal=seed->all_contacts;
#ifdef JUGGLING_SOLVER_WARMSTART_ARM_SHIFT
          if(JUGGLING_SOLVER_WARMSTART_ARM_SHIFT>0.)event.ball_contact=collision.head<3>();
#endif
#if defined(JUGGLING_LEGACY_EDGE_PREDICTION) && JUGGLING_LEGACY_EDGE_PREDICTION
          event.edge_correction_bypassed=JUGGLING_PREDICTION_EDGE_LOSS!=0.;
#endif
        }
        elapsed+=hit;c.time=elapsed;contacts.push_back(c);state=collision;if(contacts.size()>16)throw std::runtime_error("Unsupported resting contact");continue;}
#else
      if(geo.radial<=g::paddle_radius&&geo.speed<0){Contact c;impact(collision,c,false,tail);elapsed+=hit;c.time=elapsed;contacts.push_back(c);state=collision;if(contacts.size()>16)throw std::runtime_error("Unsupported resting contact");continue;}
#endif
    }
    state=end;elapsed+=dt;
  }
  return {state,contacts};
#else
  State state=x;std::vector<Contact> contacts;double elapsed=0.;
  while(elapsed<duration-1e-14){
    double dt=std::min(max_step,duration-elapsed);State end=free(state,dt,acc);auto a=prediction_geometry(state),b=prediction_geometry(end);
    if(a.gap>=-1e-10&&b.gap<=0&&(a.speed<0||b.speed<0)){
      double lo=0.,hi=dt;while(hi-lo>1e-10){double mid=.5*(lo+hi);if(prediction_geometry(free(state,mid,acc)).gap>0)lo=mid;else hi=mid;}
      double hit=.5*(lo+hi);State collision=free(state,hit,acc);auto geo=prediction_geometry(collision);
#ifdef JUGGLING_SOLVER_WARMSTART_SEED_RADIUS
      const double lead=seed?seed->stage_start+elapsed+hit:0.;
      const bool virtual_contact=seed&&seed->contacts<seed->max_contacts&&lead>seed->minimum_lead&&lead>.3
        &&geo.radial>g::paddle_radius&&geo.radial<=seed->radius&&geo.speed<0;
      if((geo.radial<=g::paddle_radius||virtual_contact)&&geo.speed<0){Contact c;impact(collision,c,virtual_contact,tail);
        if(seed&&++seed->all_contacts==1)seed->first_contact_lead=lead;
        if(virtual_contact){auto& event=seed->events[seed->contacts++];event.stage=seed->stage;event.lead=lead;event.radius=geo.radial;event.impulse=c.impulse;event.contact_ordinal=seed->all_contacts;
#ifdef JUGGLING_SOLVER_WARMSTART_ARM_SHIFT
          if(JUGGLING_SOLVER_WARMSTART_ARM_SHIFT>0.)event.ball_contact=collision.head<3>();
#endif
#if defined(JUGGLING_LEGACY_EDGE_PREDICTION) && JUGGLING_LEGACY_EDGE_PREDICTION
          event.edge_correction_bypassed=JUGGLING_PREDICTION_EDGE_LOSS!=0.;
#endif
        }
        elapsed+=hit;c.time=elapsed;contacts.push_back(c);state=collision;if(contacts.size()>16)throw std::runtime_error("Unsupported resting contact");continue;}
#else
      if(geo.radial<=g::paddle_radius&&geo.speed<0){Contact c;impact(collision,c,false,tail);elapsed+=hit;c.time=elapsed;contacts.push_back(c);state=collision;if(contacts.size()>16)throw std::runtime_error("Unsupported resting contact");continue;}
#endif
    }
    state=end;elapsed+=dt;
  }
  return {state,contacts};
#endif
}
#ifdef JUGGLING_SOLVER_WARMSTART_ARM_SHIFT
void Controller::try_arm_shift_seed(const State& x,const Joint& torque,const SeedContext& seed){
  if(JUGGLING_SOLVER_WARMSTART_ARM_SHIFT==0.||seed.contacts==0)return;
  // This is one bounded initialization proposal. Neither the NLP nor the
  // physical model is changed, and z_/dual_ stay untouched until acceptance.
  const auto& event=seed.events[0];int start=0,finish=event.stage;
  double displacement=0.,joint_shift=0.,radial_after=0.,old_merit=0.,new_merit=0.,old_arm=0.,new_arm=0.;
  const char* outcome="candidate_not_evaluated";bool accepted=false;
  auto log=[&](){if(seed_trace_){
    *seed_trace_<<"{\"event\":\"warmstart_only_arm_shift\",\"controller_time_s\":"<<seed_controller_time_
      <<",\"source_stage\":"<<event.stage<<",\"source_lead_s\":"<<event.lead<<",\"source_radius_m\":"<<event.radius
      <<",\"start_node\":"<<start<<",\"finish_node\":"<<finish<<",\"requested_shift_m\":"<<displacement
      <<",\"max_joint_shift_rad\":"<<joint_shift<<",\"static_radius_after_m\":"<<radial_after
      <<",\"old_arm_violation\":"<<old_arm<<",\"new_arm_violation\":"<<new_arm
      <<",\"old_merit\":"<<old_merit<<",\"new_merit\":"<<new_merit
      <<",\"applied\":"<<(accepted?"true":"false")<<",\"outcome\":\""<<outcome<<"\"}\n";
    seed_trace_->flush();}};
  // Freeze all state and torque variables through >=200 ms. The contact j/f
  // at that node describe the following interval and may be regenerated.
  // Finish at the LEFT endpoint of the impact interval, not at impact.
  while(start<g::N&&g::stage_times[start]<.2-1e-12)++start;
  if(finish<=start||finish>=g::N||g::stage_times[finish]-g::stage_times[start]<.25){
    outcome="insufficient_tail_time";log();return;}
  try{
    Matrix Y(g::NZ,g::N+1);
    for(int k=0;k<=g::N;++k)Y.col(k)=z_.segment<g::NZ>(k*g::NZ).cwiseProduct(map(g::scale))+map(g::offset);
    const double contact_dt=event.lead-g::stage_times[finish];
    if(contact_dt<0.||contact_dt>g::stage_steps[finish]+1e-9){outcome="invalid_source_event";log();return;}
    const Joint acc=(Y.col(finish+1).segment<7>(13)-Y.col(finish).segment<7>(13))/g::stage_steps[finish];
    const Joint qc=Y.col(finish).segment<7>(6)+contact_dt*Y.col(finish).segment<7>(13)+.5*contact_dt*contact_dt*acc;
    auto fk=prediction_contact_fk(qc);const Eigen::Vector3d normal=fk[1];
    const Eigen::Vector3d relative=event.ball_contact-(fk[0]-map(g::origin));
    const Eigen::Vector3d tangent=relative-normal*normal.dot(relative);
    const double radius=tangent.norm(),usable=g::paddle_radius-g::contact_edge_margin;
    const double target=std::min(.055,usable-.010);
    if(target<=0.||radius<=target){outcome="no_small_radial_repair";log();return;}
    displacement=radius-target;
    if(displacement>.030){outcome="translation_exceeds_30mm";log();return;}
    const Matrix T=fk[4].leftCols(2);Matrix protect(3,7);
    protect.row(0)=normal.transpose()*fk[2];protect.bottomRows(2)=.15*T.transpose()*fk[3];
    Eigen::JacobiSVD<Matrix> svd(protect,Eigen::ComputeFullV);Matrix projector=Matrix::Identity(7,7);
    for(int i=0;i<svd.singularValues().size();++i)if(svd.singularValues()[i]>1e-8)
      projector-=svd.matrixV().col(i)*svd.matrixV().col(i).transpose();
    const Matrix A=T.transpose()*fk[2];const Eigen::Vector2d request=T.transpose()*tangent*(displacement/radius);
    const Joint delta=projector*A.transpose()*(A*projector*A.transpose()+1e-4*Eigen::Matrix2d::Identity()).ldlt().solve(request);
    joint_shift=delta.cwiseAbs().maxCoeff();
    if(!delta.allFinite()||joint_shift>.08){outcome="joint_shift_exceeds_008rad";log();return;}
    const auto shifted_fk=prediction_contact_fk(qc+delta);const Eigen::Vector3d shifted_normal=shifted_fk[1];
    const Eigen::Vector3d shifted_relative=event.ball_contact-(shifted_fk[0]-map(g::origin));
    radial_after=(shifted_relative-shifted_normal*shifted_normal.dot(shifted_relative)).norm();
    if(radial_after>usable-.002||std::abs(shifted_normal.dot(shifted_relative)-normal.dot(relative))>.001
       ||normal.dot(shifted_normal)<std::cos(.5*std::acos(-1.)/180.)){
      outcome="nonlinear_geometry_check_failed";log();return;}
    // Sample a quintic velocity shape and integrate it with exactly the same
    // trapezoid rule as the position equality. Existing position defects are
    // preserved, with no new velocity jump at either end of the translation.
    std::array<double,g::N+1> rate{},integral{};
    const double duration=g::stage_times[finish]-g::stage_times[start];
    for(int k=start+1;k<finish;++k){const double u=(g::stage_times[k]-g::stage_times[start])/duration;
      rate[k]=30.*u*u*(1.-u)*(1.-u)/duration;}
    for(int k=start;k<g::N;++k)integral[k+1]=integral[k]+.5*g::stage_steps[k]*(rate[k]+rate[k+1]);
    if(integral[finish]<=1e-8){outcome="insufficient_interior_knots";log();return;}
    for(int k=start+1;k<=g::N;++k){Y.col(k).segment<7>(6)+=(integral[k]/integral[finish])*delta;
      Y.col(k).segment<7>(13)+=(rate[k]/integral[finish])*delta;}
    // Start from the unchanged causal prefix endpoint. Never import a plant
    // state or an oracle velocity, and never insert another virtual contact.
    State state=Y.col(start).head<g::NX>();bool repaired_contact=false;
    for(int k=start;k<g::N;++k){Y.col(k).head<6>()=state.head<6>();Y.col(k).segment<3>(20)=state.segment<3>(20);
      Y(g::NX+7,k)=0.;Y(g::NX+8,k)=.5;
      const Joint a=(Y.col(k+1).segment<7>(13)-Y.col(k).segment<7>(13))/g::stage_steps[k];
      const auto out=rollout(Y.col(k).head<g::NX>(),g::stage_steps[k],a,.01);state=out.first;
      if(!state.allFinite()||(guard_ball_rollout&&(state[2]<-.15||state[2]>1.5))||out.second.size()>1){
        outcome="ordinary_tail_rollout_failed";log();return;}
      if(!out.second.empty()){const auto& hit=out.second[0];
        Y(g::NX+7,k)=hit.impulse/g::mass;Y(g::NX+8,k)=hit.time/g::stage_steps[k];
        const State before=free(Y.col(k).head<g::NX>(),hit.time,a);
        const auto geometry=prediction_geometry(before);
        if(std::abs(g::stage_times[k]+hit.time-event.lead)<=.5*g::stage_steps[finish]+.002&&geometry.radial<=usable-.002)
          repaired_contact=true;}}
    if(!repaired_contact){outcome="physical_radius_contact_not_recovered";log();return;}
    Y.col(g::N).head<6>()=state.head<6>();Y.col(g::N).segment<3>(20)=state.segment<3>(20);
    Eigen::Matrix<double,30,1> parameter;parameter<<x,torque;
    Vector candidate=z_;
    for(int k=start;k<=g::N;++k)candidate.segment<g::NZ>(k*g::NZ)=(Y.col(k)-map(g::offset)).cwiseQuotient(map(g::scale));
    // Preserve the prefix endpoint bit-for-bit, including its torque. Only
    // the interval's contact variables at that node may be regenerated.
    candidate.segment<30>(start*g::NZ)=z_.segment<30>(start*g::NZ);
    candidate.segment<2>(start*g::NZ+32)=z_.segment<2>(start*g::NZ+32);
    // Recover exactly the ORIGINAL midpoint force balance. Its arm impulse
    // uses the nominal contact law, never the phenomenological ball-only bias.
    // c_dyn=h/2*M_mid^-1*(F-(tau_k+tau_k1)/2), so solve for tau_k1.
    for(int k=start;k<g::N;++k){
      const Vector c=prediction_stage_constraints(candidate.data()+k*g::NZ,parameter.data());
      const Joint qm=.5*(Y.col(k).segment<7>(6)+Y.col(k+1).segment<7>(6));
      const Joint force=.5*(Y.col(k).segment<7>(g::NX)+Y.col(k+1).segment<7>(g::NX))
        +(2./g::stage_steps[k])*robot.mass(qm)*c.segment<7>(g::seed_dynamics_begin);
      Y.col(k+1).segment<7>(g::NX)=2.*force-Y.col(k).segment<7>(g::NX);
      candidate.segment<g::NZ>((k+1)*g::NZ)=(Y.col(k+1)-map(g::offset)).cwiseQuotient(map(g::scale));}
    if(!candidate.allFinite()){outcome="nonfinite_candidate";log();return;}
    auto violation=[](double y,double lo,double hi){return std::max({0.,lo-y,y-hi});};
    // No newly material arm defect, including full-stage tail constraints
    // omitted from this candidate's NLP. Torque reconstruction is not clipped.
    for(int k=0;k<g::N;++k){
      const Vector before=prediction_stage_constraints(z_.data()+k*g::NZ,parameter.data());
      const Vector after=prediction_stage_constraints(candidate.data()+k*g::NZ,parameter.data());
      if(!after.allFinite()){outcome="nonfinite_arm_constraints";log();return;}
      for(int id:g::seed_arm_ids){const double a=violation(before[id],g::seed_arm_lower[id],g::seed_arm_upper[id]);
        const double b=violation(after[id],g::seed_arm_lower[id],g::seed_arm_upper[id]);old_arm=std::max(old_arm,a);new_arm=std::max(new_arm,b);
        if(b>std::max(a,1e-3)+1e-9){outcome="new_arm_constraint_defect";log();return;}}
      if(k>=start&&after.segment<7>(g::seed_dynamics_begin).cwiseAbs().maxCoeff()>1e-8){
        outcome="midpoint_force_balance_not_exact";log();return;}}
    for(int k=0;k<=g::N;++k)for(int i=0;i<g::NZ;++i){const int row=k*g::NZ+i;
      const double a=violation(z_[row],lower_[row],upper_[row]),b=violation(candidate[row],lower_[row],upper_[row]);
      if(b>std::max(a,1e-3)+1e-9){outcome="new_variable_bound_violation";log();return;}}
#ifdef JUGGLING_SOLVER_MERIT_MULTIPLIER
    const double penalty=JUGGLING_SOLVER_MERIT_MULTIPLIER;
#else
    constexpr double penalty=1000.;
#endif
    Vector old_constraints(m_),new_constraints(m_);
    const double old_cost=value(z_,parameter,old_constraints),new_cost=value(candidate,parameter,new_constraints);
    const double old_total=old_cost+penalty*g::objective_scale*violation_sum(old_constraints,z_);
    const double new_total=new_cost+penalty*g::objective_scale*violation_sum(new_constraints,candidate);
    if(!std::isfinite(old_total)||!std::isfinite(new_total)){outcome="nonfinite_merit";log();return;}
    old_merit=old_total;new_merit=new_total;
    if(new_merit>=old_merit-1e-10){outcome="full_merit_did_not_improve";log();return;}
    z_=candidate;
    // Retain prefix multipliers and clear only rows/variables whose seed changed.
    for(int k=start;k<g::N;++k)dual_.segment(g::stage_offsets[k],g::stage_offsets[k+1]-g::stage_offsets[k]).setZero();
    for(int k=start;k<restitution_cap_nodes;++k)dual_[g::CONSTRAINTS+k]=0.;
    dual_.segment(m_+start*g::NZ,g::NZ*(g::N+1-start)).setZero();
    accepted=true;outcome="full_merit_improved_arm_defects_preserved";log();
  }catch(const std::exception&){outcome="candidate_numerical_or_contact_rejection";log();}
}
#endif
void Controller::initialize(const State& x,const Joint& torque){
#ifdef JUGGLING_BOUNDED_MULTISTART_MODE
  multistart_pending_=false;
#endif
#ifdef JUGGLING_FIRST_EVENT_CENTER_WEIGHT
  if(JUGGLING_FIRST_EVENT_CENTER_WEIGHT>0.){const Eigen::Vector3d preferred=first_event_center_.preferred_local;
    first_event_center_=FirstEventCenter{};first_event_center_.preferred_local=preferred;}
#endif
#ifdef JUGGLING_TAIL_ENDPOINT_START_S
  tail_endpoint_elapsed_=0.;
#endif
#ifdef JUGGLING_SOLVER_WARMSTART_SEED_RADIUS
  seed_controller_time_=0.;
#endif
  Matrix Y=Matrix::Zero(g::NZ,g::N+1);State state=x;for(int k=0;k<=g::N;++k)Y(g::NX+9,k)=g::stage_steps[std::min(k,g::N-1)];Y.row(g::NX+8).setConstant(.5);
  for(int k=0;k<g::N;++k){Y.col(k).head<g::NX>()=state;
#ifdef JUGGLING_KINEMATIC_DYNAMIC_PREFIX
    if(acceleration_node(k))Y.col(k).segment<7>(g::NX).setZero();
    else
#endif
    Y.col(k).segment<7>(g::NX)=robot.rnea(state.segment<7>(6),state.segment<7>(13),Joint::Zero());
    try {auto out=rollout(state,g::stage_steps[k],Joint::Zero(),.003,nullptr,nullptr,k);state=out.first;
      if(!out.second.empty()){Y(g::NX+7,k)=out.second[0].impulse/g::mass;Y(g::NX+8,k)=out.second[0].time/g::stage_steps[k];}}
    catch(const std::runtime_error& e){if(std::string(e.what())!="Unsupported resting contact")throw;
      // Low-restitution stationary-paddle seed reaches Zeno settling inside
      // the horizon. Seed a supported ball; NEVER used by the physical plant.
      auto f=prediction_contact_fk(state.segment<7>(6));
#if defined(JUGGLING_RAMP_PREDICTION) && JUGGLING_RAMP_PREDICTION
      prediction_plane(f);
#endif
      Eigen::Vector3d n=f[1];
      Eigen::Vector3d p=f[0].col(0)-map(g::origin),rel=state.head<3>()-p;
      state.head<3>()+=n*(g::radius-n.dot(rel));state.segment<3>(3)=f[2]*state.segment<7>(13);
      Y(g::NX+7,k)=-ball_gravity[2]*g::stage_steps[k];Y(g::NX+8,k)=.5;
    }}
  Y.col(g::N).head<g::NX>()=state;
#ifdef JUGGLING_KINEMATIC_DYNAMIC_PREFIX
  Y.col(g::N).segment<7>(g::NX).setZero();
#else
  Y.col(g::N).segment<7>(g::NX)=robot.rnea(state.segment<7>(6),state.segment<7>(13),Joint::Zero());
#endif
  Y.col(0).segment<7>(g::NX)=torque;
  for(int k=0;k<=g::N;++k){Y(g::NX+10,k)=g::stage_times[k];z_.segment<g::NZ>(k*g::NZ)=(Y.col(k)-map(g::offset)).cwiseQuotient(map(g::scale));}
  initialized_=true;
#ifdef JUGGLING_FIRST_EVENT_CENTER_WEIGHT
  if(JUGGLING_FIRST_EVENT_CENTER_WEIGHT>0.)freeze_first_event_center(x,0.);
#endif
}
#ifdef JUGGLING_KINEMATIC_DYNAMIC_PREFIX
void Controller::reconcile_kinematic_tail(Matrix& Y){
#ifdef JUGGLING_KINEMATIC_ENDPOINT_ACCELERATIONS
  // Endpoint accelerations are the control proposals. Preserve them while
  // rebuilding tail states with the same trapezoidal velocity equation as
  // the factors. Only the left boundary acceleration uses rigid-body dynamics.
  const Joint q=Y.col(kinematic_prefix).segment<7>(6),dq=Y.col(kinematic_prefix).segment<7>(13);
  Joint a0=robot.mass(q).ldlt().solve(Y.col(kinematic_prefix).segment<7>(g::NX)-robot.bias(q,dq));
  for(int k=kinematic_prefix;k<g::N;++k){
    const Joint a1=Y.col(k+1).segment<7>(g::NX),average=.5*(a0+a1);
    if(!average.allFinite())throw std::runtime_error("Nonfinite kinematic-tail endpoint acceleration");
    const double h=g::stage_steps[k];
    Y.col(k+1).segment<7>(6)=Y.col(k).segment<7>(6)+h*Y.col(k).segment<7>(13)+(.5*h*h)*average;
    Y.col(k+1).segment<7>(13)=Y.col(k).segment<7>(13)+h*average;
    a0=a1;
  }
#else
  // Preserve the already shifted/feedback-corrected velocity differences as
  // tail acceleration proposals BEFORE advancing any endpoint in place.
  Matrix accelerations(7,g::N-kinematic_prefix);
  for(int k=kinematic_prefix;k<g::N;++k)
    accelerations.col(k-kinematic_prefix)=(Y.col(k+1).segment<7>(13)-Y.col(k).segment<7>(13))/g::stage_steps[k];
  const Joint q=Y.col(kinematic_prefix).segment<7>(6),dq=Y.col(kinematic_prefix).segment<7>(13);
  // Exactly one boundary forward-dynamics conversion. Tail slots themselves
  // are never read as torques, and no later tail interval evaluates M/RNEA.
  accelerations.col(0)=robot.mass(q).ldlt().solve(Y.col(kinematic_prefix).segment<7>(g::NX)-robot.bias(q,dq));
  if(!accelerations.allFinite())throw std::runtime_error("Nonfinite kinematic-tail seed acceleration");
  for(int k=kinematic_prefix;k<g::N;++k){
    const double h=g::stage_steps[k];const Joint a=accelerations.col(k-kinematic_prefix);
    Y.col(k+1).segment<7>(6)=Y.col(k).segment<7>(6)+h*Y.col(k).segment<7>(13)+(.5*h*h)*a;
    Y.col(k+1).segment<7>(13)=Y.col(k).segment<7>(13)+h*a;
    Y.col(k+1).segment<7>(g::NX)=a;
  }
#endif
}
#endif
// Decode only. In a kinematic profile the tail control slots deliberately
// remain accelerations; metadata, not a misleading torque conversion, labels
// those slots. Prefixes P1/P2 need the offline copy-only execution adapter;
// with P>=3 the first four executed nodes remain ordinary torque controls.
void Controller::set_physical_plan(){plan_.resize(g::NZ,g::N+1);for(int k=0;k<=g::N;++k)plan_.col(k)=z_.segment<g::NZ>(k*g::NZ).cwiseProduct(map(g::scale))+map(g::offset);}
void Controller::shift(double dt,const State& x,const Joint& torque){
#ifdef JUGGLING_BOUNDED_MULTISTART_MODE
  multistart_pending_=false;
#endif
  if(!initialized_)throw std::runtime_error("Cannot shift an uninitialized controller");
#ifdef JUGGLING_KINEMATIC_DYNAMIC_PREFIX
  if(!std::isfinite(dt)||dt<0.)throw std::runtime_error("Kinematic-tail shift requires finite nonnegative elapsed time");
#endif
#ifdef JUGGLING_TAIL_ENDPOINT_START_S
  if(!tail_endpoint_penalties_.empty())tail_endpoint_elapsed_+=dt;
#endif
#ifdef JUGGLING_SOLVER_WARMSTART_SEED_RADIUS
  seed_controller_time_+=dt;
#endif
  set_physical_plan();Matrix old=plan_,Y=old;Y.row(g::NX+7).setZero();Y.row(g::NX+8).setConstant(.5);Eigen::Vector3d gravity=Eigen::Map<const Eigen::Vector3d>(ball_gravity.data());
#if defined(JUGGLING_RAMP_PREDICTION) && JUGGLING_RAMP_PREDICTION
  for(int k=0;k<=g::N;++k){double t=g::stage_times[k]+dt;int i=source(t);Row row=old.col(i);double h=g::stage_steps[std::min(i,g::N-1)],s=t-g::stage_times[i];
    Joint q0=row.segment<7>(6),v0=row.segment<7>(13);
    Joint acc=robot.mass(q0).ldlt().solve(row.segment<7>(g::NX)-robot.bias(q0,v0)),jerk=Joint::Zero();
    if(i<g::N){Joint q1=old.col(i+1).segment<7>(6),v1=old.col(i+1).segment<7>(13);
      Joint end_acc=robot.mass(q1).ldlt().solve(old.col(i+1).segment<7>(g::NX)-robot.bias(q1,v1));jerk=(end_acc-acc)/h;}
    State shifted=free_ramp(row.head<g::NX>(),s,acc,jerk);Y.col(k).head<6>()=shifted.head<6>();
    Y.col(k).segment<7>(6)=shifted.segment<7>(6);Y.col(k).segment<7>(13)=shifted.segment<7>(13);
    Y.col(k).segment<3>(20)=row.segment<3>(20);
    if(i<g::N){double event=row[g::NX+8]*h;
      if(s>=event-1e-12){Eigen::Vector3d impulse=old.col(i+1).segment<3>(3)-free_ramp(row.head<g::NX>(),h,acc,jerk).segment<3>(3);
#ifdef JUGGLING_BALL_DRAG
        impulse*=std::exp(JUGGLING_BALL_DRAG*(h-event));
#endif
        State hit=free_ramp(row.head<g::NX>(),event,acc,jerk);hit.segment<3>(3)+=impulse;
        State after=free_ramp(hit,s-event,acc+event*jerk,jerk);
        Y.col(k).head<6>()=after.head<6>();Y.col(k).segment<3>(20)=old.col(i+1).segment<3>(20);}}
    // Preserve the shifted acceleration arc, rather than interpolating torque
    // through a changing inertia/bias and thereby changing the ramp itself.
    Y.col(k).segment<7>(g::NX)=robot.rnea(Y.col(k).segment<7>(6),Y.col(k).segment<7>(13),acc+s*jerk);
  }
#else
  for(int k=0;k<=g::N;++k){double t=g::stage_times[k]+dt;int i=source(t);Row row=old.col(i);double h=g::stage_steps[std::min(i,g::N-1)],s=t-g::stage_times[i];Joint acc=(old.col(std::min(i+1,g::N)).segment<7>(13)-row.segment<7>(13))/h;
#ifdef JUGGLING_KINEMATIC_DYNAMIC_PREFIX
    // Continue the last acceleration beyond the finite horizon. For interior
    // intervals retain the same q/dq interpolation used by contact factors.
    if(i==g::N)acc=row.segment<7>(g::NX);
#endif
    State shifted=free(row.head<g::NX>(),s,acc);Y.col(k).head<6>()=shifted.head<6>();
    Y.col(k).segment<7>(6)=row.segment<7>(6)+s*row.segment<7>(13)+.5*s*s*acc;Y.col(k).segment<7>(13)=row.segment<7>(13)+s*acc;
    Y.col(k).segment<3>(20)=row.segment<3>(20);
    if(i<g::N){double event=row[g::NX+8]*h;
      if(s>=event-1e-12){Eigen::Vector3d impulse=old.col(i+1).segment<3>(3)-free(row.head<g::NX>(),h,acc).segment<3>(3);
#ifdef JUGGLING_BALL_DRAG
        impulse*=std::exp(JUGGLING_BALL_DRAG*(h-event));
#endif
        State hit=free(row.head<g::NX>(),event,acc);hit.segment<3>(3)+=impulse;State after=free(hit,s-event,acc);
        Y.col(k).head<6>()=after.head<6>();Y.col(k).segment<3>(20)=old.col(i+1).segment<3>(20);}}
#ifdef JUGGLING_KINEMATIC_DYNAMIC_PREFIX
#ifdef JUGGLING_KINEMATIC_ENDPOINT_ACCELERATIONS
    if(kinematic_stage(i)){
      Joint a0=row.segment<7>(g::NX);
      if(!acceleration_node(i))a0=robot.mass(row.segment<7>(6)).ldlt().solve(
        row.segment<7>(g::NX)-robot.bias(row.segment<7>(6),row.segment<7>(13)));
      const Joint endpoint_acc=i==g::N?a0:((1-s/h)*a0+(s/h)*old.col(i+1).segment<7>(g::NX)).eval();
      if(acceleration_node(k))Y.col(k).segment<7>(g::NX)=endpoint_acc;
      else Y.col(k).segment<7>(g::NX)=robot.rnea(Y.col(k).segment<7>(6),Y.col(k).segment<7>(13),endpoint_acc);
    }else
#else
    if(acceleration_node(k))Y.col(k).segment<7>(g::NX)=acc;
    else if(kinematic_stage(i))Y.col(k).segment<7>(g::NX)=robot.rnea(Y.col(k).segment<7>(6),Y.col(k).segment<7>(13),acc);
    else
#endif
#endif
    Y.col(k).segment<7>(g::NX)=i==g::N?old.col(g::N).segment<7>(g::NX).eval():((1-s/h)*row.segment<7>(g::NX)+(s/h)*old.col(i+1).segment<7>(g::NX)).eval();
  }
#endif
  std::array<double,g::N> strongest_impulse{};
  for(int i=0;i<g::N;++i){double relative=g::stage_times[i]+old(g::NX+8,i)*g::stage_steps[i]-dt;int k=source(relative);
    if(relative>=-1e-12&&k<g::N){double impulse=old(g::NX+7,i);Y(g::NX+7,k)+=impulse;
      // A zero/tiny numerical impulse must not overwrite a real event time.
      // Preserve the accumulated impulse and time of its dominant source.
      if(std::abs(impulse)>strongest_impulse[k]){strongest_impulse[k]=std::abs(impulse);
        Y(g::NX+8,k)=std::clamp((relative-g::stage_times[k])/g::stage_steps[k],0.,1.);}}}
#ifdef JUGGLING_WARMSTART_FEEDBACK_TIME
  if(JUGGLING_WARMSTART_FEEDBACK_TIME>0){
    // Anchor the whole joint seed to readback, preserving smooth velocity
    // evolution. Replacing node zero alone turns servo lag into a spurious
    // acceleration jump across the first shooting interval.
    const double feedback_time=JUGGLING_WARMSTART_FEEDBACK_TIME;
    Joint offset=x.segment<7>(6)-Y.col(0).segment<7>(6);
    Joint delta_v=x.segment<7>(13)-Y.col(0).segment<7>(13);
    for(int k=0;k<=g::N;++k){
      double decay=std::exp(-g::stage_times[k]/feedback_time);
      Joint q=Y.col(k).segment<7>(6),dq=Y.col(k).segment<7>(13);
#ifdef JUGGLING_KINEMATIC_DYNAMIC_PREFIX
      Joint a=Joint::Zero();
      if(!acceleration_node(k))a=robot.mass(q).ldlt().solve(Y.col(k).segment<7>(g::NX)-robot.bias(q,dq));
#ifdef JUGGLING_KINEMATIC_ENDPOINT_ACCELERATIONS
      else a=Y.col(k).segment<7>(g::NX);
#endif
#else
      Joint a=robot.mass(q).ldlt().solve(Y.col(k).segment<7>(g::NX)-robot.bias(q,dq));
#endif
      Y.col(k).segment<7>(6)+=offset;Y.col(k).segment<7>(13)+=decay*delta_v;
#ifdef JUGGLING_KINEMATIC_DYNAMIC_PREFIX
#ifdef JUGGLING_KINEMATIC_ENDPOINT_ACCELERATIONS
      if(acceleration_node(k))Y.col(k).segment<7>(g::NX)=a-decay/feedback_time*delta_v;
      else
#else
      if(!acceleration_node(k))
#endif
#endif
      Y.col(k).segment<7>(g::NX)=robot.rnea(Y.col(k).segment<7>(6),Y.col(k).segment<7>(13),a-decay/feedback_time*delta_v);
      if(k<g::N)offset+=.5*g::stage_steps[k]*(decay+std::exp(-g::stage_times[k+1]/feedback_time))*delta_v;
    }
  }
#endif
  Y.col(0).head<g::NX>()=x;Y.col(0).segment<7>(g::NX)=torque;
#ifdef JUGGLING_KINEMATIC_DYNAMIC_PREFIX
  reconcile_kinematic_tail(Y);
#endif
  Vector previous=dual_;dual_.setZero();
  for(int k=0;k<=g::N;++k){int i=source(g::stage_times[k]+dt);
    if(i<g::N){
      if(k<g::N
#ifdef JUGGLING_KINEMATIC_DYNAMIC_PREFIX
         // Boundary dynamics/endpoint/control meanings differ across the
         // switch; its old conversion equality or new rate inequality does
         // not inherit multipliers from an ordinary stage.
         &&k!=kinematic_prefix&&i!=kinematic_prefix&&kinematic_stage(k)==kinematic_stage(i)
#endif
      ){int cursor=g::stage_offsets[i];for(int row=g::stage_offsets[k];row<g::stage_offsets[k+1];++row){
        int id=g::constraint_ids[row];while(cursor<g::stage_offsets[i+1]&&g::constraint_ids[cursor]<id)++cursor;
        if(cursor<g::stage_offsets[i+1]&&g::constraint_ids[cursor]==id&&!(id>=g::contact_begin&&id<g::contact_end))dual_[row]=previous[cursor];}}
      dual_.segment<g::NZ>(m_+k*g::NZ)=previous.segment<g::NZ>(m_+i*g::NZ);
#ifdef JUGGLING_KINEMATIC_DYNAMIC_PREFIX
      if(acceleration_node(k)!=acceleration_node(i))dual_.segment<7>(m_+k*g::NZ+g::NX).setZero();
#endif
    }}
  dual_.segment(m_,g::NP).setZero();
  for(int k=0;k<=g::N;++k){Y(g::NX+9,k)=g::stage_steps[std::min(k,g::N-1)];Y(g::NX+10,k)=g::stage_times[k];z_.segment<g::NZ>(k*g::NZ)=(Y.col(k)-map(g::offset)).cwiseQuotient(map(g::scale));}
#ifdef JUGGLING_FIRST_EVENT_CENTER_WEIGHT
  // Freeze against the prior shifted seed before either regenerated candidate
  // is assessed. Warmstart repair, arm repair and every inner SQP/line-search
  // therefore see exactly the same auxiliary cost and event identity.
  if(JUGGLING_FIRST_EVENT_CENTER_WEIGHT>0.)freeze_first_event_center(x,dt);
#endif
  // Regenerate ball/contact seeds on the shifted joint arcs. On failure retain
  // the already shifted primal/dual, exactly as in the Python controller.
  State state=x;fallback_stage_=-1;
#ifdef JUGGLING_SOLVER_WARMSTART_SEED_RADIUS
  SeedContext seed;seed.radius=JUGGLING_SOLVER_WARMSTART_SEED_RADIUS;
#ifdef JUGGLING_SOLVER_WARMSTART_SEED_LEAD
  seed.minimum_lead=JUGGLING_SOLVER_WARMSTART_SEED_LEAD;
#endif
#ifdef JUGGLING_SOLVER_WARMSTART_SEED_MAX_CONTACTS
  seed.max_contacts=static_cast<int>(JUGGLING_SOLVER_WARMSTART_SEED_MAX_CONTACTS);
#endif
  SeedContext* seed_context=seed.radius>0.?&seed:nullptr;
  // This shared context limits the total injected modes across the horizon.
  // Legacy failure fallback can retain old, infeasible shifted predictions;
  // neither those predictions nor an injected event certify physical contact.
#endif
#ifdef JUGGLING_SOLVER_WARMSTART_REPAIR
  const double warmstart_repair=JUGGLING_SOLVER_WARMSTART_REPAIR;
  if(warmstart_repair!=0.&&warmstart_repair!=1.&&warmstart_repair!=2.&&warmstart_repair!=3.)throw std::runtime_error("Warmstart repair must be 0, 1, 2 or 3");
#endif
  for(int k=0;k<g::N;++k){Y.col(k).head<6>()=state.head<6>();Y.col(k).segment<3>(20)=state.segment<3>(20);Y(g::NX+7,k)=0;Y(g::NX+8,k)=.5;
#ifdef JUGGLING_SOLVER_WARMSTART_SEED_RADIUS
    seed.stage=k;seed.stage_start=g::stage_times[k];
#endif
#if defined(JUGGLING_RAMP_PREDICTION) && JUGGLING_RAMP_PREDICTION
    Joint q0=Y.col(k).segment<7>(6),v0=Y.col(k).segment<7>(13),q1=Y.col(k+1).segment<7>(6),v1=Y.col(k+1).segment<7>(13);
    Joint acc=robot.mass(q0).ldlt().solve(Y.col(k).segment<7>(g::NX)-robot.bias(q0,v0));
    Joint end_acc=robot.mass(q1).ldlt().solve(Y.col(k+1).segment<7>(g::NX)-robot.bias(q1,v1));
#ifdef JUGGLING_SOLVER_WARMSTART_SEED_RADIUS
    try{auto out=rollout(Y.col(k).head<g::NX>(),g::stage_steps[k],acc,.01,&end_acc,seed_context,k);state=out.first;
#else
    try{auto out=rollout(Y.col(k).head<g::NX>(),g::stage_steps[k],acc,.01,&end_acc,nullptr,k);state=out.first;
#endif
#else
    Joint acc=(Y.col(k+1).segment<7>(13)-Y.col(k).segment<7>(13))/g::stage_steps[k];
#ifdef JUGGLING_SOLVER_WARMSTART_SEED_RADIUS
    try{auto out=rollout(Y.col(k).head<g::NX>(),g::stage_steps[k],acc,.01,nullptr,seed_context,k);state=out.first;
#else
    try{auto out=rollout(Y.col(k).head<g::NX>(),g::stage_steps[k],acc,.01,nullptr,nullptr,k);state=out.first;
#endif
#endif
      if(guard_ball_rollout&&(state[2]<-.15||state[2]>1.5)){
#ifdef JUGGLING_SOLVER_WARMSTART_REPAIR
        // An alternate seed may violate bounds; the NLP still enforces exactly
        // the same bounds. Finish its full rollout to avoid a discontinuity
        // between a repaired prefix and an unrelated saved tail.
        if(warmstart_repair==1.){if(fallback_stage_<0)fallback_stage_=k;}
        else
#endif
        throw std::runtime_error("Ball rollout left optimizer workspace");
      }
      if(!out.second.empty()){Y(g::NX+7,k)=out.second[0].impulse/g::mass;Y(g::NX+8,k)=out.second[0].time/g::stage_steps[k];}}
    catch(const std::runtime_error& e){std::string message=e.what();if(message!="Unsupported resting contact"&&message!="Ball rollout left optimizer workspace")throw;
#ifdef JUGGLING_SOLVER_WARMSTART_REPAIR
      if(fallback_stage_<0)fallback_stage_=k;
#else
      fallback_stage_=k;
#endif
#ifdef JUGGLING_SOLVER_WARMSTART_SEED_RADIUS
      trace_virtual_seed(seed,false,"regeneration_failed_legacy_seed_retained");
#endif
#ifdef JUGGLING_SOLVER_WARMSTART_ARM_SHIFT
      try_arm_shift_seed(x,torque,seed);
#endif
      return;}}
  Y.col(g::N).head<6>()=state.head<6>();Y.col(g::N).segment<3>(20)=state.segment<3>(20);
#ifdef JUGGLING_SOLVER_WARMSTART_REPAIR
  if(warmstart_repair==1.&&fallback_stage_>=0){
#ifdef JUGGLING_SOLVER_WARMSTART_SEED_RADIUS
    bool repaired=false;
#endif
    Vector regenerated(n_);for(int k=0;k<=g::N;++k)regenerated.segment<g::NZ>(k*g::NZ)=(Y.col(k)-map(g::offset)).cwiseQuotient(map(g::scale));
    // Retain the legacy shifted seed unless the fully regenerated alternative
    // is finite, inside the existing divergence envelope, and improves the
    // exact same objective-plus-violation merit used by the SQP line search.
    if(regenerated.allFinite()&&(lower_-regenerated).cwiseMax(0).maxCoeff()<=100
       &&(regenerated-upper_).cwiseMax(0).maxCoeff()<=100){
      Eigen::Matrix<double,30,1> parameter;parameter<<x,torque;
#ifdef JUGGLING_SOLVER_MERIT_MULTIPLIER
      const double penalty=JUGGLING_SOLVER_MERIT_MULTIPLIER;
      if(!std::isfinite(penalty)||penalty<=0.)throw std::runtime_error("Invalid SQP merit multiplier");
#else
      constexpr double penalty=1000.;
#endif
      double old_cost=value(z_,parameter,trial_constraints_);
      double old_merit=old_cost+penalty*g::objective_scale*violation_sum(trial_constraints_,z_);
      double new_cost=value(regenerated,parameter,trial_constraints_);
      double new_merit=new_cost+penalty*g::objective_scale*violation_sum(trial_constraints_,regenerated);
      if(std::isfinite(old_merit)&&std::isfinite(new_merit)&&new_merit<=old_merit+1e-8){z_=regenerated;
#ifdef JUGGLING_SOLVER_WARMSTART_SEED_RADIUS
        repaired=true;
#endif
      }
    }
#ifdef JUGGLING_SOLVER_WARMSTART_SEED_RADIUS
    trace_virtual_seed(seed,repaired,repaired?"repair_merit_accepted":"repair_merit_rejected_legacy_seed_retained");
#endif
#ifdef JUGGLING_SOLVER_WARMSTART_ARM_SHIFT
    try_arm_shift_seed(x,torque,seed);
#endif
    return;
  }
#endif
#ifdef JUGGLING_BOUNDED_MULTISTART_MODE
  if(multistart_mode_>0&&seed.contacts>0&&seed.events[0].contact_ordinal==1){
    multistart_shifted_=z_;multistart_dual_=dual_;multistart_regenerated_.resize(n_);
    for(int k=0;k<=g::N;++k)multistart_regenerated_.segment<g::NZ>(k*g::NZ)=(Y.col(k)-map(g::offset)).cwiseQuotient(map(g::scale));
    multistart_seed_=seed;multistart_pending_=true;
    // Leave the legacy regenerated primal visible after shift(). step() alone
    // consumes the saved pair; no candidate is pre-screened by raw merit.
    z_=multistart_regenerated_;trace_virtual_seed(seed,true,"multistart_candidates_ready");return;
  }
#endif
#if defined(JUGGLING_SOLVER_WARMSTART_REPAIR) && defined(JUGGLING_SOLVER_WARMSTART_SEED_RADIUS)
  // Mode 2 screens every SUCCESSFUL regeneration with off-face contacts.
  // Mode 3 screens only when the candidate's FIRST upcoming contact is
  // virtual: new later tail contacts must still be able to enter the horizon.
  // Mode 0 and the mode-1 failed-rollout repair remain unchanged.
  // z_ is still the shifted seed pinned to this measured state/torque; all
  // optional event/landing costs were frozen before regenerating Y.
  if(seed.contacts>0&&(warmstart_repair==2.||(warmstart_repair==3.&&seed.events[0].contact_ordinal==1))){
    SeedMeritScreen screen;screen.mode=static_cast<int>(warmstart_repair);
#ifdef JUGGLING_SOLVER_MERIT_MULTIPLIER
    screen.penalty=JUGGLING_SOLVER_MERIT_MULTIPLIER;
#else
    screen.penalty=1000.;
#endif
    if(!std::isfinite(screen.penalty)||screen.penalty<=0.)throw std::runtime_error("Invalid SQP merit multiplier");
    Vector regenerated(n_);for(int k=0;k<=g::N;++k)regenerated.segment<g::NZ>(k*g::NZ)=(Y.col(k)-map(g::offset)).cwiseQuotient(map(g::scale));
    Eigen::Matrix<double,30,1> parameter;parameter<<x,torque;
    auto measure=[&](const Vector& candidate,SeedMeritScreen::Point& point){
      point.cost=value(candidate,parameter,trial_constraints_);point.evaluated=true;
      // Check constraints explicitly: reduction of NaNs is not a reliable
      // rejection test. Keep diagnostic numbers null when evaluation failed.
      if(!std::isfinite(point.cost)||!trial_constraints_.allFinite()){
        point.merit=point.constraint_l1=point.constraint_max=point.bound_l1=point.bound_max=std::numeric_limits<double>::quiet_NaN();return false;}
      const Vector nonlinear_low=(cl_-trial_constraints_).cwiseMax(0),nonlinear_high=(trial_constraints_-cu_).cwiseMax(0);
      const Vector bound_low=(lower_-candidate).cwiseMax(0),bound_high=(candidate-upper_).cwiseMax(0);
      point.constraint_l1=nonlinear_low.sum()+nonlinear_high.sum();
      point.constraint_max=std::max(nonlinear_low.maxCoeff(),nonlinear_high.maxCoeff());
      point.bound_l1=bound_low.sum()+bound_high.sum();point.bound_max=std::max(bound_low.maxCoeff(),bound_high.maxCoeff());
      point.merit=point.cost+screen.penalty*g::objective_scale*violation_sum(trial_constraints_,candidate);
      return std::isfinite(point.merit);
    };
    bool accepted=false;const char* outcome="virtual_seed_merit_nonfinite_or_outside_model_range";
    if(z_.allFinite()&&regenerated.allFinite()&&(lower_-regenerated).cwiseMax(0).maxCoeff()<=100
       &&(regenerated-upper_).cwiseMax(0).maxCoeff()<=100){
      try{
        const bool old_finite=measure(z_,screen.shifted),new_finite=measure(regenerated,screen.regenerated);
        if(old_finite&&new_finite){
          accepted=screen.regenerated.merit<screen.shifted.merit-1e-10;
          outcome=accepted?"virtual_seed_merit_accepted":"virtual_seed_merit_rejected_shifted_seed_retained";
        }
      }catch(const std::exception&){outcome="virtual_seed_merit_evaluation_failed_shifted_seed_retained";}
    }
    if(accepted)z_=regenerated;
    trace_virtual_seed(seed,accepted,outcome,&screen);
#ifdef JUGGLING_SOLVER_WARMSTART_ARM_SHIFT
    try_arm_shift_seed(x,torque,seed);
#endif
    return;
  }
#endif
  for(int k=0;k<=g::N;++k)z_.segment<g::NZ>(k*g::NZ)=(Y.col(k)-map(g::offset)).cwiseQuotient(map(g::scale));
#ifdef JUGGLING_SOLVER_WARMSTART_SEED_RADIUS
#ifdef JUGGLING_SOLVER_WARMSTART_REPAIR
  if(warmstart_repair==3.&&seed.contacts>0&&seed.events[0].contact_ordinal>1)
    trace_virtual_seed(seed,true,"later_virtual_contact_merit_bypassed");
  else
#endif
  trace_virtual_seed(seed,true,"regenerated_seed_applied");
#endif
#ifdef JUGGLING_SOLVER_WARMSTART_ARM_SHIFT
  try_arm_shift_seed(x,torque,seed);
#endif
}
void Controller::set_warmstart(const double* z,const double* dual){z_=Eigen::Map<const Vector>(z,n_);dual_=Eigen::Map<const Vector>(dual,m_+n_);initialized_=true;
#ifdef JUGGLING_BOUNDED_MULTISTART_MODE
  multistart_pending_=false;
#endif
#ifdef JUGGLING_FIRST_EVENT_CENTER_WEIGHT
  if(JUGGLING_FIRST_EVENT_CENTER_WEIGHT>0.){first_event_center_.active=false;first_event_center_.have_observation=false;}
#endif
}
double Controller::violation_sum(const Vector& c,const Vector& z) const{return (cl_-c).cwiseMax(0).sum()+(c-cu_).cwiseMax(0).sum()+(lower_-z).cwiseMax(0).sum()+(z-upper_).cwiseMax(0).sum();}
#ifdef JUGGLING_BOUNDED_MULTISTART_MODE
void Controller::configure_multistart(){
  const double mode=JUGGLING_BOUNDED_MULTISTART_MODE;
  if(mode!=0.&&mode!=1.&&mode!=2.)throw std::runtime_error("Bounded multistart mode must be 0, 1 or 2");
  multistart_mode_=static_cast<int>(mode);
  if(multistart_mode_==0)return; // No allocation, seed capture or solver reset.
#if !defined(JUGGLING_PARAMETRIC_FACTORS) || !defined(JUGGLING_SOLVER_WARMSTART_SEED_RADIUS) || !defined(JUGGLING_SOLVER_WARMSTART_REPAIR) || !defined(JUGGLING_SOLVER_WARMSTART_ARM_SHIFT) || !defined(JUGGLING_SOLVER_ADAPTIVE_RHO) || !defined(JUGGLING_SOLVER_CONTACT_RHO_MULTIPLIER)
#error Bounded multistart requires parametric experimental solver options
#endif
  if(g::factor_parameters.size()!=35||g::sqp_iters!=4.||!std::isfinite(g::osqp_iters)
     ||g::osqp_iters<1.||g::osqp_iters>100.||g::osqp_iters!=std::floor(g::osqp_iters))
    throw std::runtime_error("Bounded multistart requires legacy factors, 4 SQP and 1..100 OSQP iterations");
  if(JUGGLING_SOLVER_WARMSTART_SEED_RADIUS<=0.||JUGGLING_SOLVER_WARMSTART_REPAIR!=0.
     ||JUGGLING_SOLVER_WARMSTART_ARM_SHIFT!=0.||JUGGLING_SOLVER_ADAPTIVE_RHO!=0.
     ||JUGGLING_SOLVER_CONTACT_RHO_MULTIPLIER!=1.)
    throw std::runtime_error("Bounded multistart requires virtual seeds, repair/arm_shift/adaptive_rho=0 and contact_rho_multiplier=1");
  // Validation instrumentation only; no runtime policy scalar or C API.
  const char* verify=std::getenv("JUGGLING_MULTISTART_VERIFY");
  if(verify&&*verify){const std::string option(verify);
    if(option!="0"&&option!="1"&&option!="2")throw std::runtime_error("JUGGLING_MULTISTART_VERIFY must be 0, 1 or 2");
    multistart_verify_=option=="2"?2:option=="1"?1:0;
  }
  const char* path=std::getenv("JUGGLING_MULTISTART_TRACE");
  if(multistart_verify_&&(!path||!*path))throw std::runtime_error("Multistart verification requires a trace path");
  if(path&&*path){
    multistart_trace_=std::make_unique<std::ofstream>(path,std::ios::out|std::ios::app);
    if(!*multistart_trace_)throw std::runtime_error("Cannot open multistart trace");
    multistart_trace_->precision(17);
  }
}
void Controller::solve_multistart(const Eigen::Matrix<double,30,1>& p,StepInfo& info){
  // The caller froze measured state, references, bias and optional guidance
  // exactly once. No initialize/shift/set_warmstart call is allowed here.
  if(!multistart_pending_||multistart_mode_==0)throw std::runtime_error("Missing multistart seeds");
  multistart_pending_=false;
  struct Workspace {
    OSQPSolver* solver=nullptr;OSQPCscMatrix* h=nullptr;OSQPCscMatrix* a=nullptr;
    ~Workspace(){if(solver)osqp_cleanup(solver);if(h)OSQPCscMatrix_free(h);if(a)OSQPCscMatrix_free(a);}
  };
  auto detach=[&](){auto owner=std::make_unique<Workspace>();owner->solver=std::exchange(solver_,nullptr);
    owner->h=std::exchange(osqp_h_,nullptr);owner->a=std::exchange(osqp_a_,nullptr);return owner;};
  auto attach=[&](std::unique_ptr<Workspace>& owner){
    if(solver_||osqp_h_||osqp_a_||!owner)throw std::runtime_error("Invalid multistart workspace ownership");
    solver_=std::exchange(owner->solver,nullptr);osqp_h_=std::exchange(owner->h,nullptr);osqp_a_=std::exchange(owner->a,nullptr);
  };
  // Each setup owns copies of QP data; both branches may evaluate serially in
  // the shared scratch arrays. Neither branch inherits the other's ADMM state.
  auto previous_workspace=detach();const Vector previous_z=z_,previous_dual=dual_;
  struct Outcome {
    Vector primal,dual,constraints;
    std::unique_ptr<Workspace> workspace;
    StepInfo info;
    bool evaluated=false,valid=false,exception=false;
    std::string error;
    int qp_calls=0;
    double initial_cost=std::numeric_limits<double>::quiet_NaN();
    double initial_merit=std::numeric_limits<double>::quiet_NaN();
    double final_merit=std::numeric_limits<double>::quiet_NaN();
    double constraint_l1=std::numeric_limits<double>::quiet_NaN();
    double bound_l1=std::numeric_limits<double>::quiet_NaN();
    double wall_ms=0.;
  };
  std::array<Outcome,2> branches;
  const double penalty=JUGGLING_SOLVER_MERIT_MULTIPLIER;
  if(!std::isfinite(penalty)||penalty<=0.){attach(previous_workspace);throw std::runtime_error("Invalid SQP merit multiplier");}
  auto run=[&](int branch,const Vector& seed,int count){
    Outcome& out=branches[branch];out.evaluated=true;out.info.evaluator_threads=evaluator_threads_;
    out.constraints.resize(m_); // value() writes fixed segments; it does not resize its destination.
    const auto started=Clock::now();z_=seed;dual_=multistart_dual_;
    try{
      if(!z_.allFinite()||!dual_.allFinite()||(lower_-z_).cwiseMax(0).maxCoeff()>100.
         ||(z_-upper_).cwiseMax(0).maxCoeff()>100.)throw std::runtime_error("Multistart seed outside model range");
      auto tic=Clock::now();out.initial_cost=value(z_,p,out.constraints);
      out.initial_merit=out.initial_cost+penalty*g::objective_scale*violation_sum(out.constraints,z_);
      out.info.merit_ms+=ms(tic);
      if(!out.constraints.allFinite()||!std::isfinite(out.initial_cost)||!std::isfinite(out.initial_merit))
        throw std::runtime_error("Nonfinite multistart seed merit");
      // solve_sqp owns its retry variables locally. Its first call sees a null
      // solver, then its second iteration may continue only this branch's QP.
      solve_sqp(p,count,out.info,&out.qp_calls);
      tic=Clock::now();out.info.cost=value(z_,p,out.constraints);
      const Vector nl=(cl_-out.constraints).cwiseMax(0),nh=(out.constraints-cu_).cwiseMax(0);
      const Vector bl=(lower_-z_).cwiseMax(0),bh=(z_-upper_).cwiseMax(0);
      out.constraint_l1=nl.sum()+nh.sum();out.bound_l1=bl.sum()+bh.sum();
      out.info.violation=std::max(nl.maxCoeff(),nh.maxCoeff());
      out.info.bound_violation=std::max(bl.maxCoeff(),bh.maxCoeff());
      out.final_merit=out.info.cost+penalty*g::objective_scale*violation_sum(out.constraints,z_);
      out.valid=z_.allFinite()&&dual_.allFinite()&&out.constraints.allFinite()
        &&std::isfinite(out.info.cost)&&std::isfinite(out.final_merit);
      out.info.merit_ms+=ms(tic);
    }catch(const std::exception& e){out.exception=true;out.valid=false;out.error=std::string(e.what()).substr(0,256);}
    out.primal=z_;out.dual=dual_;out.workspace=detach();out.wall_ms=ms(started);
  };
  int chosen=-1;
  try{
    if(multistart_mode_==1&&multistart_verify_==2){run(1,multistart_regenerated_,2);run(0,multistart_shifted_,2);}
    else {if(multistart_mode_==1)run(0,multistart_shifted_,2);
      run(1,multistart_regenerated_,multistart_mode_==1?2:4);}
    // Exact ties use legacy regeneration. Finite rejected-SQP outcomes remain
    // candidates at their last accepted NLP point, never at an unaccepted QP x.
    if(branches[1].valid)chosen=1;
    if(branches[0].valid&&(chosen<0||branches[0].final_merit<branches[1].final_merit))chosen=0;
    const int calls=branches[0].qp_calls+branches[1].qp_calls;
    if(calls>4||branches[0].info.iterations+branches[1].info.iterations>4*g::osqp_iters)
      throw std::runtime_error("Multistart exceeded the four-QP budget");
    if(multistart_trace_){auto& out=*multistart_trace_;
      const bool vectors=multistart_verify_>0&&multistart_trace_records_<4;++multistart_trace_records_;
      auto number=[&](double v){if(!std::isfinite(v))out<<"null";else if(v==0.&&std::signbit(v))out<<"-0.0";else out<<v;};
      auto quoted=[&](const std::string& text){out<<'"';for(unsigned char c:text){
        if(c=='"'||c=='\\')out<<'\\'<<c;else if(c<32)out<<' ';else out<<c;}out<<'"';};
      out<<"{\"controller_time\":"<<seed_controller_time_<<",\"mode\":"<<multistart_mode_
         <<",\"first_virtual_contact_ordinal\":"<<multistart_seed_.events[0].contact_ordinal
         <<",\"first_virtual_lead_s\":"<<multistart_seed_.events[0].lead
         <<",\"candidate_contact_count\":"<<multistart_seed_.all_contacts
         <<",\"fresh_workspaces\":true,\"verification_mode\":"<<multistart_verify_
         <<",\"vector_trace\":"<<(vectors?"true":"false")
         <<",\"branch_order\":\""<<(multistart_mode_==2?"regenerated":multistart_verify_==2?"regenerated,shifted":"shifted,regenerated")<<"\""
         <<",\"qp_calls\":"<<calls
         <<",\"admm_iterations\":"<<branches[0].info.iterations+branches[1].info.iterations
         <<",\"selected\":"<<(chosen<0?"null":chosen==0?"\"shifted\"":"\"regenerated\"")
         <<",\"branches\":[";
      for(int b=0;b<2;++b){if(b)out<<',';const auto& r=branches[b];
        out<<"{\"seed\":\""<<(b==0?"shifted":"regenerated")<<"\",\"evaluated\":"<<(r.evaluated?"true":"false")
           <<",\"valid\":"<<(r.valid?"true":"false")<<",\"exception\":"<<(r.exception?"true":"false")
           <<",\"qp_calls\":"<<r.qp_calls<<",\"sqp_iterations\":"<<r.info.sqp_iterations
           <<",\"admm_iterations\":"<<r.info.iterations<<",\"status\":"<<r.info.status
           <<",\"initial_cost\":";number(r.initial_cost);out<<",\"initial_merit\":";number(r.initial_merit);
        out<<",\"final_cost\":";number(r.evaluated?r.info.cost:std::numeric_limits<double>::quiet_NaN());
        out<<",\"error\":";quoted(r.error);
        out<<",\"final_merit\":";number(r.final_merit);out<<",\"constraint_l1\":";number(r.constraint_l1);
        out<<",\"bound_l1\":";number(r.bound_l1);out<<",\"constraint_max\":";number(r.info.violation);
        out<<",\"bound_max\":";number(r.info.bound_violation);out<<",\"last_alpha\":";number(r.info.alpha);
        out<<",\"primal_residual\":";number(r.info.primal_residual);out<<",\"dual_residual\":";number(r.info.dual_residual);
        out<<",\"wall_ms\":";number(r.wall_ms);
        if(vectors&&r.evaluated){auto vector=[&](const char* key,const Vector& data){out<<",\""<<key<<"\":[";
          for(int j=0;j<data.size();++j){if(j)out<<',';number(data[j]);}out<<']';};
          vector("final_primal",r.primal);vector("final_dual",r.dual);}
        out<<'}';}
      out<<"]}\n";out.flush();
    }
    if(chosen<0)throw std::runtime_error("Both multistart branches failed or produced nonfinite merit");
    Outcome& winner=branches[chosen];attach(winner.workspace);
    z_=winner.primal;dual_=winner.dual;trial_constraints_=winner.constraints;constraints_=winner.constraints;cost_=winner.info.cost;
    const double shift_ms=info.shift_ms;const int fallback=info.fallback_stage;info=winner.info;
    info.shift_ms=shift_ms;info.fallback_stage=fallback;
    info.data_ms=branches[0].info.data_ms+branches[1].info.data_ms;
    info.qp_ms=branches[0].info.qp_ms+branches[1].info.qp_ms;
    info.merit_ms=branches[0].info.merit_ms+branches[1].info.merit_ms;
    info.sqp_iterations=branches[0].info.sqp_iterations+branches[1].info.sqp_iterations;
    info.iterations=branches[0].info.iterations+branches[1].info.iterations;
  }catch(...){
    // Keep ownership exception-safe, including a partially-created OSQP setup.
    auto abandoned=detach();attach(previous_workspace);z_=previous_z;dual_=previous_dual;throw;
  }
  // The previous and losing workspaces are destroyed here. The winner survives
  // into the next ordinary update with its accepted external primal and dual.
}
#endif
void Controller::solve_sqp(const Eigen::Matrix<double,30,1>& p,int count,StepInfo& info,int* qp_calls){
#ifdef JUGGLING_SOLVER_MERIT_MULTIPLIER
  const double merit_multiplier=JUGGLING_SOLVER_MERIT_MULTIPLIER;
  if(!std::isfinite(merit_multiplier)||merit_multiplier<=0.)throw std::runtime_error("Invalid SQP merit multiplier");
#else
  constexpr double merit_multiplier=1000.;
#endif
#ifdef JUGGLING_SOLVER_DUAL_STEP_BLEND
  const double dual_step_blend=JUGGLING_SOLVER_DUAL_STEP_BLEND;
  if(dual_step_blend!=0.&&dual_step_blend!=1.)throw std::runtime_error("Dual step blend must be 0 or 1");
#endif
#ifdef JUGGLING_SOLVER_QP_RETRY_WARMSTART
  const double qp_retry_warmstart=JUGGLING_SOLVER_QP_RETRY_WARMSTART;
  if(qp_retry_warmstart!=0.&&qp_retry_warmstart!=1.&&qp_retry_warmstart!=2.)throw std::runtime_error("QP retry warmstart must be 0, 1 or 2");
  Vector retry_primal,retry_dual;bool retry_ready=false;
#endif
  auto solve_qp=[&](){
    if(qp_calls)++*qp_calls;
    const auto result=osqp_solve(solver_);
    // Ordinary successful calls are accounted below, exactly as before. A
    // failed branch still reports any ADMM work spent before OSQP returned.
    if(result&&qp_calls&&solver_->info){info.iterations+=solver_->info->iter;info.status=solver_->info->status_val;}
    return result;
  };
  for(int sqp=0;sqp<count;++sqp){
  info.alpha=0.;auto tic=Clock::now();
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
  linearize_lifted(z_,p);
#else
  linearize(z_,p);q_=gradient_;
  for(int c=0;c<n_;++c)for(int j=H_.columns[c];j<H_.columns[c+1];++j){int r=H_.rows[j];q_[r]-=H_.values[j]*z_[c];if(r!=c)q_[c]-=H_.values[j]*z_[r];}
#endif
  Vector az=Vector::Zero(m_);
  for(int c=0;c<n_;++c)for(int j=A_.columns[c];j<A_.columns[c+1];++j)if(A_.rows[j]<m_)az[A_.rows[j]]+=A_.values[j]*z_[c];
  l_.head(m_)=cl_-constraints_+az;u_.head(m_)=cu_-constraints_+az;
  for(int i=0;i<n_;++i){double center=std::clamp(z_[i],lower_[i],upper_[i]);int j=i%g::NZ;bool unlimited=(j>=13&&j<20)||(j>=g::NX&&j<g::NX+7);
    l_[m_+i]=unlimited?lower_[i]:std::max(lower_[i],center-g::trust);u_[m_+i]=unlimited?upper_[i]:std::min(upper_[i],center+g::trust);}
  for(int i=0;i<g::NP;++i){double measured=(p[i]-g::offset[i])/g::scale[i];l_[m_+i]=std::min(l_[m_+i],measured);u_[m_+i]=std::max(u_[m_+i],measured);}
  // OSQP uses finite infinity sentinels internally.
  for(int i=0;i<l_.size();++i){if(std::isinf(l_[i]))l_[i]=-OSQP_INFTY;if(std::isinf(u_[i]))u_[i]=OSQP_INFTY;}
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
  lift_->prepare(A_,l_,u_,z_,dual_);
  const int qp_n=lift_->variables(),qp_m=lift_->rows();const auto& qp_h=lift_->P;const auto& qp_a=lift_->A;
  const Vector& qp_q=lift_->q;const Vector& qp_l=lift_->l;const Vector& qp_u=lift_->u;
  const Vector& normal_primal=lift_->warm_primal;const Vector& normal_dual=lift_->warm_dual;
#else
  const int qp_n=n_,qp_m=m_+n_;const auto& qp_h=H_;const auto& qp_a=A_;
  const Vector& qp_q=q_;const Vector& qp_l=l_;const Vector& qp_u=u_;
  const Vector& normal_primal=z_;const Vector& normal_dual=dual_;
#endif
  info.data_ms+=ms(tic);tic=Clock::now();
  if(!solver_){OSQPSettings settings;osqp_set_default_settings(&settings);settings.verbose=0;settings.max_iter=g::osqp_iters;settings.check_termination=0;settings.polishing=0;settings.warm_starting=1;settings.adaptive_rho=0;settings.rho=g::rho;settings.eps_abs=1e-4;settings.eps_rel=1e-4;
#ifdef JUGGLING_SOLVER_ADAPTIVE_RHO
    const double adaptive_rho=JUGGLING_SOLVER_ADAPTIVE_RHO;
    if(adaptive_rho!=0.&&adaptive_rho!=1.)throw std::runtime_error("Adaptive rho must be 0 or 1");
    settings.adaptive_rho=adaptive_rho?OSQP_ADAPTIVE_RHO_UPDATE_ITERATIONS:OSQP_ADAPTIVE_RHO_UPDATE_DISABLED;
    // A nonzero iteration interval avoids OSQP's automatic wall-time heuristic.
    settings.adaptive_rho_interval=25;
#endif
#ifdef JUGGLING_SOLVER_ADAPTIVE_RHO_INTERVAL
    const double rho_interval=JUGGLING_SOLVER_ADAPTIVE_RHO_INTERVAL;
    if(!std::isfinite(rho_interval)||rho_interval<1.||rho_interval!=std::floor(rho_interval)
       ||rho_interval>=static_cast<double>(std::numeric_limits<OSQPInt>::max()))throw std::runtime_error("Invalid adaptive rho iteration interval");
    settings.adaptive_rho_interval=static_cast<OSQPInt>(rho_interval);
#endif
#ifdef JUGGLING_SOLVER_OSQP_ALPHA
    const double osqp_alpha=JUGGLING_SOLVER_OSQP_ALPHA;
    if(!std::isfinite(osqp_alpha)||osqp_alpha<=0.||osqp_alpha>=2.)throw std::runtime_error("OSQP alpha must be between 0 and 2");
    settings.alpha=osqp_alpha;
#endif
#ifdef JUGGLING_SOLVER_RHO_IS_VEC
    const double rho_is_vec=JUGGLING_SOLVER_RHO_IS_VEC,scaling=JUGGLING_SOLVER_SCALING,tolerance=JUGGLING_SOLVER_ADAPTIVE_RHO_TOLERANCE;
    if((rho_is_vec!=0.&&rho_is_vec!=1.)||!std::isfinite(scaling)||scaling<0.||scaling>20.||scaling!=std::floor(scaling)
       ||!std::isfinite(tolerance)||tolerance<1.)throw std::runtime_error("Invalid OSQP penalty/scaling option");
    settings.rho_is_vec=static_cast<OSQPInt>(rho_is_vec);settings.scaling=static_cast<OSQPInt>(scaling);settings.adaptive_rho_tolerance=tolerance;
#endif
    osqp_h_=OSQPCscMatrix_new(qp_n,qp_n,qp_h.values.size(),const_cast<double*>(qp_h.values.data()),const_cast<OSQPInt*>(qp_h.rows.data()),const_cast<OSQPInt*>(qp_h.columns.data()));
    osqp_a_=OSQPCscMatrix_new(qp_m,qp_n,qp_a.values.size(),const_cast<double*>(qp_a.values.data()),const_cast<OSQPInt*>(qp_a.rows.data()),const_cast<OSQPInt*>(qp_a.columns.data()));
    if(osqp_setup(&solver_,osqp_h_,qp_q.data(),osqp_a_,qp_l.data(),qp_u.data(),qp_m,qp_n,&settings))throw std::runtime_error("OSQP setup failed");
  // Match the reference wrapper: update vectors before matrices. Matrix updates
  // rescale the QP, so reversing these calls changes a fixed-budget solve.
  }else{
#ifdef JUGGLING_SOLVER_QP_RETRY_WARMSTART
    // After rejection the NLP point, parameters and QP are unchanged. Mode 2
    // also preserves OSQP's auxiliary projected iterate and current scaling.
    if(retry_ready&&qp_retry_warmstart==2.){
      // Reset public status/counters without changing data, scaling or ADMM
      // iterates. Otherwise a prior SOLVED status can survive a later retry.
      if(osqp_update_data_vec(solver_,nullptr,nullptr,nullptr))throw std::runtime_error("OSQP information reset failed");
    }else
#endif
    {if(osqp_update_data_vec(solver_,qp_q.data(),qp_l.data(),qp_u.data())||osqp_update_data_mat(solver_,qp_h.values.data(),nullptr,qp_h.values.size(),qp_a.values.data(),nullptr,qp_a.values.size()))throw std::runtime_error("OSQP data update failed");}}
#ifdef JUGGLING_EXPERIMENTAL_GROUP_RHO
  // Bounds updates reset OSQP's full rho vector, even with unchanged types.
  // Restore vector, reciprocal and KKT only after ALL ordinary data updates.
  // An exact rejected-QP continuation has not changed any of them.
#ifdef JUGGLING_SOLVER_QP_RETRY_WARMSTART
  if(!(retry_ready&&qp_retry_warmstart==2.))
#endif
  install_group_rho(sqp);
#endif
#if defined(JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT) && defined(JUGGLING_LIFT_LINK_RHO_MULTIPLIER)
  // Only auxiliary equality penalties change; physical QP data and iterates
  // remain intact. Preserve all ADMM state on an exact rejected-QP retry.
#ifdef JUGGLING_SOLVER_QP_RETRY_WARMSTART
  if(!(retry_ready&&qp_retry_warmstart==2.))
#endif
  lift_->install_link_rho(solver_,JUGGLING_LIFT_LINK_RHO_MULTIPLIER);
#endif
#ifdef JUGGLING_SOLVER_QP_RETRY_WARMSTART
  const Vector& qp_primal=retry_ready?retry_primal:normal_primal;
  const Vector& qp_dual=retry_ready?retry_dual:normal_dual;
  info.warm_dual_nonzeros=(qp_dual.array()!=0).count();
  if(!(retry_ready&&qp_retry_warmstart==2.)&&osqp_warm_start(solver_,qp_primal.data(),qp_dual.data()))throw std::runtime_error("OSQP warmstart failed");
  if(solve_qp())throw std::runtime_error("OSQP solve failed");
#else
  info.warm_dual_nonzeros=(normal_dual.array()!=0).count();
  if(osqp_warm_start(solver_,normal_primal.data(),normal_dual.data())||solve_qp())throw std::runtime_error("OSQP solve failed");
#endif
  info.qp_ms+=ms(tic);info.iterations+=solver_->info->iter;info.status=solver_->info->status_val;info.primal_residual=solver_->info->prim_res;info.dual_residual=solver_->info->dual_res;tic=Clock::now();
  double merit0=cost_+merit_multiplier*g::objective_scale*violation_sum(constraints_,z_);
  Vector candidate=Eigen::Map<const Vector>(solver_->solution->x,n_);
  if(candidate.allFinite()&&candidate.cwiseAbs().maxCoeff()<1e6){Vector dz=candidate-z_;for(double alpha:{1.,.5,.25,.125,.0625}){trial_=z_+alpha*dz;double objective=value(trial_,p,trial_constraints_);double merit=objective+merit_multiplier*g::objective_scale*violation_sum(trial_constraints_,trial_);
      if(merit<=merit0+1e-8){z_=trial_;
#ifdef JUGGLING_SOLVER_DUAL_STEP_BLEND
        if(dual_step_blend==1.&&alpha<1.)dual_=((1.-alpha)*dual_+alpha*Eigen::Map<const Vector>(solver_->solution->y,m_+n_)).eval();
        else
#endif
        dual_=Eigen::Map<const Vector>(solver_->solution->y,m_+n_);info.alpha=alpha;break;}}}
#ifdef JUGGLING_SOLVER_QP_RETRY_WARMSTART
  // Rejected SQP steps retain their accepted NLP state and multipliers. The
  // next inner solve has the same linearization, so continue its finite QP
  // iterate rather than spending another fixed budget from the same old seed.
  bool retry_status=info.status==OSQP_SOLVED||info.status==OSQP_SOLVED_INACCURATE||info.status==OSQP_MAX_ITER_REACHED;
  retry_ready=qp_retry_warmstart>0.&&info.alpha==0.&&retry_status&&candidate.allFinite()&&candidate.cwiseAbs().maxCoeff()<1e6;
  if(retry_ready){retry_primal=Eigen::Map<const Vector>(solver_->solution->x,qp_n);retry_dual=Eigen::Map<const Vector>(solver_->solution->y,qp_m);retry_ready=retry_primal.allFinite()&&retry_dual.allFinite()&&retry_dual.cwiseAbs().maxCoeff()<1e20;}
#endif
  info.cost=value(z_,p,trial_constraints_);info.violation=std::max((cl_-trial_constraints_).cwiseMax(0).maxCoeff(),(trial_constraints_-cu_).cwiseMax(0).maxCoeff());info.bound_violation=std::max((lower_-z_).cwiseMax(0).maxCoeff(),(z_-upper_).cwiseMax(0).maxCoeff());
  info.merit_ms+=ms(tic);++info.sqp_iterations;
  }
}
StepInfo Controller::step(const State& x,const Joint& torque,double elapsed){
#ifdef JUGGLING_BOUNDED_MULTISTART_MODE
  multistart_pending_=false;
#endif
  auto start=Clock::now();StepInfo info;info.evaluator_threads=evaluator_threads_;if(!x.allFinite()||!torque.allFinite()||!std::isfinite(elapsed)||elapsed<0)throw std::runtime_error("Invalid controller input");
#ifdef JUGGLING_SOLVER_LANDING_WEIGHT
  if(JUGGLING_SOLVER_LANDING_WEIGHT>0.)freeze_landing_guidance(x);
#endif
  if(!initialized_)initialize(x,torque);else if(elapsed>0)shift(elapsed,x,torque);
#ifdef JUGGLING_FIRST_EVENT_CENTER_WEIGHT
  else if(JUGGLING_FIRST_EVENT_CENTER_WEIGHT>0.)freeze_first_event_center(x,0.);
#endif
  info.shift_ms=ms(start);info.fallback_stage=fallback_stage_;
#ifdef JUGGLING_BOUNDED_MULTISTART_MODE
  if(!multistart_pending_)
#endif
  if(!z_.allFinite()||(lower_-z_).cwiseMax(0).maxCoeff()>100||(z_-upper_).cwiseMax(0).maxCoeff()>100)throw std::runtime_error("Warmstart diverged outside model range");
  if(g::sqp_iters<1||g::sqp_iters>4||g::sqp_iters!=std::floor(g::sqp_iters))throw std::runtime_error("SQP iteration count must be an integer in [1,4]");
  Eigen::Matrix<double,30,1> p;p<<x,torque;
#ifdef JUGGLING_BOUNDED_MULTISTART_MODE
  if(multistart_pending_)solve_multistart(p,info);else
#endif
  solve_sqp(p,static_cast<int>(g::sqp_iters),info);
#ifdef JUGGLING_TAIL_ENDPOINT_START_S
  if(!tail_endpoint_penalties_.empty())trace_tail_endpoint_elasticity(trial_constraints_,info);
#endif
  set_physical_plan();Joint desired=torque+(plan_.col(1).segment<7>(g::NX)-plan_.col(0).segment<7>(g::NX))*g::control_dt/g::stage_steps[0];
  for(int j=0;j<7;++j)target_[j]=std::clamp(desired[j],std::max(-g::torque_limit[j],torque[j]-g::torque_rate_limit[j]*g::control_dt),std::min(g::torque_limit[j],torque[j]+g::torque_rate_limit[j]*g::control_dt));
  info.clipping=(target_-desired).cwiseAbs().maxCoeff();info.total_ms=ms(start);return info;
}
}

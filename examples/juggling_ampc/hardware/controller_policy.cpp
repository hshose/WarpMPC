#include "controller.hpp"
#include "ampc_policy.hpp"
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

namespace juggling {
namespace g=generated;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static Eigen::Matrix3d skew(const Eigen::Vector3d& v){Eigen::Matrix3d s;s<<0,-v.z(),v.y(),v.z(),0,-v.x(),-v.y(),v.x(),0;return s;}
template<size_t N> static Eigen::Map<const Eigen::Matrix<double,N,1>> map(const std::array<double,N>& a){return Eigen::Map<const Eigen::Matrix<double,N,1>>(a.data());}
static int source(double time){return std::clamp(int(std::upper_bound(g::stage_times.begin(),g::stage_times.end(),time+1e-12)-g::stage_times.begin())-1,0,g::N);}

Evaluation::Evaluation(const std::string& directory,const std::string& name,bool compiled){
  std::string individual=directory+"/"+name+".so";
  std::string library=std::filesystem::exists(individual)?individual:directory+"/libjuggling_factors.so";
  // Neural inference only evaluates robot functions. Load unused optimization
  // factors for their sparsity metadata without compiling their derivatives.
  if(compiled && name.rfind("panda_",0)==0 && !std::filesystem::exists(library))
    throw std::runtime_error("Missing compiled robot function: "+name);
  function=compiled&&std::filesystem::exists(library)?casadi::external(name,library):casadi::Function::load(directory+"/"+name+".casadi");
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
Geometry RobotModel::geometry(const State& x){
  fk_.call({x.data()+6});Eigen::Vector3d p=fk_.dense(0),n=fk_.dense(1),rel=x.head<3>()-p+map(g::origin);
  Matrix J=fk_.dense(2),N=fk_.dense(3);Eigen::Vector3d tangent=rel-n*(n.dot(rel));
  return {n.dot(rel)-g::radius,tangent.norm(),n.dot(x.segment<3>(3))-(n.transpose()*J-rel.transpose()*N).dot(x.segment<7>(13)),n};
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

Controller::Controller(const std::string& d,bool c):robot(d,c),first_data_(d,"first_data",c),stage_data_(d,"stage_data",c),last_data_(d,"last_data",c),first_value_(d,"first_value",c),stage_value_(d,"stage_value",c),last_value_(d,"last_value",c),n_(g::NZ*(g::N+1)),m_(g::CONSTRAINTS){
  z_=Vector::Zero(n_);dual_=Vector::Zero(m_+n_);lower_.resize(n_);upper_.resize(n_);cl_.resize(m_);cu_.resize(m_);
  gradient_.resize(n_);constraints_.resize(m_);q_.resize(n_);l_.resize(m_+n_);u_.resize(m_+n_);trial_.resize(n_);trial_constraints_.resize(m_);
  lower_=map(g::all_lower);upper_=map(g::all_upper);cl_=map(g::constraint_lower);cu_=map(g::constraint_upper);
  layouts_.push_back({&first_data_,&first_value_,0,0,g::stage_steps[0],{},{}});
  for(int k=0;k<g::N;++k){
    stage_evaluators_.push_back(std::make_unique<Evaluation>(d,g::stage_data_names[k],c));
    stage_values_.push_back(std::make_unique<Evaluation>(d,g::stage_value_names[k],c));
    layouts_.push_back({stage_evaluators_.back().get(),stage_values_.back().get(),k*g::NZ,g::stage_offsets[k],g::stage_steps[k],{},{}});
  }
  layouts_.push_back({&last_data_,&last_value_,g::N*g::NZ,g::stage_offsets[g::N],g::stage_steps.back(),{},{}});build_pattern();
  const char* threads=std::getenv("JUGGLING_EVAL_THREADS");evaluator_threads_=threads?std::stoi(threads):2;
  if(evaluator_threads_<1||evaluator_threads_>16)throw std::runtime_error("JUGGLING_EVAL_THREADS must be 1..16");
  if(evaluator_threads_>1){std::vector<EvaluationPool::Job> jobs;for(auto& f:layouts_)jobs.push_back({f.data,f.column,f.step});pool_=std::make_unique<EvaluationPool>(evaluator_threads_,jobs);}
}
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
      gradient_.segment<7>(offset)+=weight*scale*posture_projector_*projected;
      auto h=(posture_projector_.transpose()*posture_projector_).eval();
      for(int c=0;c<7;++c)for(int r=0;r<=c;++r)
        H_.values[posture_hessian_dest_[dest++]]+=weight*scale*scale*h(r,c);
    }
  }
  return cost;
}
// This implementation never creates an OSQP solver or OSQP matrix object.
Controller::~Controller()=default;
void Controller::set_contact_bias(const Eigen::Vector3d& bias){
  if(!bias.allFinite()||(bias.array()!=0.).any())
    throw std::runtime_error("Neural policy has no contact-bias input; only zero is supported");
}
void Controller::set_reference(const Eigen::Vector3d& offset,const Joint& posture){
  if(!offset.allFinite()||!posture.allFinite()||offset.head<2>().norm()>.32||std::abs(offset[2])>.050001)
    throw std::runtime_error("Fixture reference outside local workspace");
  for(int j=0;j<7;++j)if(posture[j]<g::joint_lower[j]||posture[j]>g::joint_upper[j]||std::abs(posture[j]-g::home[j])>1.0)
    throw std::runtime_error("Fixture posture outside local joint envelope");
  reference_offset_=offset;reference_posture_delta_=posture-map(g::home);
  update_posture_projector();
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
  }
  if(JUGGLING_NULLSPACE_POSITION_WEIGHT!=0. || JUGGLING_NULLSPACE_VELOCITY_WEIGHT!=0.)
    for(int k=0;k<g::N;++k)for(int block:{6,13})for(int c=0;c<7;++c)for(int r=0;r<=c;++r)
      hp.emplace(k*g::NZ+block+c,k*g::NZ+block+r);
  auto make=[&](const std::set<Key>& pattern,Sparse& s){
    std::map<Key,int> lookup;s.columns.resize(n_+1);int column=0;
    for(auto key:pattern){while(column<=key.first)s.columns[column++]=s.rows.size();lookup[key]=s.rows.size();s.rows.push_back(key.second);}
    while(column<=n_)s.columns[column++]=s.rows.size();s.values.resize(s.rows.size());return lookup;
  };
  auto hm=make(hp,H_),am=make(ap,A_);diagonal_.resize(n_);
  for(int i=0;i<n_;++i)diagonal_[i]=hm.at({i,i});
  if(JUGGLING_NULLSPACE_POSITION_WEIGHT!=0. || JUGGLING_NULLSPACE_VELOCITY_WEIGHT!=0.){
    update_posture_projector();
    for(int k=0;k<g::N;++k)for(int block:{6,13})for(int c=0;c<7;++c)for(int r=0;r<=c;++r)
      posture_hessian_dest_.push_back(hm.at({k*g::NZ+block+c,k*g::NZ+block+r}));
  }
  for(auto& f:layouts_){
    auto jr=f.data->function.sparsity_out(1),jc=f.data->function.sparsity_out(3);
    std::vector<std::vector<std::pair<int,int>>> rows(jr.size1());
    for(int c=0;c<jr.size2();++c)for(int j=jr.colind(c);j<jr.colind(c+1);++j)rows[jr.row(j)].emplace_back(c,j);
    for(auto& row:rows)for(auto a:row)for(auto b:row)if(a.first<=b.first)f.hessian.push_back({hm.at({f.column+b.first,f.column+a.first}),a.second,b.second});
    for(int c=0;c<jc.size2();++c)for(int j=jc.colind(c);j<jc.colind(c+1);++j)f.a_dest.push_back(am.at({f.column+c,f.row+jc.row(j)}));
  }
  for(int c=0;c<n_;++c)A_.values[am.at({c,m_+c})]=1.;
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
void Controller::linearize(const Vector& at,const Eigen::Matrix<double,30,1>& parameter){
  std::fill(H_.values.begin(),H_.values.end(),0.);for(int d:diagonal_)H_.values[d]=g::proximal;
  gradient_.setZero();cost_=0.;
  if(pool_)pool_->evaluate(at.data(),parameter.data());else for(auto& f:layouts_)f.data->call({at.data()+f.column,parameter.data()});
  for(auto& f:layouts_){
    auto& out=f.data->output;
    {auto rs=f.data->function.sparsity_out(0),js=f.data->function.sparsity_out(1);
     for(size_t j=0;j<out[0].size();++j)out[0][j]*=identified_cost_scale(rs.size1(),rs.row(j));
     for(size_t j=0;j<out[1].size();++j)out[1][j]*=identified_cost_scale(js.size1(),js.row(j));}
    if(g::quadrature && f.data->function.size1_out(0)>=54){
      double weight=std::sqrt(f.step/.03);auto sparsity=f.data->function.sparsity_out(1);
      auto rs=f.data->function.sparsity_out(0);for(size_t j=0;j<out[0].size();++j){int r=rs.row(j);if(r<51)out[0][j]*=weight*((r>=30&&r<37)?.03/f.step:1.);}
      for(size_t j=0;j<out[1].size();++j){int r=sparsity.row(j);if(r<51)out[1][j]*=weight*((r>=30&&r<37)?.03/f.step:1.);}
    }
    Vector residual=f.data->dense(0);reference_residual(residual,g::quadrature&&residual.size()>=54?std::sqrt(f.step/.03):1.);
    Vector con=f.data->dense(2);constraints_.segment(f.row,con.size())=con;
    cost_+=.5*residual.squaredNorm();auto jr=f.data->function.sparsity_out(1);
    for(int c=0;c<jr.size2();++c)for(int j=jr.colind(c);j<jr.colind(c+1);++j)gradient_[f.column+c]+=out[1][j]*residual[jr.row(j)];
    for(auto product:f.hessian)H_.values[product.destination]+=out[1][product.left]*out[1][product.right];
    for(size_t j=0;j<f.a_dest.size();++j)A_.values[f.a_dest[j]]=out[3][j];
  }
  cost_+=posture_regularization(at,true);
  if(!gradient_.allFinite()||!constraints_.allFinite()||!std::isfinite(cost_)||
     !std::all_of(H_.values.begin(),H_.values.end(),[](double v){return std::isfinite(v);})||
     !std::all_of(A_.values.begin(),A_.values.end(),[](double v){return std::isfinite(v);}))throw std::runtime_error("Non-finite SQP data");
}
double Controller::value(const Vector& at,const Eigen::Matrix<double,30,1>& p,Vector& constraints){
  double cost=0.;
  for(auto& f:layouts_){f.value->call({at.data()+f.column,p.data()});Vector residual=f.value->dense(0);for(int j=0;j<residual.size();++j)residual[j]*=identified_cost_scale(residual.size(),j);reference_residual(residual);if(g::quadrature&&residual.size()>=54){residual.head(51)*=std::sqrt(f.step/.03);residual.segment(30,7)*=.03/f.step;}cost+=.5*residual.squaredNorm();Vector c=f.value->dense(1);constraints.segment(f.row,c.size())=c;}
  return cost+posture_regularization(at,false);
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
double apply_ball_impact(RobotModel& robot,State& x){
  auto f=robot.contact_fk(x.segment<7>(6));Eigen::Vector3d n=f[1],rel=x.head<3>()-f[0].col(0)+map(g::origin),tangent=rel-n*(n.dot(rel));
  Matrix J=f[2]-skew(tangent)*f[3];auto M=robot.mass(x.segment<7>(6)).ldlt();Matrix response=M.solve(J.transpose());
  Eigen::Vector3d relative=x.segment<3>(3)-x.segment<3>(20).cross(g::radius*n)-J*x.segment<7>(13);
  Matrix K=Matrix::Identity(3,3)+(Matrix::Identity(3,3)-n*n.transpose())/g::inertia_ratio+g::mass*J*response;
  Matrix T=f[4].leftCols(2);auto tangent_mass=(T.transpose()*K*T).eval().ldlt();double vn=n.dot(relative);
  auto impulse=[&](double j)->Eigen::Vector3d{Eigen::Vector2d desired=-tangent_mass.solve(T.transpose()*(relative+K*n*j));double cap=g::friction*j;return j*n+T*(desired*cap/std::sqrt(cap*cap+desired.squaredNorm()+1e-10));};
  double lo=0.,hi=std::max(1.,-3*vn);
  // Monotone normal equation; bisection tolerance is tighter than the root
  // tolerance of the reference NumPy/Brent warmstart.
  for(int k=0;k<50;++k){double mid=.5*(lo+hi);if(n.dot(K*impulse(mid))+(1+g::restitution)*vn>0)hi=mid;else lo=mid;}
  double j=.5*(lo+hi);Eigen::Vector3d u=impulse(j);x.segment<3>(3)+=u;x.segment<3>(20)-=n.cross(u)/(g::inertia_ratio*g::radius);
  x.segment<7>(13)-=g::mass*response*u;double gap=n.dot(rel)-g::radius;x.head<3>()+=(1e-10-gap)*n;return g::mass*j;
}
void Controller::impact(State& x,Contact& contact,bool,bool){contact.impulse=apply_ball_impact(robot,x);}
std::pair<State,std::vector<Contact>> Controller::rollout(const State& x,double duration,const Joint& acc,double max_step,const Joint*,SeedContext*,int){
  State state=x;std::vector<Contact> contacts;double elapsed=0.;
  while(elapsed<duration-1e-14){
    double dt=std::min(max_step,duration-elapsed);State end=free(state,dt,acc);auto a=robot.geometry(state),b=robot.geometry(end);
    if(a.gap>=-1e-10&&b.gap<=0&&(a.speed<0||b.speed<0)){
      double lo=0.,hi=dt;while(hi-lo>1e-10){double mid=.5*(lo+hi);if(robot.geometry(free(state,mid,acc)).gap>0)lo=mid;else hi=mid;}
      double hit=.5*(lo+hi);State collision=free(state,hit,acc);auto geo=robot.geometry(collision);
      if(geo.radial<=g::paddle_radius&&geo.speed<0){Contact c;impact(collision,c);elapsed+=hit;c.time=elapsed;contacts.push_back(c);state=collision;if(contacts.size()>16)throw std::runtime_error("Unsupported resting contact");continue;}
    }
    state=end;elapsed+=dt;
  }
  return {state,contacts};
}
void Controller::initialize(const State& x,const Joint& torque){
  Matrix Y=Matrix::Zero(g::NZ,g::N+1);State state=x;for(int k=0;k<=g::N;++k)Y(g::NX+9,k)=g::stage_steps[std::min(k,g::N-1)];Y.row(g::NX+8).setConstant(.5);
  for(int k=0;k<g::N;++k){Y.col(k).head<g::NX>()=state;Y.col(k).segment<7>(g::NX)=robot.rnea(state.segment<7>(6),state.segment<7>(13),Joint::Zero());
    try {auto out=rollout(state,g::stage_steps[k],Joint::Zero(),.003);state=out.first;
      if(!out.second.empty()){Y(g::NX+7,k)=out.second[0].impulse/g::mass;Y(g::NX+8,k)=out.second[0].time/g::stage_steps[k];}}
    catch(const std::runtime_error& e){if(std::string(e.what())!="Unsupported resting contact")throw;
      // Low-restitution stationary-paddle seed reaches Zeno settling inside
      // the horizon. Seed a supported ball; NEVER used by the physical plant.
      auto f=robot.contact_fk(state.segment<7>(6));Eigen::Vector3d n=f[1];
      Eigen::Vector3d p=f[0].col(0)-map(g::origin),rel=state.head<3>()-p;
      state.head<3>()+=n*(g::radius-n.dot(rel));state.segment<3>(3)=f[2]*state.segment<7>(13);
      Y(g::NX+7,k)=-ball_gravity[2]*g::stage_steps[k];Y(g::NX+8,k)=.5;
    }}
  Y.col(g::N).head<g::NX>()=state;Y.col(g::N).segment<7>(g::NX)=robot.rnea(state.segment<7>(6),state.segment<7>(13),Joint::Zero());
  Y.col(0).segment<7>(g::NX)=torque;
  for(int k=0;k<=g::N;++k){Y(g::NX+10,k)=g::stage_times[k];z_.segment<g::NZ>(k*g::NZ)=(Y.col(k)-map(g::offset)).cwiseQuotient(map(g::scale));}
  initialized_=true;
}
void Controller::set_physical_plan(){plan_.resize(g::NZ,g::N+1);for(int k=0;k<=g::N;++k)plan_.col(k)=z_.segment<g::NZ>(k*g::NZ).cwiseProduct(map(g::scale))+map(g::offset);}
void Controller::shift(double dt,const State& x,const Joint& torque){
  if(!initialized_)throw std::runtime_error("Cannot shift an uninitialized controller");
  set_physical_plan();Matrix old=plan_,Y=old;Y.row(g::NX+7).setZero();Y.row(g::NX+8).setConstant(.5);Eigen::Vector3d gravity=Eigen::Map<const Eigen::Vector3d>(ball_gravity.data());
  for(int k=0;k<=g::N;++k){double t=g::stage_times[k]+dt;int i=source(t);Row row=old.col(i);double h=g::stage_steps[std::min(i,g::N-1)],s=t-g::stage_times[i];Joint acc=(old.col(std::min(i+1,g::N)).segment<7>(13)-row.segment<7>(13))/h;
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
    Y.col(k).segment<7>(g::NX)=i==g::N?old.col(g::N).segment<7>(g::NX).eval():((1-s/h)*row.segment<7>(g::NX)+(s/h)*old.col(i+1).segment<7>(g::NX)).eval();
  }
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
      Joint a=robot.mass(q).ldlt().solve(Y.col(k).segment<7>(g::NX)-robot.bias(q,dq));
      Y.col(k).segment<7>(6)+=offset;Y.col(k).segment<7>(13)+=decay*delta_v;
      Y.col(k).segment<7>(g::NX)=robot.rnea(Y.col(k).segment<7>(6),Y.col(k).segment<7>(13),a-decay/feedback_time*delta_v);
      if(k<g::N)offset+=.5*g::stage_steps[k]*(decay+std::exp(-g::stage_times[k+1]/feedback_time))*delta_v;
    }
  }
#endif
  Y.col(0).head<g::NX>()=x;Y.col(0).segment<7>(g::NX)=torque;
  Vector previous=dual_;dual_.setZero();
  for(int k=0;k<=g::N;++k){int i=source(g::stage_times[k]+dt);
    if(i<g::N){
      if(k<g::N){int cursor=g::stage_offsets[i];for(int row=g::stage_offsets[k];row<g::stage_offsets[k+1];++row){
        int id=g::constraint_ids[row];while(cursor<g::stage_offsets[i+1]&&g::constraint_ids[cursor]<id)++cursor;
        if(cursor<g::stage_offsets[i+1]&&g::constraint_ids[cursor]==id&&!(id>=g::contact_begin&&id<g::contact_end))dual_[row]=previous[cursor];}}
      dual_.segment<g::NZ>(m_+k*g::NZ)=previous.segment<g::NZ>(m_+i*g::NZ);
    }}
  dual_.segment(m_,g::NP).setZero();
  for(int k=0;k<=g::N;++k){Y(g::NX+9,k)=g::stage_steps[std::min(k,g::N-1)];Y(g::NX+10,k)=g::stage_times[k];z_.segment<g::NZ>(k*g::NZ)=(Y.col(k)-map(g::offset)).cwiseQuotient(map(g::scale));}
  // Regenerate ball/contact seeds on the shifted joint arcs. On failure retain
  // the already shifted primal/dual, exactly as in the Python controller.
  State state=x;fallback_stage_=-1;
  for(int k=0;k<g::N;++k){Y.col(k).head<6>()=state.head<6>();Y.col(k).segment<3>(20)=state.segment<3>(20);Y(g::NX+7,k)=0;Y(g::NX+8,k)=.5;
    Joint acc=(Y.col(k+1).segment<7>(13)-Y.col(k).segment<7>(13))/g::stage_steps[k];
    try{auto out=rollout(Y.col(k).head<g::NX>(),g::stage_steps[k],acc,.01);state=out.first;if(guard_ball_rollout&&(state[2]<-.15||state[2]>1.5))throw std::runtime_error("Ball rollout left optimizer workspace");
      if(!out.second.empty()){Y(g::NX+7,k)=out.second[0].impulse/g::mass;Y(g::NX+8,k)=out.second[0].time/g::stage_steps[k];}}
    catch(const std::runtime_error& e){std::string message=e.what();if(message!="Unsupported resting contact"&&message!="Ball rollout left optimizer workspace")throw;fallback_stage_=k;return;}}
  Y.col(g::N).head<6>()=state.head<6>();Y.col(g::N).segment<3>(20)=state.segment<3>(20);
  for(int k=0;k<=g::N;++k)z_.segment<g::NZ>(k*g::NZ)=(Y.col(k)-map(g::offset)).cwiseQuotient(map(g::scale));
}
void Controller::set_warmstart(const double* z,const double* dual){z_=Eigen::Map<const Vector>(z,n_);dual_=Eigen::Map<const Vector>(dual,m_+n_);initialized_=true;}
double Controller::violation_sum(const Vector& c,const Vector& z) const{return (cl_-c).cwiseMax(0).sum()+(c-cu_).cwiseMax(0).sum()+(lower_-z).cwiseMax(0).sum()+(z-upper_).cwiseMax(0).sum();}
StepInfo Controller::step(const State& x,const Joint& torque,double elapsed){
  auto start=Clock::now();
  if(!x.allFinite()||!torque.allFinite()||!std::isfinite(elapsed)||elapsed<0)
    throw std::runtime_error("Invalid AMPC input");
  Eigen::Matrix<float,40,1> input;
  input.segment<23>(0)=x.cast<float>();input.segment<7>(23)=torque.cast<float>();
  input.segment<3>(30)=reference_offset_.cast<float>();
  input.segment<7>(33)=(map(g::home)+reference_posture_delta_).cast<float>();
  auto prediction=juggling_ampc::predict(input);
  plan_=Matrix::Zero(g::NZ,g::N+1);
  if constexpr(juggling_ampc::output_size==7){
    // Only the first future acceleration knot is reached before the next 20 ms
    // update. Encode it in the existing supervisor transport; hold it afterwards.
    Joint q=x.segment<7>(6),dq=x.segment<7>(13);
    Joint next_tau=robot.rnea(q,dq,prediction.head<7>().cast<double>());
    for(int k=0;k<4;++k){
      plan_.col(k).segment<7>(6)=q;plan_.col(k).segment<7>(13)=dq;
      plan_.col(k).segment<7>(23)=k==0?torque:next_tau;
      plan_(32,k)=g::stage_steps[k];plan_(33,k)=g::stage_times[k];
    }
  }else{
    for(int k=0;k<4;++k){
      plan_.col(k).segment<7>(6)=prediction.segment<7>(21*k).cast<double>();
      plan_.col(k).segment<7>(13)=prediction.segment<7>(21*k+7).cast<double>();
      plan_.col(k).segment<7>(23)=prediction.segment<7>(21*k+14).cast<double>();
      plan_(32,k)=g::stage_steps[k];plan_(33,k)=g::stage_times[k];
    }
  }
  // The unchanged supervisor projects these four nodes and the unchanged
  // session callback enforces all existing robot operating limits.
  plan_.col(0).head<6>()=x.head<6>();plan_.col(0).segment<3>(20)=x.tail<3>();
  StepInfo info;info.sqp_iterations=0;info.iterations=0;info.status=1;
  target_=torque;info.total_ms=ms(start);return info;
}
}

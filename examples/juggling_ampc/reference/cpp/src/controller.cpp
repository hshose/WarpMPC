#include "controller.hpp"
#include <cmath>
#include "model_config.hpp"
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

namespace juggling {
namespace g=generated;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static Eigen::Matrix3d skew(const Eigen::Vector3d& v){Eigen::Matrix3d s;s<<0,-v.z(),v.y(),v.z(),0,-v.x(),-v.y(),v.x(),0;return s;}
template<size_t N> static Eigen::Map<const Eigen::Matrix<double,N,1>> map(const std::array<double,N>& a){return Eigen::Map<const Eigen::Matrix<double,N,1>>(a.data());}
static int source(double time){return std::min(static_cast<int>(std::floor(time/g::h+1e-12)),g::N);}

Evaluation::Evaluation(const std::string& directory,const std::string& name,bool compiled){
  function=compiled?casadi::external(name,directory+"/libjuggling_factors.so"):casadi::Function::load(directory+"/"+name+".casadi");
  args_.resize(function.sz_arg(),nullptr);results_.resize(function.sz_res(),nullptr);
  iw_.resize(function.sz_iw());work_.resize(function.sz_w());output.resize(function.n_out());
  for(int i=0;i<function.n_out();++i){output[i].resize(function.nnz_out(i));results_[i]=output[i].data();}
  memory_=function.checkout();
}
Evaluation::~Evaluation(){function.release(memory_);}
void Evaluation::call(const std::vector<const double*>& inputs){
  std::copy(inputs.begin(),inputs.end(),args_.begin());
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
  struct Job {Evaluation* evaluator;int column;};
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

Controller::Controller(const std::string& d,bool c):robot(d,c),first_data_(d,"first_data",c),stage_data_(d,"stage_data",c),last_data_(d,"last_data",c),first_value_(d,"first_value",c),stage_value_(d,"stage_value",c),last_value_(d,"last_value",c),n_(g::NZ*(g::N+1)),m_(g::NF+g::N*g::NC+g::NL){
  z_=Vector::Zero(n_);dual_=Vector::Zero(m_+n_);lower_.resize(n_);upper_.resize(n_);cl_.resize(m_);cu_.resize(m_);
  gradient_.resize(n_);constraints_.resize(m_);q_.resize(n_);l_.resize(m_+n_);u_.resize(m_+n_);trial_.resize(n_);trial_constraints_.resize(m_);
  for(int k=0;k<=g::N;++k){lower_.segment<g::NZ>(k*g::NZ)=map(g::lower);upper_.segment<g::NZ>(k*g::NZ)=map(g::upper);}
  cl_.head(g::NF).setZero();cu_.head(g::NF).setZero();cl_[m_-1]=cu_[m_-1]=g::N*g::h;
  layouts_.push_back({&first_data_,&first_value_,0,0,{},{}});
  for(int k=0;k<g::N;++k){cl_.segment(k*g::NC+g::NF,g::NC)=map(g::stage_lower);cu_.segment(k*g::NC+g::NF,g::NC)=map(g::stage_upper);stage_evaluators_.push_back(std::make_unique<Evaluation>(d,"stage_data",c));layouts_.push_back({stage_evaluators_.back().get(),&stage_value_,k*g::NZ,g::NF+k*g::NC,{},{}});}
  layouts_.push_back({&last_data_,&last_value_,g::N*g::NZ,m_-1,{},{}});build_pattern();
  const char* threads=std::getenv("JUGGLING_EVAL_THREADS");evaluator_threads_=threads?std::stoi(threads):2;
  if(evaluator_threads_<1||evaluator_threads_>16)throw std::runtime_error("JUGGLING_EVAL_THREADS must be 1..16");
  if(evaluator_threads_>1){std::vector<EvaluationPool::Job> jobs;for(auto& f:layouts_)jobs.push_back({f.data,f.column});pool_=std::make_unique<EvaluationPool>(evaluator_threads_,jobs);}
}
Controller::~Controller(){if(solver_)osqp_cleanup(solver_);if(osqp_h_)OSQPCscMatrix_free(osqp_h_);if(osqp_a_)OSQPCscMatrix_free(osqp_a_);}
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
  auto make=[&](const std::set<Key>& pattern,Sparse& s){
    std::map<Key,int> lookup;s.columns.resize(n_+1);int column=0;
    for(auto key:pattern){while(column<=key.first)s.columns[column++]=s.rows.size();lookup[key]=s.rows.size();s.rows.push_back(key.second);}
    while(column<=n_)s.columns[column++]=s.rows.size();s.values.resize(s.rows.size());return lookup;
  };
  auto hm=make(hp,H_),am=make(ap,A_);diagonal_.resize(n_);
  for(int i=0;i<n_;++i)diagonal_[i]=hm.at({i,i});
  for(auto& f:layouts_){
    auto jr=f.data->function.sparsity_out(1),jc=f.data->function.sparsity_out(3);
    std::vector<std::vector<std::pair<int,int>>> rows(jr.size1());
    for(int c=0;c<jr.size2();++c)for(int j=jr.colind(c);j<jr.colind(c+1);++j)rows[jr.row(j)].emplace_back(c,j);
    for(auto& row:rows)for(auto a:row)for(auto b:row)if(a.first<=b.first)f.hessian.push_back({hm.at({f.column+b.first,f.column+a.first}),a.second,b.second});
    for(int c=0;c<jc.size2();++c)for(int j=jc.colind(c);j<jc.colind(c+1);++j)f.a_dest.push_back(am.at({f.column+c,f.row+jc.row(j)}));
  }
  for(int c=0;c<n_;++c)A_.values[am.at({c,m_+c})]=1.;
}
void Controller::linearize(const Vector& at,const Eigen::Matrix<double,30,1>& parameter){
  std::fill(H_.values.begin(),H_.values.end(),0.);for(int d:diagonal_)H_.values[d]=g::proximal;
  gradient_.setZero();cost_=0.;
  if(pool_)pool_->evaluate(at.data(),parameter.data());else for(auto& f:layouts_)f.data->call({at.data()+f.column,parameter.data()});
  for(auto& f:layouts_){
    const auto& out=f.data->output;
    Vector residual=f.data->dense(0);Vector con=f.data->dense(2);constraints_.segment(f.row,con.size())=con;
    cost_+=.5*residual.squaredNorm();auto jr=f.data->function.sparsity_out(1);
    for(int c=0;c<jr.size2();++c)for(int j=jr.colind(c);j<jr.colind(c+1);++j)gradient_[f.column+c]+=out[1][j]*residual[jr.row(j)];
    for(auto product:f.hessian)H_.values[product.destination]+=out[1][product.left]*out[1][product.right];
    for(size_t j=0;j<f.a_dest.size();++j)A_.values[f.a_dest[j]]=out[3][j];
  }
  if(!gradient_.allFinite()||!constraints_.allFinite()||!std::isfinite(cost_)||
     !std::all_of(H_.values.begin(),H_.values.end(),[](double v){return std::isfinite(v);})||
     !std::all_of(A_.values.begin(),A_.values.end(),[](double v){return std::isfinite(v);}))throw std::runtime_error("Non-finite SQP data");
}
double Controller::value(const Vector& at,const Eigen::Matrix<double,30,1>& p,Vector& constraints){
  double cost=0.;
  for(auto& f:layouts_){f.value->call({at.data()+f.column,p.data()});cost+=.5*f.value->dense(0).squaredNorm();Vector c=f.value->dense(1);constraints.segment(f.row,c.size())=c;}
  return cost;
}
State Controller::free(const State& x,double dt,const Joint& acc) const {
  State out=x;Eigen::Vector3d gravity(0,0,-9.81);
  out.head<3>()+=dt*x.segment<3>(3)+.5*dt*dt*gravity;out.segment<3>(3)+=dt*gravity;
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
void Controller::impact(State& x,Contact& contact){contact.impulse=apply_ball_impact(robot,x);}
std::pair<State,std::vector<Contact>> Controller::rollout(const State& x,double duration,const Joint& acc,double max_step){
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
  Matrix Y=Matrix::Zero(g::NZ,g::N+1);State state=x;Y.row(g::NX+9).setConstant(g::h);Y.row(g::NX+8).setConstant(.5);
  for(int k=0;k<g::N;++k){Y.col(k).head<g::NX>()=state;Y.col(k).segment<7>(g::NX)=robot.rnea(state.segment<7>(6),state.segment<7>(13),Joint::Zero());
    auto out=rollout(state,g::h,Joint::Zero(),.003);state=out.first;
    if(!out.second.empty()){Y(g::NX+7,k)=out.second[0].impulse/g::mass;Y(g::NX+8,k)=out.second[0].time/g::h;}}
  Y.col(g::N).head<g::NX>()=state;Y.col(g::N).segment<7>(g::NX)=robot.rnea(state.segment<7>(6),state.segment<7>(13),Joint::Zero());
  Y.col(0).segment<7>(g::NX)=torque;
  for(int k=0;k<=g::N;++k){Y(g::NX+10,k)=k*g::h;z_.segment<g::NZ>(k*g::NZ)=(Y.col(k)-map(g::offset)).cwiseQuotient(map(g::scale));}
  initialized_=true;
}
void Controller::set_physical_plan(){plan_.resize(g::NZ,g::N+1);for(int k=0;k<=g::N;++k)plan_.col(k)=z_.segment<g::NZ>(k*g::NZ).cwiseProduct(map(g::scale))+map(g::offset);}
void Controller::shift(double dt,const State& x,const Joint& torque){
  if(!initialized_)throw std::runtime_error("Cannot shift an uninitialized controller");
  set_physical_plan();Matrix old=plan_,Y=old;Y.row(g::NX+7).setZero();Y.row(g::NX+8).setConstant(.5);Eigen::Vector3d gravity(0,0,-9.81);
  for(int k=0;k<=g::N;++k){double t=k*g::h+dt;int i=source(t);Row row=old.col(i);double s=t-i*g::h;Joint acc=(old.col(std::min(i+1,g::N)).segment<7>(13)-row.segment<7>(13))/g::h;
    Y.col(k).head<3>()=row.head<3>()+s*row.segment<3>(3)+.5*s*s*gravity;Y.col(k).segment<3>(3)=row.segment<3>(3)+s*gravity;
    Y.col(k).segment<7>(6)=row.segment<7>(6)+s*row.segment<7>(13)+.5*s*s*acc;Y.col(k).segment<7>(13)=row.segment<7>(13)+s*acc;
    Y.col(k).segment<3>(20)=row.segment<3>(20);
    if(i<g::N){double event=row[g::NX+8]*g::h;
      if(s>=event-1e-12){Eigen::Vector3d impulse=old.col(i+1).segment<3>(3)-row.segment<3>(3)-g::h*gravity;Y.col(k).segment<3>(20)=old.col(i+1).segment<3>(20);Y.col(k).head<3>()+=(s-event)*impulse;Y.col(k).segment<3>(3)+=impulse;}}
    Y.col(k).segment<7>(g::NX)=i==g::N?old.col(g::N).segment<7>(g::NX).eval():((1-s/g::h)*row.segment<7>(g::NX)+(s/g::h)*old.col(i+1).segment<7>(g::NX)).eval();
  }
  for(int i=0;i<g::N;++i){double event=(i+old(g::NX+8,i))*g::h;int k=static_cast<int>(std::ceil((event+1e-12-dt)/g::h))-1;
    if(k>=0&&k<g::N){Y(g::NX+7,k)+=old(g::NX+7,i);Y(g::NX+8,k)=std::clamp((event-(k*g::h+dt))/g::h,0.,1.);}}
  Y.col(0).head<g::NX>()=x;Y.col(0).segment<7>(g::NX)=torque;
  Vector previous=dual_;dual_.setZero();
  for(int k=0;k<=g::N;++k){int i=source(k*g::h+dt);
    if(i<g::N){if(k<g::N)dual_.segment(g::NF+k*g::NC,g::NC)=previous.segment(g::NF+i*g::NC,g::NC);dual_.segment<g::NZ>(m_+k*g::NZ)=previous.segment<g::NZ>(m_+i*g::NZ);}}
  dual_.segment(m_,g::NP).setZero();
  for(int k=0;k<g::N;++k)dual_.segment(g::NF+k*g::NC+g::contact_begin,g::contact_end-g::contact_begin).setZero();
  for(int k=0;k<=g::N;++k){Y(g::NX+9,k)=g::h;Y(g::NX+10,k)=k*g::h;z_.segment<g::NZ>(k*g::NZ)=(Y.col(k)-map(g::offset)).cwiseQuotient(map(g::scale));}
  // Regenerate ball/contact seeds on the shifted joint arcs. On failure retain
  // the already shifted primal/dual, exactly as in the Python controller.
  State state=x;fallback_stage_=-1;
  for(int k=0;k<g::N;++k){Y.col(k).head<6>()=state.head<6>();Y.col(k).segment<3>(20)=state.segment<3>(20);Y(g::NX+7,k)=0;Y(g::NX+8,k)=.5;
    Joint acc=(Y.col(k+1).segment<7>(13)-Y.col(k).segment<7>(13))/g::h;
    try{auto out=rollout(Y.col(k).head<g::NX>(),g::h,acc,.01);state=out.first;if(guard_ball_rollout&&(state[2]<-.15||state[2]>1.5))throw std::runtime_error("Ball rollout left optimizer workspace");
      if(!out.second.empty()){Y(g::NX+7,k)=out.second[0].impulse/g::mass;Y(g::NX+8,k)=out.second[0].time/g::h;}}
    catch(const std::runtime_error& e){std::string message=e.what();if(message!="Unsupported resting contact"&&message!="Ball rollout left optimizer workspace")throw;fallback_stage_=k;return;}}
  Y.col(g::N).head<6>()=state.head<6>();Y.col(g::N).segment<3>(20)=state.segment<3>(20);
  for(int k=0;k<=g::N;++k)z_.segment<g::NZ>(k*g::NZ)=(Y.col(k)-map(g::offset)).cwiseQuotient(map(g::scale));
}
void Controller::set_warmstart(const double* z,const double* dual){z_=Eigen::Map<const Vector>(z,n_);dual_=Eigen::Map<const Vector>(dual,m_+n_);initialized_=true;}
double Controller::violation_sum(const Vector& c,const Vector& z) const{return (cl_-c).cwiseMax(0).sum()+(c-cu_).cwiseMax(0).sum()+(lower_-z).cwiseMax(0).sum()+(z-upper_).cwiseMax(0).sum();}
StepInfo Controller::step(const State& x,const Joint& torque,double elapsed){
  auto start=Clock::now();StepInfo info;info.evaluator_threads=evaluator_threads_;if(!x.allFinite()||!torque.allFinite()||!std::isfinite(elapsed)||elapsed<0)throw std::runtime_error("Invalid controller input");
  if(!initialized_)initialize(x,torque);else if(elapsed>0)shift(elapsed,x,torque);info.shift_ms=ms(start);info.fallback_stage=fallback_stage_;
  if(!z_.allFinite()||(lower_-z_).cwiseMax(0).maxCoeff()>100||(z_-upper_).cwiseMax(0).maxCoeff()>100)throw std::runtime_error("Warmstart diverged outside model range");
  Eigen::Matrix<double,30,1> p;p<<x,torque;auto tic=Clock::now();linearize(z_,p);q_=gradient_;
  for(int c=0;c<n_;++c)for(int j=H_.columns[c];j<H_.columns[c+1];++j){int r=H_.rows[j];q_[r]-=H_.values[j]*z_[c];if(r!=c)q_[c]-=H_.values[j]*z_[r];}
  Vector az=Vector::Zero(m_);
  for(int c=0;c<n_;++c)for(int j=A_.columns[c];j<A_.columns[c+1];++j)if(A_.rows[j]<m_)az[A_.rows[j]]+=A_.values[j]*z_[c];
  l_.head(m_)=cl_-constraints_+az;u_.head(m_)=cu_-constraints_+az;
  for(int i=0;i<n_;++i){double center=std::clamp(z_[i],lower_[i],upper_[i]);int j=i%g::NZ;bool unlimited=(j>=13&&j<20)||(j>=g::NX&&j<g::NX+7);
    l_[m_+i]=unlimited?lower_[i]:std::max(lower_[i],center-g::trust);u_[m_+i]=unlimited?upper_[i]:std::min(upper_[i],center+g::trust);}
  for(int i=0;i<g::NP;++i){double measured=(p[i]-g::offset[i])/g::scale[i];l_[m_+i]=std::min(l_[m_+i],measured);u_[m_+i]=std::max(u_[m_+i],measured);}
  // OSQP uses finite infinity sentinels internally.
  for(int i=0;i<l_.size();++i){if(std::isinf(l_[i]))l_[i]=-OSQP_INFTY;if(std::isinf(u_[i]))u_[i]=OSQP_INFTY;}
  info.data_ms=ms(tic);tic=Clock::now();
  if(!solver_){OSQPSettings settings;osqp_set_default_settings(&settings);settings.verbose=0;settings.max_iter=g::osqp_iters;settings.check_termination=0;settings.polishing=0;settings.warm_starting=1;settings.adaptive_rho=0;settings.rho=g::rho;settings.eps_abs=1e-4;settings.eps_rel=1e-4;
    osqp_h_=OSQPCscMatrix_new(n_,n_,H_.values.size(),H_.values.data(),H_.rows.data(),H_.columns.data());
    osqp_a_=OSQPCscMatrix_new(m_+n_,n_,A_.values.size(),A_.values.data(),A_.rows.data(),A_.columns.data());
    if(osqp_setup(&solver_,osqp_h_,q_.data(),osqp_a_,l_.data(),u_.data(),m_+n_,n_,&settings))throw std::runtime_error("OSQP setup failed");
  // Match the reference wrapper: update vectors before matrices. Matrix updates
  // rescale the QP, so reversing these calls changes a fixed-budget solve.
  }else{if(osqp_update_data_vec(solver_,q_.data(),l_.data(),u_.data())||osqp_update_data_mat(solver_,H_.values.data(),nullptr,H_.values.size(),A_.values.data(),nullptr,A_.values.size()))throw std::runtime_error("OSQP data update failed");}
  info.warm_dual_nonzeros=(dual_.array()!=0).count();
  if(osqp_warm_start(solver_,z_.data(),dual_.data())||osqp_solve(solver_))throw std::runtime_error("OSQP solve failed");
  info.qp_ms=ms(tic);info.iterations=solver_->info->iter;info.status=solver_->info->status_val;info.primal_residual=solver_->info->prim_res;info.dual_residual=solver_->info->dual_res;tic=Clock::now();
  double merit0=cost_+1000*g::objective_scale*violation_sum(constraints_,z_);
  Vector candidate=Eigen::Map<const Vector>(solver_->solution->x,n_);
  if(candidate.allFinite()&&candidate.cwiseAbs().maxCoeff()<1e6){Vector dz=candidate-z_;for(double alpha:{1.,.5,.25,.125,.0625}){trial_=z_+alpha*dz;double objective=value(trial_,p,trial_constraints_);double merit=objective+1000*g::objective_scale*violation_sum(trial_constraints_,trial_);
      if(merit<=merit0+1e-8){z_=trial_;dual_=Eigen::Map<const Vector>(solver_->solution->y,m_+n_);info.alpha=alpha;break;}}}
  info.cost=value(z_,p,trial_constraints_);info.violation=std::max((cl_-trial_constraints_).cwiseMax(0).maxCoeff(),(trial_constraints_-cu_).cwiseMax(0).maxCoeff());info.bound_violation=std::max((lower_-z_).cwiseMax(0).maxCoeff(),(z_-upper_).cwiseMax(0).maxCoeff());
  set_physical_plan();Joint desired=torque+(plan_.col(1).segment<7>(g::NX)-plan_.col(0).segment<7>(g::NX))*g::control_dt/g::h;
  for(int j=0;j<7;++j)target_[j]=std::clamp(desired[j],std::max(-g::torque_limit[j],torque[j]-g::torque_rate_limit[j]*g::control_dt),std::min(g::torque_limit[j],torque[j]+g::torque_rate_limit[j]*g::control_dt));
  info.clipping=(target_-desired).cwiseAbs().maxCoeff();info.merit_ms=ms(tic);info.total_ms=ms(start);return info;
}
}

#pragma once
#include <casadi/casadi.hpp>
#include <osqp.h>
#include <Eigen/Dense>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace juggling {
using Vector=Eigen::VectorXd;
using Matrix=Eigen::MatrixXd;
using State=Eigen::Matrix<double,23,1>;
using Joint=Eigen::Matrix<double,7,1>;
using Row=Eigen::Matrix<double,34,1>;

class Evaluation {
 public:
  Evaluation(const std::string& directory,const std::string& name,bool compiled=true);
  ~Evaluation();
  Evaluation(const Evaluation&)=delete;
  void call(const std::vector<const double*>& inputs);
  Matrix dense(int output) const;
  casadi::Function function;
  std::vector<std::vector<double>> output;
 private:
  std::vector<const double*> args_;
  std::vector<double*> results_;
  std::vector<casadi_int> iw_;
  std::vector<double> work_;
  int memory_;
};

struct Sparse {
  std::vector<OSQPInt> columns,rows;
  std::vector<double> values;
};
struct StepInfo {
  double total_ms=0,shift_ms=0,data_ms=0,qp_ms=0,merit_ms=0;
  double cost=0,violation=0,bound_violation=0,alpha=0,primal_residual=0,dual_residual=0,clipping=0;
  int sqp_iterations=1,iterations=0,status=0,warm_dual_nonzeros=0,fallback_stage=-1,evaluator_threads=1;
};
struct Geometry { double gap,radial,speed; Eigen::Vector3d normal; };
struct Contact { double time=0,impulse=0; };

class RobotModel {
 public:
  explicit RobotModel(const std::string& directory,bool compiled=true);
  Matrix mass(const Joint& q);
  Joint bias(const Joint& q,const Joint& dq);
  Joint rnea(const Joint& q,const Joint& dq,const Joint& ddq);
  Geometry geometry(const State& x);
  Eigen::Vector3d normal(const Joint& q);
  std::array<Matrix,5> contact_fk(const Joint& q);
 private:
  Evaluation mass_,bias_,rnea_,fk_,contact_;
};

double apply_ball_impact(RobotModel& robot,State& x);

class EvaluationPool;

class Controller {
 public:
  explicit Controller(const std::string& directory,bool compiled=true);
  ~Controller();
  StepInfo step(const State& x,const Joint& torque,double elapsed);
  void initialize(const State& x,const Joint& torque);
  void shift(double dt,const State& x,const Joint& torque);
  void linearize(const Vector& at,const Eigen::Matrix<double,30,1>& parameter);
  double value(const Vector& at,const Eigen::Matrix<double,30,1>& parameter,Vector& constraints);
  void set_warmstart(const double* z,const double* dual);
  const Vector& solution() const { return z_; }
  const Vector& dual() const { return dual_; }
  const Matrix& physical_plan() const { return plan_; }
  const Joint& target() const { return target_; }
  const Sparse& hessian() const { return H_; }
  const Sparse& constraint_matrix() const { return A_; }
  const Vector& gradient() const { return gradient_; }
  const Vector& constraints() const { return constraints_; }
  double cost() const { return cost_; }
  int variables() const { return n_; }
  int nonlinear_rows() const { return m_; }
  RobotModel robot;
  bool guard_ball_rollout=true;
 private:
  struct Product { int destination,left,right; };
  struct Layout {
    Evaluation *data,*value;
    int column,row;
    std::vector<Product> hessian;
    std::vector<int> a_dest;
  };
  Evaluation first_data_,stage_data_,last_data_,first_value_,stage_value_,last_value_;
  std::vector<std::unique_ptr<Evaluation>> stage_evaluators_;
  std::vector<Layout> layouts_;
  std::unique_ptr<EvaluationPool> pool_;
  int evaluator_threads_=1;
  Sparse H_,A_;
  std::vector<int> diagonal_;
  int n_,m_,fallback_stage_=-1;
  Vector z_,dual_,lower_,upper_,cl_,cu_,gradient_,constraints_,q_,l_,u_,trial_,trial_constraints_;
  Matrix plan_;
  Joint target_;
  double cost_=0;
  bool initialized_=false;
  OSQPSolver* solver_=nullptr;
  OSQPCscMatrix *osqp_h_=nullptr,*osqp_a_=nullptr;
  void build_pattern();
  State free(const State& x,double dt,const Joint& acceleration) const;
  std::pair<State,std::vector<Contact>> rollout(const State& x,double duration,const Joint& acceleration,double max_step);
  void impact(State& x,Contact& contact);
  double violation_sum(const Vector& constraints,const Vector& z) const;
  void set_physical_plan();
};
}

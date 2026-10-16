#pragma once
#include <casadi/casadi.hpp>
#include <osqp.h>
#include <Eigen/Dense>
#include <array>
#include <iosfwd>
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
  int sqp_iterations=0,iterations=0,status=0,warm_dual_nonzeros=0,fallback_stage=-1,evaluator_threads=1;
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
class ResidualLift;

class Controller {
 public:
  explicit Controller(const std::string& directory,bool compiled=true);
  ~Controller();
  StepInfo step(const State& x,const Joint& torque,double elapsed);
  void initialize(const State& x,const Joint& torque);
  // References may move between solves without resetting the warm start.
  // Coordinates stay in the original model frame;
  // only cost references move. Constraints/kinematics are unchanged.
  void set_reference(const Eigen::Vector3d& face_offset,const Joint& posture);
  // Added rebound velocity in nominal paddle axes, normalized by j/8.
  // Private to this controller; call between solves on its owning thread.
  void set_contact_bias(const Eigen::Vector3d& bias_per_j8_local);
  void shift(double dt,const State& x,const Joint& torque);
  void linearize(const Vector& at,const Eigen::Matrix<double,30,1>& parameter);
  double value(const Vector& at,const Eigen::Matrix<double,30,1>& parameter,Vector& constraints);
  void set_warmstart(const double* z,const double* dual);
  const Vector& solution() const { return z_; }
  const Vector& dual() const { return dual_; }
  // Decoded physical units. A kinematic-tail profile keeps acceleration in
  // control slots at nodes P+1..N; executable prefix slots remain torques.
  const Matrix& physical_plan() const { return plan_; }
  const Joint& target() const { return target_; }
  const Sparse& hessian() const { return H_; }
  const Sparse& constraint_matrix() const { return A_; }
  const Vector& gradient() const { return gradient_; }
  const Vector& constraints() const { return constraints_; }
  double cost() const { return cost_; }
  int variables() const { return n_; }
  int nonlinear_rows() const { return m_; }
  std::array<long long,12> qp_diagnostics() const;
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
  const ResidualLift& residual_lift() const { return *lift_; }
  void prepare_lifted_qp(const Vector& at,const Eigen::Matrix<double,30,1>& parameter);
#endif
  RobotModel robot;
  bool guard_ball_rollout=true;
 private:
  struct Product { int destination,left,right; };
  struct Layout {
    Evaluation *data,*value;
    int column,row;
    double step;
    std::vector<Product> hessian;
    std::vector<int> a_dest;
    int restitution_cap_row=-1;
    // {destination in assembled A, source in residual Jacobian output}.
    std::vector<std::array<int,2>> restitution_cap_a;
    int edge_residual_dest=-1;
    // {destination in existing JR row 52, source in correction Jacobian}.
    std::vector<std::array<int,2>> edge_jacobian_dest;
    // Existing ball-constraint output slots and JC destinations for bias.
    std::array<int,6> bias_constraint_dest{{-1,-1,-1,-1,-1,-1}};
    std::vector<std::array<int,2>> bias_jacobian_dest;
    // Contact-plane corrections reuse the original factor sparsity. Entries
    // map {destination output slot/local dense row, helper sparse slot}.
    std::vector<std::array<int,2>> plane_residual_dest,plane_residual_rows;
    std::vector<std::array<int,2>> plane_constraint_dest,plane_constraint_rows;
    std::vector<std::array<int,2>> plane_residual_jacobian_dest,plane_constraint_jacobian_dest;
    int radial_constraint_row=-1,radial_constraint_dest=-1,radial_impulse_jacobian_dest=-1;
    std::vector<int> radial_jacobian_dest;
    // Optional fixed left transform of seven dynamics equality rows. Each
    // Jacobian entry contains all seven existing slots for one variable.
    std::array<int,7> midpoint_constraint_rows{},midpoint_constraint_dest{};
    std::vector<std::array<int,7>> midpoint_jacobian_dest;
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
    int lift_block=-1;
#endif
  };
  struct SeedContext {
    struct Event {
      double lead=0.,radius=0.,impulse=0.;
      int stage=-1,contact_ordinal=0;
      bool edge_correction_bypassed=false;
      Eigen::Vector3d ball_contact=Eigen::Vector3d::Zero();
    };
    double radius=0.,minimum_lead=.3,stage_start=0.,first_contact_lead=0.;
    int stage=-1,contacts=0,max_contacts=1,all_contacts=0;
    std::array<Event,4> events{};
  };
  struct SeedMeritScreen {
    struct Point {
      bool evaluated=false;
      double cost=0.,merit=0.,constraint_l1=0.,constraint_max=0.,bound_l1=0.,bound_max=0.;
    };
    Point shifted,regenerated;
    double penalty=0.;
    int mode=2;
  };
  struct TailEndpointPenalty {
    int layout=-1,local_row=-1,row=-1,node=-1,joint=-1;
    double original_lower=0.,original_upper=0.,time_weight=0.,gain_squared=0.;
    std::vector<int> duplicate_rows;
    // {global normalized variable column, source slot in stage JC}.
    std::vector<std::array<int,2>> jacobian;
    std::vector<Product> hessian;
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
    int lift_block=-1;
#endif
  };
  struct FirstEventCenter {
    bool active=false,have_observation=false,has_seed_match=false;
    int stage=-1;
    size_t generation=0;
    double clock=0.,event_time=0.,lead=0.,tau=0.,seed_impulse=0.;
    double matched_event_time=0.,applied_refresh=0.;
    State previous_observation=State::Zero();
    Eigen::Vector3d target=Eigen::Vector3d::Zero();
    Eigen::Vector3d preferred_local=Eigen::Vector3d::Zero();
  };
  Evaluation first_data_,stage_data_,last_data_,first_value_,stage_value_,last_value_;
  std::unique_ptr<Evaluation> edge_prediction_data_,edge_prediction_value_;
  std::unique_ptr<Evaluation> contact_bias_data_,contact_bias_value_;
  std::unique_ptr<Evaluation> contact_plane_data_,contact_plane_value_;
  std::vector<std::unique_ptr<Evaluation>> stage_evaluators_,stage_values_;
  std::vector<Layout> layouts_;
  std::unique_ptr<EvaluationPool> pool_;
  int evaluator_threads_=1;
  Sparse H_,A_;
  std::vector<int> diagonal_;
  int n_,m_,fallback_stage_=-1;
  double restitution_cap_normalization_=1.;
  Vector z_,dual_,lower_,upper_,cl_,cu_,gradient_,constraints_,q_,l_,u_,trial_,trial_constraints_;
  Matrix plan_;
  Joint target_;
  double cost_=0;
  Eigen::Matrix<double,7,7> midpoint_transform_;
  void configure_midpoint_preconditioner();
  void transform_midpoint_data(Layout& layout);
  void transform_midpoint_value(const Layout& layout,Vector& constraints) const;
#ifdef JUGGLING_EXPERIMENTAL_RESIDUAL_LIFT
  std::unique_ptr<ResidualLift> lift_;
  bool assemble_normal_=true;
  std::vector<int> posture_lift_blocks_,terminal_lift_blocks_,center_lift_blocks_,landing_lift_blocks_;
  void configure_residual_lift();
  void linearize_lifted(const Vector& at,const Eigen::Matrix<double,30,1>& parameter);
#endif
  bool initialized_=false;
  std::unique_ptr<std::ofstream> seed_trace_;
  double seed_controller_time_=0.;
#ifdef JUGGLING_EXPERIMENTAL_GROUP_RHO
  double contact_rho_multiplier_=1.;
  std::vector<unsigned char> group_rho_selected_;
  std::vector<OSQPInt> group_rho_types_;
  std::vector<OSQPFloat> group_rho_values_;
  std::unique_ptr<std::ofstream> group_rho_trace_;
  size_t group_rho_updates_=0;
  void install_group_rho(int sqp);
#endif
  Eigen::Vector3d reference_offset_=Eigen::Vector3d::Zero();
  Eigen::Vector3d bias_per_j8_local_=Eigen::Vector3d::Zero();
  bool contact_bias_active_=false;
  Joint reference_posture_delta_=Joint::Zero();
  Eigen::Matrix<double,7,7> posture_projector_=Eigen::Matrix<double,7,7>::Zero();
  std::vector<int> posture_hessian_dest_;
  Eigen::Vector2d landing_target_=Eigen::Vector2d::Zero();
  Eigen::Vector3d landing_preferred_local_=Eigen::Vector3d::Zero();
  std::vector<double> landing_node_weights_;
  std::vector<int> landing_hessian_dest_;
  FirstEventCenter first_event_center_;
  std::vector<std::array<int,231>> first_event_center_hessian_dest_;
  std::unique_ptr<std::ofstream> first_event_center_trace_;
  size_t first_event_center_updates_=0,first_event_center_trace_records_=0;
  void configure_first_event_center();
  void freeze_first_event_center(const State& state,double elapsed);
  double first_event_center_cost(const Vector& at,bool assemble);
  void trace_first_event_center(const char* action,const char* reason,bool force);
  std::vector<TailEndpointPenalty> tail_endpoint_penalties_;
  std::vector<int> tail_endpoint_protected_rows_;
  int tail_endpoint_boundary_=-1;
  double tail_endpoint_elapsed_=0.,tail_endpoint_logged_debt_=0.;
  size_t tail_endpoint_trace_step_=0,tail_endpoint_trace_records_=0;
  std::unique_ptr<std::ofstream> tail_endpoint_trace_;
  void configure_tail_endpoint_elasticity();
  double tail_endpoint_regularization(const Vector& constraints,bool assemble);
  void trace_tail_endpoint_elasticity(const Vector& constraints,const StepInfo& info);
  void update_posture_projector();
  double posture_regularization(const Vector& at,bool assemble);
  double terminal_regularization(const Vector& at,bool assemble);
  void freeze_landing_guidance(const State& state);
  double landing_guidance(const Vector& at,bool assemble);
  void reference_residual(Vector& residual,double quadrature_weight=1.) const;
  void configure_radial_contact_scaling();
  // Optional draft only: both contact-mode seeds share one measured state and
  // frozen guidance. Presence does not change StepInfo or generated factors.
  int multistart_mode_=0,multistart_verify_=0;
  size_t multistart_trace_records_=0;
  bool multistart_pending_=false;
  Vector multistart_shifted_,multistart_regenerated_,multistart_dual_;
  SeedContext multistart_seed_;
  std::unique_ptr<std::ofstream> multistart_trace_;
  void configure_multistart();
  void solve_sqp(const Eigen::Matrix<double,30,1>& parameter,int count,StepInfo& info,int* qp_calls=nullptr);
  void solve_multistart(const Eigen::Matrix<double,30,1>& parameter,StepInfo& info);
  OSQPSolver* solver_=nullptr;
  OSQPCscMatrix *osqp_h_=nullptr,*osqp_a_=nullptr;
  void build_pattern();
  State free(const State& x,double dt,const Joint& acceleration) const;
  State free_ramp(const State& x,double dt,const Joint& acceleration,const Joint& jerk) const;
  std::pair<State,std::vector<Contact>> rollout(const State& x,double duration,const Joint& acceleration,double max_step,const Joint* end_acceleration=nullptr,SeedContext* seed=nullptr,int prediction_stage=-1);
  void trace_virtual_seed(const SeedContext& seed,bool applied,const char* outcome,const SeedMeritScreen* merit=nullptr);
  void try_arm_shift_seed(const State& x,const Joint& torque,const SeedContext& seed);
  void impact(State& x,Contact& contact,bool virtual_proposal=false,bool kinematic_tail=false);
  Geometry prediction_geometry(const State& x);
  std::array<Matrix,5> prediction_contact_fk(const Joint& q);
  Vector prediction_stage_constraints(const double* stage,const double* parameter);
  double violation_sum(const Vector& constraints,const Vector& z) const;
  void reconcile_kinematic_tail(Matrix& decoded_plan);
  void set_physical_plan();
};
}

#pragma once
#include "controller.hpp"
#include <utility>

namespace juggling {
// A numerical Gauss--Newton lift. Each block owns fixed Jacobian sparsity;
// inactive guidance/hinges contribute zero rows without changing the QP graph.
class ResidualLift {
 public:
  struct Block {
    int row=0,size=0;
    std::vector<std::pair<int,int>> entries; // {local row, physical column}
    std::vector<int> destination;
  };
  ResidualLift(int physical_variables,int physical_rows,double proximal);
  int add_block(int rows,const std::vector<std::pair<int,int>>& entries);
  int add_dense_block(int rows,const std::vector<int>& columns);
  void finalize(const Sparse& physical_a);
  void clear();
  void set_sparse(int block,const Vector& residual,const std::vector<double>& jacobian);
  void set_dense(int block,const Vector& residual,const Matrix& jacobian);
  void prepare(const Sparse& physical_a,const Vector& physical_l,const Vector& physical_u,
               const Vector& physical_z,const Vector& physical_dual);
  void install_link_rho(OSQPSolver* solver,double multiplier);
  int variables() const { return physical_variables_+residual.size(); }
  int rows() const { return physical_rows_+residual.size(); }
  const std::vector<Block>& blocks() const { return blocks_; }
  Sparse P,A;
  Vector residual,q,l,u,warm_primal,warm_dual;
 private:
  int physical_variables_,physical_rows_,residual_rows_=0;
  double proximal_;
  bool finalized_=false;
  std::vector<Block> blocks_;
  std::vector<int> physical_dest_;
  std::vector<OSQPInt> rho_type_scratch_;
  std::vector<OSQPFloat> rho_scratch_;
};
}

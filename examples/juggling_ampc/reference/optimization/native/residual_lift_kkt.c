/* Read-only diagnostics, confined to the pinned builtin OSQP/QDLDL ABI. */
#include "qdldl_interface.h"
#include <string.h>
_Static_assert(sizeof(OSQPFloat)==8 && sizeof(OSQPInt)==4, "Pinned OSQP scalar ABI mismatch");
_Static_assert(sizeof(QDLDL_float)==8 && sizeof(QDLDL_int)==4, "Pinned QDLDL scalar ABI mismatch");
long long juggling_osqp_factor_nnz(const OSQPSolver* solver){
  const qdldl_solver* factor;
  if(!solver||!solver->work||!solver->work->linsys_solver||
      solver->settings->linsys_solver!=OSQP_DIRECT_SOLVER)return -1;
  if(!solver->work->linsys_solver->name||strncmp(solver->work->linsys_solver->name(solver->work->linsys_solver),"QDLDL",5)!=0)return -1;
  factor=(const qdldl_solver*)solver->work->linsys_solver;
  if(!factor->L||!factor->L->p)return -1;
  return factor->L->p[factor->L->n]; // Stored strict lower triangle, unit diagonal implicit.
}

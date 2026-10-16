/* OSQP commit 1572ae068e9ce9ca723cf8223548ade1ff7acc29 only.
 * CMake verifies source headers, update implementation, and installed config.
 * Keep all private layout access confined to this optional translation unit. */
#include "osqp_group_rho.h"
#include "types.h"
#include "algebra_vector.h"
#include <math.h>
#include <float.h>

#if !defined(OSQP_ALGEBRA_BUILTIN) || !defined(OSQP_ENABLE_PROFILING) || \
    defined(OSQP_USE_FLOAT) || defined(OSQP_USE_LONG) || \
    defined(OSQP_EMBEDDED_MODE) || defined(OSQP_ENABLE_DERIVATIVES) || \
    defined(OSQP_PACK_SETTINGS) || defined(OSQP_ENABLE_INTERRUPT) || \
    defined(OSQP_ENABLE_PRINTING) || defined(OSQP_ENABLE_DEBUG) || \
    defined(OSQP_CUSTOM_MEMORY) || defined(OSQP_CODEGEN) || \
    defined(OSQP_ALGEBRA_MKL) || defined(OSQP_ALGEBRA_CUDA) || \
    defined(OSQP_PROFILER_ANNOTATIONS)
#error "Group-rho bridge requires the pinned builtin/double/int32/profiling OSQP configuration"
#endif
_Static_assert(sizeof(OSQPFloat) == 8, "OSQPFloat ABI mismatch");
_Static_assert(sizeof(OSQPInt) == 4, "OSQPInt ABI mismatch");

OSQPInt juggling_osqp_install_group_rho(
    OSQPSolver* solver, OSQPInt rows, const unsigned char* selected,
    OSQPFloat multiplier, OSQPInt* type_scratch, OSQPFloat* rho_scratch,
    JugglingGroupRhoReport* report) {
  OSQPInt i;
  OSQPWorkspace* work;
  OSQPFloat base;
  JugglingGroupRhoReport result = {0, 0, 0, 0., DBL_MAX, 0., DBL_MAX, 0.};
  if (multiplier == 1.) return 0;
  if (!solver || !solver->settings || !solver->work || !selected ||
      !type_scratch || !rho_scratch || !isfinite(multiplier) ||
      multiplier < 1. || multiplier > 1000.) return 1;
  work = solver->work;
  if (!work->data || work->data->m != rows || !work->rho_vec ||
      !work->rho_inv_vec || !work->constr_type || !work->linsys_solver ||
      !work->linsys_solver->update_rho_vec || !solver->settings->rho_is_vec ||
      solver->settings->adaptive_rho != OSQP_ADAPTIVE_RHO_UPDATE_DISABLED ||
      solver->settings->linsys_solver != OSQP_DIRECT_SOLVER) return 2;
  base = solver->settings->rho;
  if (!isfinite(base) || base < OSQP_RHO_MIN || base > OSQP_RHO_MAX) return 2;
  result.base_rho = base;
  OSQPVectori_to_raw(type_scratch, work->constr_type);
  for (i = 0; i < rows; ++i) {
    /* Reconstruct OSQP's FULL policy, not a cumulative modification of a
     * vector which update_data_vec may already have reset behind our back. */
    const OSQPInt type = type_scratch[i];
    if (type < -1 || type > 1) return 2;
    rho_scratch[i] = type < 0 ? OSQP_RHO_MIN :
        (type > 0 ? OSQP_RHO_EQ_OVER_RHO_INEQ * base : base);
    if (!selected[i]) continue;
    if (type < 0) { ++result.selected_free; continue; }
    if (type > 0) { ++result.selected_equalities; continue; }
    rho_scratch[i] = fmin(OSQP_RHO_MAX, base * multiplier);
    ++result.selected_inequalities;
    result.selected_rho_min = fmin(result.selected_rho_min, rho_scratch[i]);
    result.selected_rho_max = fmax(result.selected_rho_max, rho_scratch[i]);
    if (report) {
      /* OSQP's penalty is in scaled coordinates: rho_original=rho*E^2/c. */
      const OSQPFloat e = work->scaling ? OSQPVectorf_data(work->scaling->E)[i] : 1.;
      const OSQPFloat c = work->scaling ? work->scaling->c : 1.;
      const OSQPFloat effective = rho_scratch[i] * e * e / c;
      result.effective_rho_min = fmin(result.effective_rho_min, effective);
      result.effective_rho_max = fmax(result.effective_rho_max, effective);
    }
  }
  OSQPVectorf_from_raw(work->rho_vec, rho_scratch);
  OSQPVectorf_ew_reciprocal(work->rho_inv_vec, work->rho_vec);
  /* Always synchronize the linear solver's inverse vector, KKT diagonal and
   * numeric factorization. Unchanged constraint types are NOT sufficient. */
  if (work->linsys_solver->update_rho_vec(work->linsys_solver, work->rho_vec, base)) return 3;
  if (!result.selected_inequalities) {
    result.selected_rho_min = result.effective_rho_min = 0.;
  }
  if (report) *report = result;
  return 0;
}

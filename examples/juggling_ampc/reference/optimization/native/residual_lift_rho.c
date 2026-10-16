/* OSQP commit 1572ae068e9ce9ca723cf8223548ade1ff7acc29 only.
 * CMake must apply osqp_group_rho.cmake's source/config/archive verification.
 * This changes ADMM penalties only; QP data and primal/dual iterates stay put. */
#include "residual_lift_rho.h"
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
#error "Lift-link rho requires the pinned builtin/double/int32/profiling OSQP configuration"
#endif
_Static_assert(sizeof(OSQPFloat) == 8, "OSQPFloat ABI mismatch");
_Static_assert(sizeof(OSQPInt) == 4, "OSQPInt ABI mismatch");

OSQPInt juggling_osqp_install_lift_link_rho(
    OSQPSolver* solver, OSQPInt physical_rows, OSQPInt total_rows,
    OSQPFloat multiplier, OSQPInt* type_scratch, OSQPFloat* rho_scratch,
    JugglingLiftLinkRhoReport* report) {
  OSQPInt i;
  OSQPWorkspace* work;
  OSQPFloat base;
  JugglingLiftLinkRhoReport result = {0, 0, 0., DBL_MAX, 0., DBL_MAX, 0.};
  if (multiplier == 1.) return 0;
  if (!solver || !solver->settings || !solver->work || !type_scratch ||
      !rho_scratch || physical_rows <= 0 || physical_rows >= total_rows ||
      !isfinite(multiplier) || multiplier < 1. || multiplier > 1000.) return 1;
  work = solver->work;
  if (!work->data || work->data->m != total_rows || !work->rho_vec ||
      !work->rho_inv_vec || !work->constr_type || !work->linsys_solver ||
      !work->linsys_solver->update_rho_vec || !solver->settings->rho_is_vec ||
      solver->settings->adaptive_rho != OSQP_ADAPTIVE_RHO_UPDATE_DISABLED ||
      solver->settings->linsys_solver != OSQP_DIRECT_SOLVER) return 2;
  base = solver->settings->rho;
  if (!isfinite(base) || base < OSQP_RHO_MIN || base > OSQP_RHO_MAX) return 2;
  result.physical_rows = physical_rows;
  result.link_rows = total_rows - physical_rows;
  result.base_rho = base;
  OSQPVectori_to_raw(type_scratch, work->constr_type);
  for (i = 0; i < total_rows; ++i) {
    const OSQPInt type = type_scratch[i];
    if (type < -1 || type > 1 || (i >= physical_rows && type != 1)) return 2;
    /* Bounds updates may restore the ordinary policy even without a type
     * change. Rebuild it, then multiply only the added equalities once. */
    rho_scratch[i] = type < 0 ? OSQP_RHO_MIN :
        (type > 0 ? OSQP_RHO_EQ_OVER_RHO_INEQ * base : base);
    if (i < physical_rows) continue;
    rho_scratch[i] = fmin(OSQP_RHO_MAX, rho_scratch[i] * multiplier);
    result.link_rho_min = fmin(result.link_rho_min, rho_scratch[i]);
    result.link_rho_max = fmax(result.link_rho_max, rho_scratch[i]);
    if (report) {
      const OSQPFloat e = work->scaling ? OSQPVectorf_data(work->scaling->E)[i] : 1.;
      const OSQPFloat c = work->scaling ? work->scaling->c : 1.;
      const OSQPFloat effective = rho_scratch[i] * e * e / c;
      result.effective_link_rho_min = fmin(result.effective_link_rho_min, effective);
      result.effective_link_rho_max = fmax(result.effective_link_rho_max, effective);
    }
  }
  OSQPVectorf_from_raw(work->rho_vec, rho_scratch);
  OSQPVectorf_ew_reciprocal(work->rho_inv_vec, work->rho_vec);
  /* Synchronize the cached inverse penalties and numeric KKT factorization,
   * including when the preceding bounds update retained all row types. */
  if (work->linsys_solver->update_rho_vec(work->linsys_solver, work->rho_vec, base)) return 3;
  if (report) *report = result;
  return 0;
}

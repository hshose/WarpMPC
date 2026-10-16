#pragma once
#include <osqp.h>

/* Private, pinned-OSQP bridge; never part of the supported controller C API. */
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  OSQPInt selected_inequalities, selected_equalities, selected_free;
  OSQPFloat base_rho, selected_rho_min, selected_rho_max;
  OSQPFloat effective_rho_min, effective_rho_max;
} JugglingGroupRhoReport;

/* Called after setup / ALL normal data updates, never on exact continuation.
 * Scratch arrays and selected mask have solver->work->data->m entries.
 * multiplier=1 is an exact no-op, including no private workspace access. */
OSQPInt juggling_osqp_install_group_rho(
    OSQPSolver* solver, OSQPInt rows, const unsigned char* selected,
    OSQPFloat multiplier, OSQPInt* type_scratch, OSQPFloat* rho_scratch,
    JugglingGroupRhoReport* report);
#ifdef __cplusplus
}
#endif

#pragma once
#include <osqp.h>

/* Optional, private bridge for the same pinned OSQP ABI as osqp_group_rho.c. */
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  OSQPInt physical_rows, link_rows;
  OSQPFloat base_rho, link_rho_min, link_rho_max;
  OSQPFloat effective_link_rho_min, effective_link_rho_max;
} JugglingLiftLinkRhoReport;

/* Original physical rows precede the contiguous residual-link equalities.
 * Reconstruct the FULL vector after setup / ordinary data updates; never call
 * on an exact ADMM continuation. Scratch arrays have total_rows elements.
 * multiplier=1 returns before accessing any solver or scratch memory. */
OSQPInt juggling_osqp_install_lift_link_rho(
    OSQPSolver* solver, OSQPInt physical_rows, OSQPInt total_rows,
    OSQPFloat multiplier, OSQPInt* type_scratch, OSQPFloat* rho_scratch,
    JugglingLiftLinkRhoReport* report);
#ifdef __cplusplus
}
#endif

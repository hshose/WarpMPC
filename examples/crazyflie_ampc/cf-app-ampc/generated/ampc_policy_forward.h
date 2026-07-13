/* Auto-generated AMPC policy forward pass. */
#ifndef AMPC_POLICY_FORWARD_H_
#define AMPC_POLICY_FORWARD_H_

#include "ampc_policy_data.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    ampc_policy_float_t input_scaled[AMPC_POLICY_INPUT_DIM];
    ampc_policy_float_t layer_a[AMPC_POLICY_MAX_LAYER_SIZE];
    ampc_policy_float_t layer_b[AMPC_POLICY_MAX_LAYER_SIZE];
} ampc_policy_workspace_t;

int ampc_policy_forward(
    const ampc_policy_float_t input[AMPC_POLICY_INPUT_DIM],
    ampc_policy_float_t output[AMPC_POLICY_OUTPUT_DIM],
    ampc_policy_workspace_t *workspace
);

#ifdef __cplusplus
}
#endif

#endif /* AMPC_POLICY_FORWARD_H_ */

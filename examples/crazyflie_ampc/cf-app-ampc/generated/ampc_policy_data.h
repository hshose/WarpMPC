/* Auto-generated AMPC policy data. */
#ifndef AMPC_POLICY_DATA_H_
#define AMPC_POLICY_DATA_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef AMPC_POLICY_FLOAT_TYPE
#define AMPC_POLICY_FLOAT_TYPE float
#endif
typedef AMPC_POLICY_FLOAT_TYPE ampc_policy_float_t;

#define AMPC_POLICY_INPUT_DIM 12u
#define AMPC_POLICY_OUTPUT_DIM 4u
#define AMPC_POLICY_NUM_LAYERS 4u
#define AMPC_POLICY_MAX_LAYER_SIZE 32u
#define AMPC_POLICY_MAX_INPUT_SIZE 32u

typedef enum {
    AMPC_POLICY_ACTIVATION_LINEAR = 0,
    AMPC_POLICY_ACTIVATION_RELU = 1,
    AMPC_POLICY_ACTIVATION_LEAKY_RELU = 2,
    AMPC_POLICY_ACTIVATION_TANH = 3,
    AMPC_POLICY_ACTIVATION_GELU = 4,
    AMPC_POLICY_ACTIVATION_SILU = 5
} ampc_policy_activation_t;

typedef struct {
    uint16_t input_dim;
    uint16_t output_dim;
    ampc_policy_activation_t activation;
    ampc_policy_float_t negative_slope;
    const ampc_policy_float_t *weights;  /* row-major: output_dim x input_dim */
    const ampc_policy_float_t *biases;   /* output_dim */
} ampc_policy_layer_t;

extern const ampc_policy_float_t ampc_policy_x_mean[AMPC_POLICY_INPUT_DIM];
extern const ampc_policy_float_t ampc_policy_x_std[AMPC_POLICY_INPUT_DIM];
extern const ampc_policy_float_t ampc_policy_y_mean[AMPC_POLICY_OUTPUT_DIM];
extern const ampc_policy_float_t ampc_policy_y_std[AMPC_POLICY_OUTPUT_DIM];
extern const ampc_policy_float_t ampc_policy_y_clip_low[AMPC_POLICY_OUTPUT_DIM];
extern const ampc_policy_float_t ampc_policy_y_clip_high[AMPC_POLICY_OUTPUT_DIM];
extern const ampc_policy_layer_t ampc_policy_layers[AMPC_POLICY_NUM_LAYERS];

#ifdef __cplusplus
}
#endif

#endif /* AMPC_POLICY_DATA_H_ */

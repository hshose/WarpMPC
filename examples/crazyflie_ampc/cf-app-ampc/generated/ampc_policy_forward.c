/* Auto-generated plain-C AMPC policy forward pass. */
#include "ampc_policy_forward.h"

#include <math.h>
#include <stddef.h>

static ampc_policy_float_t ampc_policy_activate(ampc_policy_float_t x, ampc_policy_activation_t activation, ampc_policy_float_t negative_slope) {
    switch (activation) {
        case AMPC_POLICY_ACTIVATION_RELU:
            return x > (ampc_policy_float_t)0 ? x : (ampc_policy_float_t)0;
        case AMPC_POLICY_ACTIVATION_LEAKY_RELU:
            return x >= (ampc_policy_float_t)0 ? x : negative_slope * x;
        case AMPC_POLICY_ACTIVATION_TANH:
            return (ampc_policy_float_t)tanh((double)x);
        case AMPC_POLICY_ACTIVATION_GELU:
            return (ampc_policy_float_t)(0.5 * (double)x * (1.0 + erf((double)x * 0.70710678118654752440)));
        case AMPC_POLICY_ACTIVATION_SILU:
            return (ampc_policy_float_t)((double)x / (1.0 + exp(-(double)x)));
        case AMPC_POLICY_ACTIVATION_LINEAR:
        default:
            return x;
    }
}

int ampc_policy_forward(
    const ampc_policy_float_t input[AMPC_POLICY_INPUT_DIM],
    ampc_policy_float_t output[AMPC_POLICY_OUTPUT_DIM],
    ampc_policy_workspace_t *workspace
) {
    if (input == NULL || output == NULL || workspace == NULL) {
        return -1;
    }

    for (uint16_t i = 0u; i < AMPC_POLICY_INPUT_DIM; ++i) {
        workspace->input_scaled[i] = (input[i] - ampc_policy_x_mean[i]) / ampc_policy_x_std[i];
    }

    const ampc_policy_float_t *layer_input = workspace->input_scaled;
    ampc_policy_float_t *buffers[2] = {workspace->layer_a, workspace->layer_b};
    uint16_t buffer_index = 0u;

    for (uint16_t layer_index = 0u; layer_index < AMPC_POLICY_NUM_LAYERS; ++layer_index) {
        const ampc_policy_layer_t *layer = &ampc_policy_layers[layer_index];
        ampc_policy_float_t *layer_output = buffers[buffer_index];

        for (uint16_t row = 0u; row < layer->output_dim; ++row) {
            ampc_policy_float_t acc = layer->biases[row];
            const ampc_policy_float_t *weight_row = &layer->weights[(uint32_t)row * (uint32_t)layer->input_dim];
            for (uint16_t col = 0u; col < layer->input_dim; ++col) {
                acc += weight_row[col] * layer_input[col];
            }
            layer_output[row] = ampc_policy_activate(acc, layer->activation, layer->negative_slope);
        }

        layer_input = layer_output;
        buffer_index = (uint16_t)(1u - buffer_index);
    }

    for (uint16_t i = 0u; i < AMPC_POLICY_OUTPUT_DIM; ++i) {
        ampc_policy_float_t y = layer_input[i] * ampc_policy_y_std[i] + ampc_policy_y_mean[i];
        if (y < ampc_policy_y_clip_low[i]) {
            y = ampc_policy_y_clip_low[i];
        }
        if (y > ampc_policy_y_clip_high[i]) {
            y = ampc_policy_y_clip_high[i];
        }
        output[i] = y;
    }

    return 0;
}

/* Auto-generated AMPC policy test data. */
#ifndef AMPC_POLICY_TEST_DATA_H_
#define AMPC_POLICY_TEST_DATA_H_

#include "ampc_policy_data.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AMPC_POLICY_TEST_CASE_COUNT 64u

typedef struct {
    ampc_policy_float_t input[AMPC_POLICY_INPUT_DIM];
    ampc_policy_float_t output[AMPC_POLICY_OUTPUT_DIM];
} ampc_policy_test_case_t;

extern const ampc_policy_test_case_t ampc_policy_test_cases[AMPC_POLICY_TEST_CASE_COUNT];

#ifdef __cplusplus
}
#endif

#endif /* AMPC_POLICY_TEST_DATA_H_ */

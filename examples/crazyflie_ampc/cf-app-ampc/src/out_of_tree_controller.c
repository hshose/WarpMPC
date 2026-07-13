/**
 * Out-of-tree AMPC neural controller for Crazyflie.
 *
 * The generated policy evaluates the 12-dimensional AMPC state
 * [position error, Euler attitude error in radians, velocity, angular velocity] and
 * returns the four SQP action components used by benchmarks/problems/crazyflie_sqp.py.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "app.h"
#include "controller.h"
#include "platform_defaults.h"

#include "FreeRTOS.h"
#include "task.h"

#define DEBUG_MODULE "AMPCTRL"
#include "debug.h"

#include "log.h"
#include "param.h"
#include "pm.h"
#include "usec_time.h"

#include "ampc_policy_forward.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define AMPC_GRAVITY_ACCEL 9.8066f
#define AMPC_ACTION_SCALE_MX 1.0e-2f
#define AMPC_ACTION_SCALE_MY 1.0e-2f
#define AMPC_ACTION_SCALE_MZ 1.0e-3f
#define AMPC_TOTAL_THRUST_MAX_N 0.72f
#define AMPC_TORQUE_XY_MAX_NM 0.02f
#define AMPC_TORQUE_Z_MAX_NM 0.004f

#if AMPC_POLICY_INPUT_DIM != 12u
#error "Crazyflie AMPC controller expects AMPC_POLICY_INPUT_DIM == 12"
#endif

#if AMPC_POLICY_OUTPUT_DIM != 4u
#error "Crazyflie AMPC controller expects AMPC_POLICY_OUTPUT_DIM == 4"
#endif

static ampc_policy_workspace_t g_policy_workspace;
static bool g_controller_initialized = false;

static uint16_t g_controller_rate_hz = 500;
static uint8_t g_clip_enabled = 1;
static uint8_t g_bounds_enabled = 1;

static uint32_t g_nn_last_us = 0;
static uint32_t g_nn_eval_failures = 0;
static uint64_t g_timing_sum_nn_us = 0;
static uint32_t g_timing_count_nn = 0;
static uint32_t g_timing_last_print_ms = 0;

static float g_log_pos_x = 0.0f;
static float g_log_pos_y = 0.0f;
static float g_log_pos_z = 0.0f;
static float g_log_phi_x = 0.0f;
static float g_log_phi_y = 0.0f;
static float g_log_phi_z = 0.0f;
static float g_log_vel_x = 0.0f;
static float g_log_vel_y = 0.0f;
static float g_log_vel_z = 0.0f;
static float g_log_u0 = 0.0f;
static float g_log_u1 = 0.0f;
static float g_log_u2 = 0.0f;
static float g_log_u3 = 0.0f;
static float g_log_thrust = 0.0f;

typedef struct {
    float x;
    float y;
    float z;
    float w;
} ampc_quat_t;

static float clampf_local(float value, float lower, float upper) {
    if (value < lower) {
        return lower;
    }
    if (value > upper) {
        return upper;
    }
    return value;
}

static float wrap_pi(float angle) {
    while (angle > (float)M_PI) {
        angle -= 2.0f * (float)M_PI;
    }
    while (angle < -(float)M_PI) {
        angle += 2.0f * (float)M_PI;
    }
    return angle;
}

static float deg_to_rad(float degrees) {
    return degrees * ((float)M_PI / 180.0f);
}

static ampc_quat_t quat_make(float x, float y, float z, float w) {
    const ampc_quat_t q = {x, y, z, w};
    return q;
}

static ampc_quat_t quat_identity(void) {
    return quat_make(0.0f, 0.0f, 0.0f, 1.0f);
}

static ampc_quat_t quat_normalize(ampc_quat_t q) {
    const float norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    if (norm_sq <= 1.0e-12f) {
        return quat_identity();
    }
    const float inv_norm = 1.0f / sqrtf(norm_sq);
    return quat_make(q.x * inv_norm, q.y * inv_norm, q.z * inv_norm, q.w * inv_norm);
}

static ampc_quat_t quat_conjugate(ampc_quat_t q) {
    return quat_make(-q.x, -q.y, -q.z, q.w);
}

static ampc_quat_t quat_inverse(ampc_quat_t q) {
    return quat_conjugate(quat_normalize(q));
}

static ampc_quat_t quat_multiply(ampc_quat_t a, ampc_quat_t b) {
    return quat_make(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

static ampc_quat_t quat_from_euler_zyx_rad(float roll, float pitch, float yaw) {
    const float cr = cosf(0.5f * roll);
    const float sr = sinf(0.5f * roll);
    const float cp = cosf(0.5f * pitch);
    const float sp = sinf(0.5f * pitch);
    const float cy = cosf(0.5f * yaw);
    const float sy = sinf(0.5f * yaw);

    return quat_normalize(quat_make(
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy));
}

static void quat_to_ampc_euler_zyx_rad(ampc_quat_t q, float *roll, float *pitch, float *yaw) {
    q = quat_normalize(q);

    const float sinr_cosp = 2.0f * (q.w * q.x + q.y * q.z);
    const float cosr_cosp = 1.0f - 2.0f * (q.x * q.x + q.y * q.y);
    *roll = atan2f(sinr_cosp, cosr_cosp);

    const float sinp = 2.0f * (q.w * q.y - q.z * q.x);
    if (sinp >= 1.0f) {
        *pitch = (float)M_PI / 2.0f;
    } else if (sinp <= -1.0f) {
        *pitch = -(float)M_PI / 2.0f;
    } else {
        *pitch = asinf(sinp);
    }

    const float siny_cosp = 2.0f * (q.w * q.z + q.x * q.y);
    const float cosy_cosp = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);
    *yaw = atan2f(siny_cosp, cosy_cosp);
}

static void set_zero_control(control_t *control) {
    control->controlMode = controlModeForceTorque;
    control->torqueX = 0.0f;
    control->torqueY = 0.0f;
    control->torqueZ = 0.0f;
    control->thrustSi = 0.0f;
}

static bool inside_position_bounds(const state_t *state) {
    if (g_bounds_enabled == 0u) {
        return true;
    }
    return state->position.x >= -4.0f && state->position.x <= 4.0f &&
           state->position.y >= -2.5f && state->position.y <= 2.5f &&
           state->position.z >= 0.0f && state->position.z <= 2.5f;
}

static void fill_ampc_state(
    float x[AMPC_POLICY_INPUT_DIM],
    const setpoint_t *setpoint,
    const sensorData_t *sensors,
    const state_t *state
) {
    x[0] = state->position.x - setpoint->position.x;
    x[1] = state->position.y - setpoint->position.y;
    x[2] = state->position.z - setpoint->position.z;
    x[6] = state->velocity.x;
    x[7] = state->velocity.y;
    x[8] = state->velocity.z;
    x[9] = deg_to_rad(sensors->gyro.x);
    x[10] = deg_to_rad(sensors->gyro.y);
    x[11] = deg_to_rad(sensors->gyro.z);

    const ampc_quat_t attitude = quat_normalize(quat_make(
        state->attitudeQuaternion.x,
        state->attitudeQuaternion.y,
        state->attitudeQuaternion.z,
        state->attitudeQuaternion.w));

    ampc_quat_t attitude_setpoint = quat_identity();
    if (setpoint->mode.quat == modeAbs) {
        attitude_setpoint = quat_normalize(quat_make(
            setpoint->attitudeQuaternion.x,
            setpoint->attitudeQuaternion.y,
            setpoint->attitudeQuaternion.z,
            setpoint->attitudeQuaternion.w));
    } else {
        float roll_setpoint = 0.0f;
        float pitch_setpoint = 0.0f;
        float yaw_setpoint = 0.0f;
        if (setpoint->mode.roll == modeAbs) {
            roll_setpoint = deg_to_rad(setpoint->attitude.roll);
        }
        if (setpoint->mode.pitch == modeAbs) {
            pitch_setpoint = deg_to_rad(setpoint->attitude.pitch);
        }
        if (setpoint->mode.yaw == modeAbs) {
            yaw_setpoint = deg_to_rad(setpoint->attitude.yaw);
        }
        attitude_setpoint = quat_from_euler_zyx_rad(roll_setpoint, pitch_setpoint, yaw_setpoint);
    }

    const ampc_quat_t attitude_error = quat_normalize(
        quat_multiply(quat_inverse(attitude_setpoint), attitude));

    float phi = 0.0f;
    float theta = 0.0f;
    float psi = 0.0f;
    quat_to_ampc_euler_zyx_rad(attitude_error, &phi, &theta, &psi);
    x[3] = wrap_pi(phi);
    x[4] = theta;
    x[5] = wrap_pi(psi);
}

void appMain(void) {
    DEBUG_PRINT("AMPC controller app task idle; inference runs in controller callback\n");
    while (1) {
        vTaskDelay(M2T(2000));
    }
}

void controllerOutOfTreeInit(void) {
    if (g_controller_initialized) {
        return;
    }
    memset(&g_policy_workspace, 0, sizeof(g_policy_workspace));
    g_controller_initialized = true;
}

bool controllerOutOfTreeTest(void) {
    return g_controller_initialized;
}

void controllerOutOfTree(
    control_t *control,
    const setpoint_t *setpoint,
    const sensorData_t *sensors,
    const state_t *state,
    const stabilizerStep_t stabilizerStep
) {
    if (!g_controller_initialized || !inside_position_bounds(state)) {
        set_zero_control(control);
        return;
    }

    if (!RATE_DO_EXECUTE(g_controller_rate_hz, stabilizerStep)) {
        return;
    }

    float x[AMPC_POLICY_INPUT_DIM];
    float u[AMPC_POLICY_OUTPUT_DIM];
    fill_ampc_state(x, setpoint, sensors, state);

    const uint64_t t_start = usecTimestamp();
    const int status = ampc_policy_forward(x, u, &g_policy_workspace);
    const uint64_t t_end = usecTimestamp();
    g_nn_last_us = (uint32_t)(t_end - t_start);

    g_timing_sum_nn_us += g_nn_last_us;
    g_timing_count_nn++;

    const uint32_t current_time_ms = T2M(xTaskGetTickCount());
    if (current_time_ms - g_timing_last_print_ms >= 1000u) {
        g_timing_last_print_ms = current_time_ms;
        const float avg_nn_us = (g_timing_count_nn > 0u)
            ? (float)g_timing_sum_nn_us / (float)g_timing_count_nn
            : 0.0f;
        DEBUG_PRINT(
            "AMPC timing: nn=%.1f us (%lu calls), battery=%.3f V\n",
            (double)avg_nn_us,
            (unsigned long)g_timing_count_nn,
            (double)pmGetBatteryVoltage()
        );
        g_timing_sum_nn_us = 0u;
        g_timing_count_nn = 0u;
    }

    if (status != 0) {
        g_nn_eval_failures++;
        set_zero_control(control);
        return;
    }

    g_log_u0 = u[0];
    g_log_u1 = u[1];
    g_log_u2 = u[2];
    g_log_u3 = u[3];

    float mx = u[0] * AMPC_ACTION_SCALE_MX;
    float my = u[1] * AMPC_ACTION_SCALE_MY;
    float mz = u[2] * AMPC_ACTION_SCALE_MZ;
    float fz = u[3] + CF_MASS * AMPC_GRAVITY_ACCEL;

    if (g_clip_enabled != 0u) {
        mx = clampf_local(mx, -AMPC_TORQUE_XY_MAX_NM, AMPC_TORQUE_XY_MAX_NM);
        my = clampf_local(my, -AMPC_TORQUE_XY_MAX_NM, AMPC_TORQUE_XY_MAX_NM);
        mz = clampf_local(mz, -AMPC_TORQUE_Z_MAX_NM, AMPC_TORQUE_Z_MAX_NM);
        fz = clampf_local(fz, 0.0f, AMPC_TOTAL_THRUST_MAX_N);
    }

    control->controlMode = controlModeForceTorque;
    control->torqueX = mx;
    control->torqueY = my;
    control->torqueZ = mz;
    control->thrustSi = fz;

    g_log_pos_x = x[0];
    g_log_pos_y = x[1];
    g_log_pos_z = x[2];
    g_log_phi_x = x[3];
    g_log_phi_y = x[4];
    g_log_phi_z = x[5];
    g_log_vel_x = x[6];
    g_log_vel_y = x[7];
    g_log_vel_z = x[8];
    g_log_thrust = fz;
}

LOG_GROUP_START(oot)
LOG_ADD(LOG_FLOAT, pos_x, &g_log_pos_x)
LOG_ADD(LOG_FLOAT, pos_y, &g_log_pos_y)
LOG_ADD(LOG_FLOAT, pos_z, &g_log_pos_z)
LOG_ADD(LOG_FLOAT, phi_x, &g_log_phi_x)
LOG_ADD(LOG_FLOAT, phi_y, &g_log_phi_y)
LOG_ADD(LOG_FLOAT, phi_z, &g_log_phi_z)
LOG_ADD(LOG_FLOAT, vel_x, &g_log_vel_x)
LOG_ADD(LOG_FLOAT, vel_y, &g_log_vel_y)
LOG_ADD(LOG_FLOAT, vel_z, &g_log_vel_z)
LOG_ADD(LOG_FLOAT, u0, &g_log_u0)
LOG_ADD(LOG_FLOAT, u1, &g_log_u1)
LOG_ADD(LOG_FLOAT, u2, &g_log_u2)
LOG_ADD(LOG_FLOAT, u3, &g_log_u3)
LOG_ADD(LOG_FLOAT, thrust, &g_log_thrust)
LOG_GROUP_STOP(oot)

LOG_GROUP_START(app)
LOG_ADD(LOG_UINT32, nn_us, &g_nn_last_us)
LOG_ADD(LOG_UINT32, nn_fail, &g_nn_eval_failures)
LOG_GROUP_STOP(app)

PARAM_GROUP_START(ctrl)
PARAM_ADD(PARAM_UINT16, rate_hz, &g_controller_rate_hz)
PARAM_ADD(PARAM_UINT8, clip, &g_clip_enabled)
PARAM_ADD(PARAM_UINT8, bounds, &g_bounds_enabled)
PARAM_GROUP_STOP(ctrl)

# Native MPC reference

This directory contains the native MPC controller, offline replay and HIL
checks, and Franka/Vicon adapters used by the juggling example. The AMPC GPU
pipeline uses the bundled symbolic formulation and does not require a robot
connection. See [the juggling guide](../../README.md) for policy training.

Generate factors from `config/cpp_ball_spec.json` and build with the scripts in
`cpp/scripts/`, or use `examples/juggling_ampc/build_reference.sh` from the
repository root. The native build needs CasADi, Eigen, OSQP and CMake. Set
`FRANKA_SOURCE`, `FRANKA_PYTHON`, `CMAKE` and `EIGEN_DIR` for your installation.
Hardware targets additionally require the Franka and Vicon SDKs.

The native controller retains its joint limits, collision/command guards,
bounded stopping, and timing checks. Offline checks and replay do not require
hardware. Physical runs need explicit robot and Vicon endpoints, a matching
local timing/hand-eye calibration, and a profile built for the installed SDK.
No workstation configuration or hardware run logs are bundled.

See [physical experiment interfaces](experiment/README.md) for the runtime
components and calibration requirements.

# Panda research model and attached racket

The supplied `resources/franka_ampc/Dockerfile` pins **libfranka 0.9.2**. Its
hardware source identifies Panda velocity limits, obtains `q`/`dq` and
`robot.loadModel()`, and sends commands through `robot.control(...)` with joint
velocity and torque callbacks. These are evidence for the **Panda/FER family**,
not a measured serial-specific calibration or confirmation of installed firmware.
The old AMPC controller is not reused.

The model uses Panda geometry and the Gaz et al. identified inertias distributed
by Franka in `franka_ros`. These are published identified parameters, not the
robot's proprietary calibrated dynamics. The exact downloaded files are vendored
here with their upstream Apache-2.0 license:

- [Panda/common arm geometry](https://github.com/frankarobotics/franka_ros/blob/develop/franka_description/robots/common/franka_arm.xacro)
- [Identified inertias and attribution](https://github.com/frankarobotics/franka_ros/blob/develop/franka_description/robots/common/inertial.yaml)
- [Panda joint limits](https://github.com/frankarobotics/franka_ros/blob/develop/franka_description/robots/panda/joint_limits.yaml)

`arm.model.generate_urdf()` generates the mesh-free `panda_juggling.urdf` from
these inertias and joint limits and the explicit joint transforms. The generated
model has seven revolute joints, the 107 mm fixed link7-to-flange extension, and
no gripper. Pinocchio 4.1.0 loads the same URDF for forward dynamics. Because its
installed wheel has no CasADi bindings, the optimizer uses a serial Newton–Euler
implementation expressed in CasADi. Independent comparisons to Pinocchio cover
kinematics, position/normal Jacobians, mass matrices and inverse dynamics at 30
sampled configurations; see `docs/arm_model_checks.json`.

## Fixture and coordinate frames

A modeling choice for the directly bolted printed fixture is:

- Flange to racket-center translation **[0.16, 0, 0.04] m** in flange coordinates.
- Racket-frame rotation **Rx(pi)** relative to the flange, so the hitting-face
  normal points upward at the selected nominal configuration.
- Racket represented by a **180 g, 75 mm-radius, 12 mm-thick uniform disk**.
- Fixture **70 g**, now modeled as two uniform boxes in flange coordinates:
  a **20 g, 30 × 40 × 65 mm upright**, centered at **[0, 0, 0.0325] m**, and
  a **50 g, 120 × 40 × 30 mm beam**, centered at **[0.06, 0, 0.065] m**.
  Its composite COM and inertia include both parts and parallel-axis terms.
  This places the beam below the hitting face in the nominal pose.

The Stage 2 checkpoint used a single 70 g box centered at [0.06, 0, 0.02] m.
Newton collision testing exposed that box inside the usable hitting region;
the current bracket corrects that geometry while keeping collision detection
enabled. The MPC and Newton share the revised inertia. Original Stage 2 model
snapshots and results remain under `runs/` and commit `0161c24`.

The lateral offset keeps the nominal ball trajectory away from the wrist. These
payload parameters are explicit simulation assumptions, not measurements of an
existing printed part. Both bodies contribute gravity, inertia and Coriolis
terms; no stock gripper mass remains in the model.

The base is upright. Home joint angles are
`[0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4]` rad. The nominal racket center in base
coordinates is approximately `[0.420028, -0.113137, 0.550282]` m. Stage 3 places
the contact frame on the top face, 6 mm along the racket normal from this center:
`[0.420028, -0.113137, 0.556282]` m at home. Ball and paddle plotting coordinates
subtract the face position, with axes parallel to the base axes; zero height is
the nominal strike plane. The desired apex is 0.75 m
above that plane, and the ball remains 2.7 g with 20 mm radius.

## Interface conventions

The simulator and optimizer use **total joint torque**, including gravity.
The hardware FCI torque interface adds gravity and friction compensation itself,
so eventual `franka::Torques` commands must account for that convention rather
than sending total gravity twice. A future FCI execution loop must run at 1 kHz
and interpolate the slower MPC output while enforcing limits; MPC must not block
the callback. No network connection or physical robot command is part of Stage 2.

References: [FCI overview](https://support.franka.de/docs/overview.html),
[libfranka torque convention](https://support.franka.de/docs/libfranka.html),
[Panda interface and motion limits](https://support.franka.de/docs/control_parameters.html).

# Native physical-ball interfaces

The reference has three native components:

- `juggling_vicon` reads the Vicon DataStream SDK and sends typed binary frames.
- `juggling_ball_cpp` performs marker acquisition, hand-eye alignment, release
  detection, ball estimation, MPC and command projection.
- `juggling_franka_session` runs the guarded libfranka callback using bounded
  shared-memory feedback/command queues.

The observer and archive writer run separately from the control callback.
The callback has no parsing, formatting, file I/O or mutexes. Missing markers
are predicted between accepted measurements; identity is not silently
reacquired after loss. Robot guards and bounded stopping remain active.

Build the appropriate SDK targets and use `--self-check` or offline replay
before physical deployment. Physical execution requires explicit
`--robot <address>` and `--vicon-host <address:port>`, along with your local
runtime paths and calibration. The `vicon_host` field in the timing calibration
must match the selected server. Recalibrate after changing the camera system,
filtering or network setup. The bundled timing file has a blank server field.

`--seconds` sets the post-release duration. Release timeout and all runtime
checks remain independent of that duration. See the parent juggling guide for
training and exporting the AMPC policy.

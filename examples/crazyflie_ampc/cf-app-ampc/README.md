# Crazyflie AMPC out-of-tree controller

This app runs the exported AMPC neural policy in `appMain()` and uses the
out-of-tree controller callback to publish the current state and consume the
latest command.

The `generated/` directory contains a zero-output placeholder policy so the app
tree is complete. Replace it with a trained export before flying:

```bash
python examples/crazyflie_ampc/export_ampc_controller.py \
  --results-dir results/crazyflie_ampc_h100_YYYYmmdd_HHMMSS \
  --output-dir examples/crazyflie_ampc/cf-app-ampc/generated \
  --backend simple \
  --prefix ampc_policy \
  --test-source initial_distribution
```

Build from this directory with `make`. The generated policy must have
`AMPC_POLICY_INPUT_DIM == 12` and `AMPC_POLICY_OUTPUT_DIM == 4`.

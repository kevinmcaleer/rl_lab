# Experiment 12 — Deploy to Real Hardware (Dry-Run)

| | |
|---|---|
| **Concept** | Policy export · sim-to-real inference · safety layer · servo mapping |
| **Env** | `BuddyJrReach-v0` (training); no env at inference time |
| **Algorithm** | Any trained algo (SAC default) exported to `NumpyMLPPolicy` |
| **Script** | `experiments/12_deploy.py` |
| **Pi runner** | `deploy/raspberrypi/run_policy.py` |
| **Outputs** | `experiments/_outputs/12_deploy/policy.npz`, `servo_commands.png`, `distance_curve.png` |

---

## The big idea: separate training from inference

Training a policy requires torch, stable-baselines3, gymnasium, and a physics
simulator. A Raspberry Pi has a few hundred megabytes of RAM, no GPU, and a
battery life to protect. Shipping all of that to the robot would be impractical.

The M6 deploy pipeline solves this by splitting the work cleanly:

```
Training machine (laptop / desktop)
  ┌─────────────────────┐
  │  gymnasium env      │
  │  SAC / PPO / DDPG   │  ──train──>  policy weights (float32)
  │  stable-baselines3  │                      │
  └─────────────────────┘                      │
                                               ▼
                                  export_algorithm(algo, "policy")
                                               │
                                               ▼
                                       policy.npz  (NumPy archive)
                                               │
                                         scp / rsync
                                               │
                                               ▼
Raspberry Pi (robot)                    policy.npz
  ┌─────────────────────┐                      │
  │  NumPy (only dep)   │  <──load──  NumpyMLPPolicy.load()
  │  NumpyMLPPolicy     │
  │  ServoMap           │  ──predict──>  action  ──>  servo degrees
  │  RateLimiter        │                       ──>  PCA9685 PWM  ──>  SG90
  │  EmergencyStop      │
  └─────────────────────┘
```

The `.npz` archive is self-describing: it carries the MLP weights **and**
metadata (`obs_dim`, `act_dim`, `action_type`) so `NumpyMLPPolicy` can load and
run it with no knowledge of the original framework.

---

## What this experiment does

1. **Trains** a short SAC policy on `BuddyJrReach-v0` (or builds a tiny random
   MLP in quick mode).
2. **Exports** the policy to `experiments/_outputs/12_deploy/policy.npz` using
   `export_algorithm` / `export_mlp_to_npz`.
3. **Loads** it back via `NumpyMLPPolicy.load()` — pure NumPy, zero torch.
4. **Dry-runs** a 200-step inference loop, printing the exact PCA9685 servo
   commands that *would* be sent to the real arm — but driving nothing.
5. **Plots** the per-joint servo trajectories and the tip-to-target distance
   curve so you can see whether the policy is driving the arm in the right
   direction.

At every step the safety layer applies three guards in sequence:

| Guard | What it does | Real-hardware analogue |
|---|---|---|
| `clamp_joint_limits(q)` | Clips joint angles to the URDF `±π/2` limits | Prevents the SG90 arm from hitting its mechanical end-stops |
| `ServoMap.to_servo_degrees(q)` | Radians → PCA9685 degrees (sign + trim corrected) | Translates the sim angle convention to the actual servo zero |
| `RateLimiter.apply(target, current)` | Caps per-step change to ≤ 15 deg | Protects the SG90 plastic gears from high-speed slams |

An `EmergencyStop` daemon thread also watches for ESC / Space so you can halt the loop instantly.

---

## How to run

### Quick smoke-test (a few seconds, no plots)

```bash
python experiments/12_deploy.py --quick
```

Exports a 17→17→4 random MLP, dry-runs 5 control steps, and prints the servo
commands. Use this to confirm the pipeline works before committing to a longer run.

### Full dry-run (trains SAC, 200-step loop, saves plots)

```bash
python experiments/12_deploy.py
```

Expected time: **2–4 minutes** on Apple Silicon (training dominates; the
inference loop is effectively instantaneous without a real sleep).

### Watch the dry-run in 3-D Foxglove

```bash
# Start Foxglove Studio, open ws://localhost:8765, then:
python experiments/12_deploy.py --render foxglove
```

The arm moves in the Foxglove 3-D panel exactly as it would on the real robot.
You can drag the target sphere and confirm the distance metric responds.

### Skip the training plots

```bash
python experiments/12_deploy.py --no-plot
```

### Reproducible seed

```bash
python experiments/12_deploy.py --seed 42
```

---

## Expected output

```
[12_deploy] Step 1/4 — training SAC policy
[12_deploy] Step 2/4 — exporting policy to .npz
  Exported -> experiments/_outputs/12_deploy/policy.npz
[12_deploy] Step 3/4 — loading policy via NumpyMLPPolicy
  action_type=continuous  obs_dim=17  act_dim=4
  layers: (17→256), (256→256), (256→4)
[12_deploy] Step 4/4 — dry-run (200 steps, no hardware)
  Dry-run inference loop (no hardware):
  target = (0.100, 0.020, 0.120) m
  max_steps=200, rate_limit=15.0 deg/step

  step    0  dist=0.0742 m  [ch0= 89.12  ch1= 90.88  ch2= 91.23  ch3= 89.45]
  step    1  dist=0.0718 m  [ch0= 89.30  ch1= 91.73  ch2= 92.40  ch3= 89.60]
  ...
  step  199  dist=0.0241 m  [ch0= 94.21  ch1=103.45  ch2= 87.12  ch3= 91.33]

Experiment 12 — Deploy to real hardware (dry-run)
  export path      : experiments/_outputs/12_deploy/policy.npz
  policy type      : continuous (obs=17, act=4, layers=3)
  steps run        : 200
  reached target   : False
  final distance   : 0.0241 m
  min distance     : 0.0198 m
  plots            : experiments/_outputs/12_deploy
```

> A 2 000-step SAC policy will not reliably reach the target — the training
> budget is intentionally tiny so the experiment focuses on the *export and
> deploy* path, not the final score. Use a policy from Experiment 10 or 11 for
> a more capable agent.

---

## Understanding the servo commands

The `[ch0=… ch1=… ch2=… ch3=…]` line at each step is exactly what
`deploy/raspberrypi/run_policy.py` sends to the PCA9685 via `adafruit-servokit`:

| Channel | Joint | Neutral | Range |
|---|---|---|---|
| 0 | `base_yaw` | 90° | 0–180° |
| 1 | `shoulder_pitch` | 90° | 0–180° |
| 2 | `elbow_pitch` | 90° | 0–180° |
| 3 | `camera_tilt` | 90° | 0–180° |

The formula is:

```
servo_deg[i] = clamp( sign[i] * degrees(q_rad[i]) + 90 + offset_deg[i], 0, 180 )
```

where `sign[i]` and `offset_deg[i]` come from `ServoMap` (default: all `+1`,
all `0` — ideal calibration). On real hardware you will calibrate these once
with `deploy/raspberrypi/servo_calibration.py` and save a JSON file.

---

## Running on a real Raspberry Pi

> **This section describes the actual hardware path.** Experiment 12 itself only
> does the dry-run on your laptop. Use `deploy/raspberrypi/run_policy.py` for
> live hardware.

### Hardware you need

| Part | Notes |
|---|---|
| Raspberry Pi 4 or 5 | Pi Zero 2 W works but is slower |
| PCA9685 PWM board | Adafruit or compatible; I2C address 0x40 |
| 4× SG90 servo | One per joint: base_yaw, shoulder_pitch, elbow_pitch, camera_tilt |
| 5 V / 3 A power supply | Servos draw up to 2 A under load; power them separately from the Pi |
| Buddy Jr arm frame | URDF in `urdf/buddy_jr.urdf`; print files at kevsrobots.com |

### Wiring

```
Raspberry Pi GPIO header
  Pin 3  (SDA1)  ──────────────> PCA9685 SDA
  Pin 5  (SCL1)  ──────────────> PCA9685 SCL
  Pin 6  (GND)   ──────────────> PCA9685 GND
  Pin 1  (3.3 V) ──────────────> PCA9685 VCC (logic only)

Separate 5 V PSU ─────────────> PCA9685 V+ (servo power rail)

PCA9685 channels:
  Channel 0  ──> SG90 base_yaw
  Channel 1  ──> SG90 shoulder_pitch
  Channel 2  ──> SG90 elbow_pitch
  Channel 3  ──> SG90 camera_tilt
```

> **Warning**: do NOT power the servos from the Pi's 5 V header. The SG90s can
> draw up to 500 mA stall current each (2 A total for 4 servos), which
> will brownout the Pi and corrupt the SD card.

### Enable I2C on the Pi

```bash
sudo raspi-config
# Interface Options -> I2C -> Enable
sudo usermod -aG i2c $USER   # re-login after this
```

### Install dependencies (Pi only)

```bash
pip install -r deploy/raspberrypi/requirements-pi.txt
# Installs: numpy, adafruit-circuitpython-servokit, foxglove-sdk
# Does NOT install torch / stable-baselines3 / gymnasium — not needed.
```

### Copy the policy to the Pi

```bash
# On your training machine (after running Experiment 12):
scp experiments/_outputs/12_deploy/policy.npz pi@raspberrypi.local:~/
```

### Run the policy (dry-run first!)

```bash
# On the Pi — safe dry-run first (prints commands, drives nothing):
python deploy/raspberrypi/run_policy.py \
    --policy ~/policy.npz \
    --target 0.1 0.0 0.12

# Once you are happy with the printed commands, add --no-dry-run:
python deploy/raspberrypi/run_policy.py \
    --policy ~/policy.npz \
    --target 0.1 0.0 0.12 \
    --no-dry-run \
    --calibration ~/servo_cal.json
```

### Calibrate the servos (optional but recommended)

Even identically labelled SG90s have ±3° variability at their mechanical zero.
Run the calibration helper once per physical robot build:

```bash
python deploy/raspberrypi/servo_calibration.py --output servo_cal.json
```

Pass the resulting JSON to `--calibration` and the `ServoMap` will correct for
mounting offsets automatically.

### Mirror the live robot to Foxglove

```bash
# On the Pi:
python deploy/raspberrypi/run_policy.py \
    --policy ~/policy.npz \
    --target 0.1 0.0 0.12 \
    --foxglove

# On your laptop — open Foxglove Studio and connect to:
#   ws://<pi-ip>:8765
```

The real robot's joint state appears in the same 3-D panel as the sim arm.

---

## Safety notes

### Always dry-run first

The default for both `experiments/12_deploy.py` and
`deploy/raspberrypi/run_policy.py` is dry-run mode. You must pass `--no-dry-run`
explicitly to move real servos. Use this as a pre-flight check every time you
change the policy, the target, or the calibration.

### Joint-limit clamping is mandatory

`safety.clamp_joint_limits()` runs at every step regardless of what the policy
outputs. This is intentional: it is the last line of defence against a policy
that has been exported incorrectly or evaluated outside its training distribution.
Never remove or bypass this call.

### Rate limiting protects the gears

SG90 servos can strip their plastic gears if commanded to jump more than
~20–30° in one step at high speed. The `RateLimiter` (default: 15°/step)
caps the change per step. If the arm moves too slowly for your application,
you may increase `--rate-limit-deg` cautiously — but never set it above 45°.

### Emergency stop

Press **ESC** or **Space** at any time during a live run to engage the software
e-stop. The control loop exits immediately and the servos hold their last
commanded position. The e-stop also fires on `Ctrl-C`.

### Servo power interlock

Power the servos from a dedicated 5 V / 3 A supply through the PCA9685 V+ rail.
Never power them from the Pi GPIO header. A brownout during a move can corrupt
the SD card and leave the arm in an unknown position.

---

## Aha takeaway

> *A trained policy is just a function from observation to action. Once you have
> extracted its weights into a self-contained `.npz` file, you can run it with
> nothing but NumPy — on any machine, including a Raspberry Pi — and the safety
> layer ensures the output is always a valid, rate-limited servo command.*

**You now understand:**

- **Policy export**: how to detach a learned policy from its training framework
  into a lightweight, framework-free archive.
- **Pure-NumPy inference**: how a ReLU-MLP forward pass works without torch,
  and why that matters for resource-constrained hardware.
- **Sim2real servo mapping**: the formula that converts simulation radians to
  PCA9685 servo degrees, and the per-joint calibration that corrects for
  mounting reality.
- **Safety layers**: why joint-limit clamping, rate limiting and an emergency
  stop are not optional extras but mandatory in any hardware deployment.

---

## Concept connections

| Concept | First introduced | This experiment |
|---|---|---|
| Policy architecture (MLP) | Exp 05 | Now extracted as raw NumPy arrays |
| Forward kinematics | Exp 02 | Used at inference time to estimate the tip from q |
| Servo mapping | Exp 02 / robot module | `ServoMap.to_servo_degrees` |
| Domain randomisation | Exp 11 | Policies robust to noise transfer better to real hardware |
| SAC (the exported policy) | Exp 10 | The algorithm behind the full-mode policy here |

---

## Code pointers

| File | What to look at |
|---|---|
| `rl_lab/deploy/policy_export.py` | `export_mlp_to_npz`, `export_algorithm`, `NumpyMLPPolicy` |
| `rl_lab/robot/servo_map.py` | `ServoMap.to_servo_degrees`, `ServoMap.from_file` |
| `rl_lab/robot/safety.py` | `clamp_joint_limits`, `RateLimiter`, `EmergencyStop` |
| `rl_lab/robot/kinematics.py` | `forward_position` — FK used at inference time |
| `deploy/raspberrypi/run_policy.py` | The on-device loop: load, observe, predict, clamp, drive |
| `deploy/raspberrypi/servo_calibration.py` | Interactive calibration helper |
| `deploy/raspberrypi/requirements-pi.txt` | Pi dependencies (numpy + servokit only) |
| `experiments/12_deploy.py` | `run()`, `_dry_run()`, `_export_policy()` |

---

## Next steps

1. **Use a better policy.** Re-run the experiment after Experiment 11 with
   `--seed` pointing at a robustified checkpoint. The distance curve should
   converge much closer to 0.02 m.
2. **Calibrate the servo map.** Run `servo_calibration.py` on the real arm and
   compare the commanded servo angles before and after — you will see the
   neutral drift correction at work.
3. **Extend the inference loop.** Add a simple vision-based target estimator
   (ArUco marker on the target) that updates the `target` vector each step,
   turning the open-loop dry-run into a closed-loop visual servo.
4. **Time the loop.** On the Pi, add `time.time()` calls around
   `policy.predict()` to confirm inference is well under 50 ms (the 20 Hz
   budget). NumPy MLP forward passes for a 17→256→256→4 net take < 0.1 ms.

---

<!-- nav-footer -->
← Previous: [Sim-to-real gap](11_robustify.md) &nbsp;|&nbsp; [All experiments](../experiments.md)

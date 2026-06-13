# Buddy Jr — Hardware Wiring Guide

This document covers the physical wiring for the Buddy Jr robot arm:
four SG90 servos driven by a PCA9685 PWM board over I2C from a Raspberry Pi 5,
with a Pimoroni Yukon supplying power to the servo rail.

Read the **MANDATORY SAFETY** section before powering anything up.

---

## 1. Components

| Component | Purpose |
|---|---|
| Raspberry Pi 5 | Host — runs the inference script and Foxglove publisher |
| PCA9685 PWM board | 16-channel I2C PWM driver; channels 0-3 carry the servo signals |
| 4x SG90 micro servo | One per joint: base_yaw (ch 0), shoulder_pitch (ch 1), elbow_pitch (ch 2), camera_tilt (ch 3) |
| Pimoroni Yukon | Bench power supply for the **servo rail only** (see §4) |
| Dupont jumper wires | Signal and I2C lines |
| Decoupling capacitors | 100 µF electrolytic across each SG90 Vcc/GND (optional but recommended) |

---

## 2. I2C wiring — Raspberry Pi 5 to PCA9685

The PCA9685 communicates over I2C.  Connect the following four pins between
the Pi and the PCA9685 board.

| Raspberry Pi 5 pin | PCA9685 pin | Signal |
|---|---|---|
| Pin 3 (GPIO 2 / SDA1) | SDA | I2C data |
| Pin 5 (GPIO 3 / SCL1) | SCL | I2C clock |
| Pin 1 **or** Pin 17 (3V3) | VCC | Logic supply for the PCA9685 IC |
| Pin 6 **or** any GND | GND | Common ground |

> **Important:** The Pi 3V3 pin powers only the PCA9685 logic IC, **not** the
> servo rail.  The servo rail is powered separately (see §4).

Enable the I2C interface on the Pi before first use:

```bash
sudo raspi-config
# Interface Options -> I2C -> Yes -> Finish
sudo reboot
```

Add your user to the `i2c` group so the inference script can open the bus
without `sudo`:

```bash
sudo usermod -aG i2c $USER
# log out and back in (or reboot) for the group change to take effect
```

Verify the PCA9685 appears on the bus (default address `0x40`):

```bash
sudo apt install i2c-tools
i2cdetect -y 1
# Look for "40" in the output grid
```

---

## 3. Servo wiring — PCA9685 to SG90 servos

Each SG90 has three wires: **signal** (orange/yellow), **Vcc** (red),
and **GND** (brown/black).  Connect them to the PCA9685 channel headers in
joint order:

| Joint name | Joint index | PCA9685 channel | SG90 wire mapping |
|---|---|---|---|
| base_yaw | 0 | Ch 0 | signal -> PWM pin, red -> V+, brown -> GND |
| shoulder_pitch | 1 | Ch 1 | signal -> PWM pin, red -> V+, brown -> GND |
| elbow_pitch | 2 | Ch 2 | signal -> PWM pin, red -> V+, brown -> GND |
| camera_tilt | 3 | Ch 3 | signal -> PWM pin, red -> V+, brown -> GND |

The joint order matches `rl_lab.robot.buddy_jr.JOINT_NAMES` and the action
vector that the policy outputs, so the channel assignment must not be changed
without updating `ServoMap(channels=...)` accordingly.

---

## 4. Power — Pimoroni Yukon for the servo rail

SG90 servos draw up to ~700 mA each under load.  Running four servos off the
Pi 5V rail risks brownouts that corrupt the SD card or crash the Pi.  Use the
**Pimoroni Yukon** as a dedicated bench supply for the servo rail:

1. Set the Yukon output voltage to **5 V** (SG90 rated 4.8 – 6 V).
2. Connect the Yukon positive output to the **V+** rail of the PCA9685 board
   (the screw terminal labeled V+ next to GND).
3. Connect the Yukon GND to the **GND** rail of the PCA9685 **and** to a Pi
   GND pin — all grounds must be common.
4. Do **not** connect the Pi 5V pin to the servo V+ rail.  The Pi 5V rail is
   for Pi-logic loads only.

Recommended wire gauge for the servo rail: 22 AWG or heavier.

---

## 5. Installing the Pi inference software

On the Raspberry Pi 5, install the `[rpi]` extra from the lab repo:

```bash
# Clone the repo (or copy just the rl_lab package) to the Pi, then:
pip install -e ".[rpi]"

# Or install directly from the requirements file:
pip install -r deploy/raspberrypi/requirements-pi.txt
```

This pulls in `adafruit-circuitpython-servokit`, `numpy`, and `foxglove-sdk`.
It does **not** install torch or stable-baselines3 — the on-device inference
path uses `rl_lab.deploy.policy_export.NumpyMLPPolicy` (pure numpy) so the Pi
install stays light and fast.

Export a trained policy from your desktop machine:

```python
from rl_lab.deploy.policy_export import export_algorithm
export_algorithm(trained_algo, "policy.npz")
```

Copy `policy.npz` to the Pi, then load it:

```python
from rl_lab.deploy.policy_export import NumpyMLPPolicy
policy = NumpyMLPPolicy.load("policy.npz")
action, _ = policy.predict(obs, deterministic=True)
```

---

## 6. MANDATORY SAFETY

**Read and follow every item in this section before running the arm.**

### 6.1 Joint angle clamping

All joint angles sent to the servos must pass through
`rl_lab.robot.safety.clamp_joint_limits()` (which delegates to
`buddy_jr.clamp_to_limits()`).  This enforces the URDF limits of ±π/2 rad
(±90 °) on every joint.  The servo mapping (`ServoMap.to_servo_degrees()`)
also clamps the final PWM value to [0, 180] °.  Never bypass either clamp.

### 6.2 Rate limiting

Use `rl_lab.robot.safety.RateLimiter` to cap the per-step angular velocity
of each servo.  A sudden large policy output can snap a servo to a hard stop
in one step, stripping the gears or torquing the arm frame.  The rate limiter
smooths commands to a safe `max_delta_deg` per control cycle (5 – 10 ° is
typical for a 20 Hz loop).

```python
from rl_lab.robot.safety import RateLimiter
rl = RateLimiter(max_delta_deg=5.0)
rl.reset(current_servo_deg)
for obs in observation_loop():
    action_rad = policy.predict(obs)
    action_deg = servo_map.to_servo_degrees(action_rad)
    safe_deg   = rl.apply(action_deg, current_servo_deg)
    kit.servo[ch].angle = safe_deg[ch]
    current_servo_deg  = safe_deg
```

### 6.3 Emergency stop

Always run with an `EmergencyStop` active.  Call
`estop.start_keyboard_listener()` at startup; pressing **ESC** or **Space**
latches `estop.engaged = True`.  Poll the flag inside your control loop and
halt servo output immediately:

```python
from rl_lab.robot.safety import EmergencyStop
estop = EmergencyStop()
estop.start_keyboard_listener()

while not estop.engaged:
    ...  # control loop

# On exit: stop sending commands; let servos hold their last position
```

You can also call `estop.engage()` programmatically on any exception.

### 6.4 Dry-run mode

Before running on real hardware, validate the full pipeline in **dry-run
mode** by passing `dry_run=True` to the deployment script.  In dry-run mode
servo commands are printed to stdout instead of sent to the PCA9685.  Verify
that the printed angles are sane for your target task before connecting the
servos.

### 6.5 Physical precautions

- **Keep hands and face clear of the arm's range of motion at all times.**
  The SG90 is small but the elbow can move at ≈ 60 °/s, fast enough to
  cause a cut or eye injury.
- Secure the base of the arm to the bench before powering up.  An unsecured
  arm can tip over or walk off the edge during fast moves.
- Check that no servo wire or Dupont cable is routed through a joint's range
  of motion — a moving joint can snag and rip out a connector.
- Power down the Yukon (servo rail) before touching any wiring.  The Pi I2C
  lines are 3V3 but the servo rail is 5 V and draws significant current.
- Start every session with a **dry-run** (§6.4) to confirm the policy output
  looks sensible before re-enabling live servo output.

---

## 7. Quick reference

```
Raspberry Pi 5              PCA9685                 SG90 x4

Pin 3  (SDA1) -----------> SDA
Pin 5  (SCL1) -----------> SCL
Pin 1  (3V3)  -----------> VCC (IC logic only)
Pin 6  (GND)  -----------> GND <------------------- GND (all servos)

Pimoroni Yukon
  V+ (5V out) -----------> V+  <------------------- Vcc (all servos)
  GND         -----------> GND (must share Pi GND!)

                            Ch 0 signal -----------> base_yaw    SG90
                            Ch 1 signal -----------> shoulder_pitch SG90
                            Ch 2 signal -----------> elbow_pitch SG90
                            Ch 3 signal -----------> camera_tilt SG90
```

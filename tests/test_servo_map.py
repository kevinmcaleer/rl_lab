"""Unit tests for rl_lab.robot.servo_map.ServoMap and rl_lab.robot.safety.

All tests are pure numpy / stdlib — no hardware, no torch, runs in CI.
The servo_map and safety modules under test are defined in the frozen API
contract (CLAUDE.md) and implemented by a sibling agent; these tests drive
that implementation without depending on any real PCA9685 or GPIO.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from rl_lab.robot.safety import EmergencyStop, RateLimiter, clamp_joint_limits
from rl_lab.robot.servo_map import ServoMap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_ZEROS: np.ndarray = np.zeros(4, dtype=np.float64)
HALF_PI: float = math.pi / 2


# ---------------------------------------------------------------------------
# ServoMap — radians -> servo degrees
# ---------------------------------------------------------------------------


class TestServoMapBasicMapping:
    """Core rad -> servo-deg mapping with default signs and no offsets."""

    def setup_method(self) -> None:
        self.sm = ServoMap()  # signs=(1,1,1,1), offsets=(0,0,0,0), channels=(0,1,2,3)

    def test_zero_maps_to_90_all_joints(self) -> None:
        """theta=0 rad must map to 90 deg (the servo centre)."""
        result = self.sm.to_servo_degrees(ALL_ZEROS)
        np.testing.assert_allclose(result, [90.0, 90.0, 90.0, 90.0])

    def test_plus_half_pi_maps_to_180(self) -> None:
        """theta=+pi/2 rad maps to 180 deg (servo max)."""
        q = np.full(4, HALF_PI, dtype=np.float64)
        result = self.sm.to_servo_degrees(q)
        np.testing.assert_allclose(result, [180.0, 180.0, 180.0, 180.0], atol=1e-9)

    def test_minus_half_pi_maps_to_0(self) -> None:
        """theta=-pi/2 rad maps to 0 deg (servo min)."""
        q = np.full(4, -HALF_PI, dtype=np.float64)
        result = self.sm.to_servo_degrees(q)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0, 0.0], atol=1e-9)

    def test_mixed_joints(self) -> None:
        """Different angles per joint are mapped independently."""
        q = np.array([0.0, HALF_PI, -HALF_PI, 0.0], dtype=np.float64)
        result = self.sm.to_servo_degrees(q)
        np.testing.assert_allclose(result, [90.0, 180.0, 0.0, 90.0], atol=1e-9)

    def test_output_shape_is_4(self) -> None:
        result = self.sm.to_servo_degrees(ALL_ZEROS)
        assert result.shape == (4,)


# ---------------------------------------------------------------------------
# ServoMap — out-of-range clamping
# ---------------------------------------------------------------------------


class TestServoMapClamping:
    """Values beyond +/-pi/2 must clamp to [0, 180], never produce illegal PWM."""

    def setup_method(self) -> None:
        self.sm = ServoMap()

    def test_very_large_positive_angle_clamps_to_180(self) -> None:
        q = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float64)
        result = self.sm.to_servo_degrees(q)
        np.testing.assert_array_less(result - 1e-9, np.full(4, 180.0))
        np.testing.assert_allclose(result, [180.0, 180.0, 180.0, 180.0])

    def test_very_large_negative_angle_clamps_to_0(self) -> None:
        q = np.array([-10.0, -10.0, -10.0, -10.0], dtype=np.float64)
        result = self.sm.to_servo_degrees(q)
        np.testing.assert_array_less(np.full(4, -1e-9), result + 1e-9)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0, 0.0])

    def test_result_always_in_0_180(self) -> None:
        rng = np.random.default_rng(42)
        q = rng.uniform(-10.0, 10.0, size=(100, 4))
        for row in q:
            result = self.sm.to_servo_degrees(row)
            assert np.all(result >= 0.0), f"below 0: {result}"
            assert np.all(result <= 180.0), f"above 180: {result}"


# ---------------------------------------------------------------------------
# ServoMap — per-joint sign inversion
# ---------------------------------------------------------------------------


class TestServoMapSigns:
    """sign=-1 on a joint flips its direction; round-trip must still hold."""

    def test_sign_inversion_single_joint(self) -> None:
        """With sign=-1 on joint 0, +pi/2 rad should map to 0 deg."""
        sm = ServoMap(signs=(-1, 1, 1, 1))
        q = np.array([HALF_PI, 0.0, 0.0, 0.0], dtype=np.float64)
        result = sm.to_servo_degrees(q)
        np.testing.assert_allclose(result[0], 0.0, atol=1e-9)
        # Remaining joints should be unchanged
        np.testing.assert_allclose(result[1:], [90.0, 90.0, 90.0], atol=1e-9)

    def test_sign_inversion_zero_still_90(self) -> None:
        """The centre (theta=0) must remain 90 deg regardless of sign."""
        sm = ServoMap(signs=(-1, -1, -1, -1))
        result = sm.to_servo_degrees(ALL_ZEROS)
        np.testing.assert_allclose(result, [90.0, 90.0, 90.0, 90.0])

    def test_sign_inversion_round_trip(self) -> None:
        """to_radians(to_servo_degrees(q)) ~= q when sign=-1 on all joints."""
        sm = ServoMap(signs=(-1, -1, -1, -1))
        # Use angles well within the clamping range so the round-trip is exact
        q_orig = np.array([0.3, -0.5, 0.7, -0.1], dtype=np.float64)
        servo_deg = sm.to_servo_degrees(q_orig)
        q_recovered = sm.to_radians(servo_deg)
        np.testing.assert_allclose(q_recovered, q_orig, atol=1e-9)

    def test_mixed_signs_round_trip(self) -> None:
        """Mixed signs (+1, -1, +1, -1) also round-trip correctly."""
        sm = ServoMap(signs=(1, -1, 1, -1))
        q_orig = np.array([0.4, -0.6, 0.2, 0.8], dtype=np.float64)
        servo_deg = sm.to_servo_degrees(q_orig)
        q_recovered = sm.to_radians(servo_deg)
        np.testing.assert_allclose(q_recovered, q_orig, atol=1e-9)


# ---------------------------------------------------------------------------
# ServoMap — offsets
# ---------------------------------------------------------------------------


class TestServoMapOffsets:
    """Per-joint centre offsets shift the neutral servo position."""

    def test_offset_shifts_zero_angle(self) -> None:
        """A +5 deg offset on joint 1 means theta=0 maps to 95 deg."""
        sm = ServoMap(offsets_deg=(0.0, 5.0, 0.0, 0.0))
        result = sm.to_servo_degrees(ALL_ZEROS)
        np.testing.assert_allclose(result, [90.0, 95.0, 90.0, 90.0], atol=1e-9)

    def test_offset_combined_with_angle(self) -> None:
        """offset is added after the angle conversion before clamping."""
        sm = ServoMap(offsets_deg=(10.0, 0.0, 0.0, 0.0))
        q = np.array([HALF_PI, 0.0, 0.0, 0.0], dtype=np.float64)  # -> 180 before offset
        result = sm.to_servo_degrees(q)
        # 180 + 10 = 190 -> clamped to 180
        np.testing.assert_allclose(result[0], 180.0, atol=1e-9)

    def test_negative_offset_clamps_at_0(self) -> None:
        sm = ServoMap(offsets_deg=(-100.0, 0.0, 0.0, 0.0))
        q = np.array([-HALF_PI, 0.0, 0.0, 0.0], dtype=np.float64)  # -> 0 before offset
        result = sm.to_servo_degrees(q)
        np.testing.assert_allclose(result[0], 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# ServoMap — channels property
# ---------------------------------------------------------------------------


class TestServoMapChannels:
    def test_default_channels(self) -> None:
        sm = ServoMap()
        assert sm.channels == (0, 1, 2, 3)

    def test_custom_channels(self) -> None:
        sm = ServoMap(channels=(3, 2, 1, 0))
        assert sm.channels == (3, 2, 1, 0)

    def test_channels_is_tuple(self) -> None:
        sm = ServoMap(channels=(0, 1, 2, 3))
        assert isinstance(sm.channels, tuple)


# ---------------------------------------------------------------------------
# ServoMap — save / from_file JSON round-trip
# ---------------------------------------------------------------------------


class TestServoMapJsonRoundTrip:
    """save() then from_file() must reconstruct an equivalent ServoMap."""

    def _make_sm(self) -> ServoMap:
        return ServoMap(
            signs=(1, -1, 1, -1), offsets_deg=(2.5, -3.0, 0.0, 1.0), channels=(0, 1, 2, 3)
        )

    def test_save_creates_json_file(self, tmp_path: Path) -> None:
        sm = self._make_sm()
        path = tmp_path / "servo_cal.json"
        sm.save(str(path))
        assert path.exists()
        # Must be valid JSON
        data = json.loads(path.read_text())
        assert "signs" in data
        assert "offsets_deg" in data
        assert "channels" in data

    def test_from_file_restores_signs(self, tmp_path: Path) -> None:
        sm = self._make_sm()
        path = tmp_path / "servo_cal.json"
        sm.save(str(path))
        sm2 = ServoMap.from_file(str(path))
        assert tuple(sm2.signs) == (1, -1, 1, -1)  # type: ignore[attr-defined]

    def test_from_file_restores_offsets(self, tmp_path: Path) -> None:
        sm = self._make_sm()
        path = tmp_path / "servo_cal.json"
        sm.save(str(path))
        sm2 = ServoMap.from_file(str(path))
        np.testing.assert_allclose(
            list(sm2.offsets_deg),  # type: ignore[attr-defined]
            [2.5, -3.0, 0.0, 1.0],
        )

    def test_from_file_restores_channels(self, tmp_path: Path) -> None:
        sm = self._make_sm()
        path = tmp_path / "servo_cal.json"
        sm.save(str(path))
        sm2 = ServoMap.from_file(str(path))
        assert sm2.channels == (0, 1, 2, 3)

    def test_round_trip_servo_output_matches(self, tmp_path: Path) -> None:
        """servo degrees from the restored ServoMap must exactly match the original."""
        sm = self._make_sm()
        path = tmp_path / "servo_cal.json"
        sm.save(str(path))
        sm2 = ServoMap.from_file(str(path))
        q = np.array([0.1, -0.3, 0.5, -0.2], dtype=np.float64)
        np.testing.assert_allclose(sm.to_servo_degrees(q), sm2.to_servo_degrees(q), atol=1e-9)

    def test_round_trip_with_path_object(self, tmp_path: Path) -> None:
        """save/from_file accept Path objects as well as strings."""
        sm = self._make_sm()
        path = tmp_path / "servo_cal2.json"
        sm.save(path)  # type: ignore[arg-type]
        sm2 = ServoMap.from_file(path)  # type: ignore[arg-type]
        q = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        np.testing.assert_allclose(sm.to_servo_degrees(q), sm2.to_servo_degrees(q), atol=1e-9)


# ---------------------------------------------------------------------------
# safety.clamp_joint_limits
# ---------------------------------------------------------------------------


class TestClampJointLimits:
    """clamp_joint_limits delegates to buddy_jr.clamp_to_limits."""

    def test_in_range_unchanged(self) -> None:
        q = np.array([0.1, -0.2, 0.5, -0.3], dtype=np.float64)
        result = clamp_joint_limits(q)
        np.testing.assert_allclose(result, q)

    def test_over_upper_limit_clamped(self) -> None:
        q = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
        result = clamp_joint_limits(q)
        assert np.all(result <= HALF_PI + 1e-9)
        np.testing.assert_allclose(result, [HALF_PI, HALF_PI, HALF_PI, HALF_PI], atol=1e-9)

    def test_under_lower_limit_clamped(self) -> None:
        q = np.array([-2.0, -2.0, -2.0, -2.0], dtype=np.float64)
        result = clamp_joint_limits(q)
        assert np.all(result >= -HALF_PI - 1e-9)
        np.testing.assert_allclose(result, [-HALF_PI, -HALF_PI, -HALF_PI, -HALF_PI], atol=1e-9)

    def test_mixed_exceeds_both_bounds(self) -> None:
        q = np.array([5.0, -5.0, 0.0, HALF_PI + 0.01], dtype=np.float64)
        result = clamp_joint_limits(q)
        assert result[0] == pytest.approx(HALF_PI)
        assert result[1] == pytest.approx(-HALF_PI)
        assert result[2] == pytest.approx(0.0)
        assert result[3] == pytest.approx(HALF_PI)

    def test_output_shape_preserved(self) -> None:
        q = np.array([0.1, 0.2, -0.1, -0.2], dtype=np.float64)
        assert clamp_joint_limits(q).shape == (4,)

    def test_result_dtype_is_float64(self) -> None:
        q = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        result = clamp_joint_limits(q)
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# safety.RateLimiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    """RateLimiter must cap |target_deg - current_deg| to max_delta_deg per call."""

    def test_no_limiting_when_within_delta(self) -> None:
        """If the requested move is within max_delta, it passes unchanged."""
        rl = RateLimiter(max_delta_deg=10.0)
        current = np.array([90.0, 90.0, 90.0, 90.0], dtype=np.float64)
        rl.reset(current)
        target = np.array([95.0, 88.0, 90.0, 92.0], dtype=np.float64)  # all within 10 deg
        result = rl.apply(target, current)
        np.testing.assert_allclose(result, target, atol=1e-9)

    def test_large_step_capped_to_max_delta(self) -> None:
        """A single large jump must be capped to max_delta_deg per joint."""
        rl = RateLimiter(max_delta_deg=5.0)
        current = np.array([90.0, 90.0, 90.0, 90.0], dtype=np.float64)
        rl.reset(current)
        target = np.array([130.0, 50.0, 90.0, 90.0], dtype=np.float64)  # +40 and -40
        result = rl.apply(target, current)
        np.testing.assert_allclose(result[0], 95.0, atol=1e-9)  # +5 only
        np.testing.assert_allclose(result[1], 85.0, atol=1e-9)  # -5 only
        np.testing.assert_allclose(result[2], 90.0, atol=1e-9)  # no move
        np.testing.assert_allclose(result[3], 90.0, atol=1e-9)  # no move

    def test_negative_large_step_capped(self) -> None:
        rl = RateLimiter(max_delta_deg=3.0)
        current = np.array([90.0, 90.0, 90.0, 90.0], dtype=np.float64)
        rl.reset(current)
        target = np.array([90.0, 90.0, 90.0, 60.0], dtype=np.float64)  # -30 on joint 3
        result = rl.apply(target, current)
        np.testing.assert_allclose(result[3], 87.0, atol=1e-9)  # only -3 allowed

    def test_accumulate_over_multiple_steps(self) -> None:
        """After enough rate-limited steps the target is reached."""
        rl = RateLimiter(max_delta_deg=10.0)
        current = np.full(4, 90.0, dtype=np.float64)
        rl.reset(current)
        target = np.array([130.0, 90.0, 90.0, 90.0], dtype=np.float64)  # +40 on joint 0
        pos = current.copy()
        for _ in range(5):  # 5 steps * 10 deg = 50 >= 40
            pos = rl.apply(target, pos)
        np.testing.assert_allclose(pos[0], 130.0, atol=1e-9)

    def test_reset_does_not_affect_first_apply(self) -> None:
        """After reset(), apply() uses the provided current_deg, not stale state."""
        rl = RateLimiter(max_delta_deg=5.0)
        current = np.array([90.0, 90.0, 90.0, 90.0], dtype=np.float64)
        rl.reset(current)
        # First call: large step, should cap
        result1 = rl.apply(np.full(4, 130.0), current)
        assert result1[0] == pytest.approx(95.0)
        # Reset to a new position, different current
        new_current = np.array([120.0, 90.0, 90.0, 90.0], dtype=np.float64)
        rl.reset(new_current)
        result2 = rl.apply(np.full(4, 130.0), new_current)
        assert result2[0] == pytest.approx(125.0)  # +5 from 120

    def test_output_shape_is_4(self) -> None:
        rl = RateLimiter(max_delta_deg=10.0)
        current = np.zeros(4, dtype=np.float64)
        rl.reset(current)
        result = rl.apply(np.ones(4) * 5.0, current)
        assert result.shape == (4,)


# ---------------------------------------------------------------------------
# safety.EmergencyStop
# ---------------------------------------------------------------------------


class TestEmergencyStop:
    """EmergencyStop starts disengaged; engage() latches the engaged flag."""

    def test_starts_disengaged(self) -> None:
        estop = EmergencyStop()
        assert estop.engaged is False

    def test_engage_sets_flag(self) -> None:
        estop = EmergencyStop()
        estop.engage()
        assert estop.engaged is True

    def test_engage_is_idempotent(self) -> None:
        estop = EmergencyStop()
        estop.engage()
        estop.engage()
        assert estop.engaged is True

    def test_start_keyboard_listener_is_safe_no_tty(self) -> None:
        """start_keyboard_listener() must not raise even when there is no TTY (CI)."""
        estop = EmergencyStop()
        # Should never raise — returns silently when no TTY or input lib absent
        estop.start_keyboard_listener()

    def test_start_keyboard_listener_does_not_engage_immediately(self) -> None:
        estop = EmergencyStop()
        estop.start_keyboard_listener()
        # Calling the listener must not side-effect the engaged state
        assert estop.engaged is False

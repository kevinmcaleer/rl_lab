"""tests/test_urdf_loads.py

Pytest suite for the Buddy Jr URDF (urdf/buddy_jr.urdf).

Covers:
- XML is parseable and the root element is <robot name='buddy_jr'>.
- Exactly one root link (base_link); the joint-link graph is acyclic.
- Exactly 4 revolute joints and 2 fixed joints (joint type census).
- Revolute joint names are exactly ['base_yaw','shoulder_pitch','elbow_pitch','camera_tilt']
  in that order (order matters for the downstream joint-index mapping).
- Revolute joint axes are Z, Y, Y, Y in joint order ([0,0,1] then three [0,1,0]).
- Every revolute joint has lower=-1.5708, upper=+1.5708, effort>0, velocity>0.
- Each of the 4 revolute joints has a unit-norm axis, lower < upper,
  effort > 0, velocity > 0  (parametrized over the joint names).
- Every <inertial> link has a positive-definite inertia tensor and its
  principal diagonal moments satisfy the triangle inequality.
- PyBullet DIRECT-mode load (skipped if pybullet is not installed) with joint
  count assertion.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Resolve the URDF path relative to the repo root (works regardless of where
# pytest is invoked from).
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_URDF_PATH = _REPO_ROOT / "urdf" / "buddy_jr.urdf"

# Expected joint configuration from the Buddy Jr kinematic design.
# base_yaw rotates about Z (vertical); shoulder, elbow, camera all pitch about Y.
_EXPECTED_REVOLUTE_NAMES: list[str] = [
    "base_yaw",
    "shoulder_pitch",
    "elbow_pitch",
    "camera_tilt",
]

# Axis vectors for each revolute joint in _EXPECTED_REVOLUTE_NAMES order.
_EXPECTED_REVOLUTE_AXES: list[list[float]] = [
    [0.0, 0.0, 1.0],  # base_yaw  – rotation about vertical Z
    [0.0, 1.0, 0.0],  # shoulder_pitch – pitch about Y
    [0.0, 1.0, 0.0],  # elbow_pitch    – pitch about Y
    [0.0, 1.0, 0.0],  # camera_tilt    – pitch about Y
]

# SG90 servo span: ±90 deg = ±π/2 rad, stored to 4 decimal places in the URDF.
_EXPECTED_LOWER: float = -1.5708
_EXPECTED_UPPER: float = +1.5708
_LIMIT_TOL: float = 1e-4  # tolerance for floating-point comparison of limits

# PyBullet counts ALL joints (revolute + fixed) when calling getNumJoints.
# 4 revolute + 2 fixed (camera_joint, camera_optical_joint) = 6 total.
_EXPECTED_TOTAL_JOINTS_PYBULLET: int = 6


@pytest.fixture(scope="module")
def urdf_root() -> ET.Element:
    """Parse buddy_jr.urdf once and share across all tests in this module."""
    assert _URDF_PATH.is_file(), (
        f"URDF not found at {_URDF_PATH}. "
        "Has the file been relocated? Expected: urdf/buddy_jr.urdf"
    )
    tree = ET.parse(_URDF_PATH)
    return tree.getroot()


# ---------------------------------------------------------------------------
# 1. Basic XML / robot-element checks
# ---------------------------------------------------------------------------


def test_xml_is_parseable() -> None:
    """The URDF file must be well-formed XML."""
    ET.parse(_URDF_PATH)  # raises ET.ParseError on bad XML


def test_root_element_is_robot(urdf_root: ET.Element) -> None:
    """The root XML element must be <robot>."""
    assert urdf_root.tag == "robot", f"Expected <robot>, got <{urdf_root.tag}>"


def test_robot_name(urdf_root: ET.Element) -> None:
    """The robot must be named 'buddy_jr'."""
    assert urdf_root.get("name") == "buddy_jr"


# ---------------------------------------------------------------------------
# 2. Single root link + acyclic tree
# ---------------------------------------------------------------------------


def test_single_root_link_is_base_link(urdf_root: ET.Element) -> None:
    """There must be exactly one root link and it must be named 'base_link'.

    A root link is any link that is never listed as a <child> in a joint.
    """
    all_links = {link.get("name") for link in urdf_root.findall("link")}
    child_links = {
        j.find("child").get("link")
        for j in urdf_root.findall("joint")
        if j.find("child") is not None
    }
    root_links = all_links - child_links
    assert root_links == {
        "base_link"
    }, f"Expected exactly one root link called 'base_link', found: {root_links}"


def test_joint_tree_is_acyclic(urdf_root: ET.Element) -> None:
    """The kinematic tree must be acyclic and all links reachable from base_link.

    Uses a depth-first search; any repeated visit signals a cycle.
    """
    children: dict[str, list[str]] = {link.get("name"): [] for link in urdf_root.findall("link")}
    for joint in urdf_root.findall("joint"):
        p_el = joint.find("parent")
        c_el = joint.find("child")
        if p_el is not None and c_el is not None:
            p = p_el.get("link")
            c = c_el.get("link")
            if p in children:
                children[p].append(c)

    all_links = set(children.keys())
    visited: set[str] = set()
    stack = ["base_link"]

    while stack:
        node = stack.pop()
        assert node not in visited, f"Cycle detected: '{node}' visited more than once"
        visited.add(node)
        stack.extend(children.get(node, []))

    assert visited == all_links, f"Unreachable links from base_link: {all_links - visited}"


# ---------------------------------------------------------------------------
# 3. Joint type census: exactly 4 revolute and 2 fixed
# ---------------------------------------------------------------------------


def test_exactly_four_revolute_joints(urdf_root: ET.Element) -> None:
    """The URDF must declare exactly 4 revolute joints.

    Buddy Jr has:
      base_yaw, shoulder_pitch, elbow_pitch, camera_tilt (all revolute)
    Any deviation (e.g. a new joint added without updating this test) should
    be a deliberate, reviewed change to keep the downstream joint-index mapping
    in sync with rl_lab.robot.buddy_jr.JOINTS.
    """
    revolute_joints = [j for j in urdf_root.findall("joint") if j.get("type") == "revolute"]
    assert len(revolute_joints) == 4, (
        f"Expected 4 revolute joints, found {len(revolute_joints)}: "
        f"{[j.get('name') for j in revolute_joints]}"
    )


def test_exactly_two_fixed_joints(urdf_root: ET.Element) -> None:
    """The URDF must declare exactly 2 fixed joints.

    Buddy Jr has:
      camera_joint (camera_mount -> camera_link)
      camera_optical_joint (camera_link -> camera_optical_frame)
    These anchor the end-effector frame used by the IK and RL reward functions.
    """
    fixed_joints = [j for j in urdf_root.findall("joint") if j.get("type") == "fixed"]
    assert len(fixed_joints) == 2, (
        f"Expected 2 fixed joints, found {len(fixed_joints)}: "
        f"{[j.get('name') for j in fixed_joints]}"
    )


# ---------------------------------------------------------------------------
# 4. Revolute joint name order
# ---------------------------------------------------------------------------


def test_revolute_joint_names_and_order(urdf_root: ET.Element) -> None:
    """Revolute joints must appear in the URDF with the exact names and order.

    The order determines the joint-index mapping used throughout rl_lab
    (rl_lab.robot.buddy_jr.JOINTS, gym observations, PyBullet joint indices).
    Changing either a name or the order without updating this test and the
    downstream code will silently break the whole stack.
    """
    revolute_joints = [j for j in urdf_root.findall("joint") if j.get("type") == "revolute"]
    actual_names = [j.get("name") for j in revolute_joints]
    assert actual_names == _EXPECTED_REVOLUTE_NAMES, (
        f"Revolute joint names/order mismatch.\n"
        f"  expected : {_EXPECTED_REVOLUTE_NAMES}\n"
        f"  got      : {actual_names}"
    )


# ---------------------------------------------------------------------------
# 5. Revolute joint axes (Z then Y, Y, Y)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "joint_name, expected_axis",
    list(zip(_EXPECTED_REVOLUTE_NAMES, _EXPECTED_REVOLUTE_AXES, strict=True)),
)
def test_revolute_joint_axis_direction(
    urdf_root: ET.Element, joint_name: str, expected_axis: list[float]
) -> None:
    """Each revolute joint must rotate about the expected world axis.

    base_yaw rotates the whole arm about the vertical (Z) axis.
    The remaining three joints all pitch about the lateral (Y) axis.
    An incorrect axis will produce wrong kinematics even if the limits look fine.
    """
    j = _get_joint(urdf_root, joint_name)
    axis_el = j.find("axis")
    assert axis_el is not None, f"Joint '{joint_name}': missing <axis>"
    raw = axis_el.get("xyz", "1 0 0")
    actual = np.array([float(v) for v in raw.split()])
    expected = np.array(expected_axis, dtype=float)
    assert np.allclose(actual, expected, atol=1e-6), (
        f"Joint '{joint_name}': axis mismatch.\n"
        f"  expected : {expected.tolist()}\n"
        f"  got      : {actual.tolist()}"
    )


# ---------------------------------------------------------------------------
# 6. Revolute joint exact limits (+/-1.5708, effort>0, velocity>0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("joint_name", _EXPECTED_REVOLUTE_NAMES)
def test_revolute_joint_exact_limits(urdf_root: ET.Element, joint_name: str) -> None:
    """Every revolute joint must have lower=-1.5708 and upper=+1.5708.

    These values correspond to ±90 deg (π/2 rad), matching the SG90 servo
    travel defined in rl_lab.robot.buddy_jr.JOINT_LIMIT.  A mismatch here
    means the URDF and the Python constants are out of sync, which breaks
    the clamping logic and the gym observation/action spaces.
    """
    j = _get_joint(urdf_root, joint_name)
    limit_el = j.find("limit")
    assert limit_el is not None, f"Joint '{joint_name}': missing <limit>"
    lower = float(limit_el.get("lower", 0))
    upper = float(limit_el.get("upper", 0))
    effort = float(limit_el.get("effort", 0))
    velocity = float(limit_el.get("velocity", 0))

    assert (
        abs(lower - _EXPECTED_LOWER) < _LIMIT_TOL
    ), f"Joint '{joint_name}': lower limit is {lower:.6f}, expected {_EXPECTED_LOWER}"
    assert (
        abs(upper - _EXPECTED_UPPER) < _LIMIT_TOL
    ), f"Joint '{joint_name}': upper limit is {upper:.6f}, expected {_EXPECTED_UPPER}"
    assert effort > 0.0, f"Joint '{joint_name}': effort={effort} must be > 0"
    assert velocity > 0.0, f"Joint '{joint_name}': velocity={velocity} must be > 0"


# ---------------------------------------------------------------------------
# 7. Revolute joint checks kept from original suite (parametrized)
#    (unit-norm axis, lower < upper, effort > 0, velocity > 0)
#    These overlap with the new exact-limit test but are kept because they
#    are more readable as independent failure messages for learners.
# ---------------------------------------------------------------------------

_REVOLUTE_JOINTS = _EXPECTED_REVOLUTE_NAMES  # alias kept for clarity


def _get_joint(root: ET.Element, joint_name: str) -> ET.Element:
    """Return the <joint> element with the given name, or fail the test."""
    for j in root.findall("joint"):
        if j.get("name") == joint_name:
            return j
    pytest.fail(f"Joint '{joint_name}' not found in URDF")


@pytest.mark.parametrize("joint_name", _REVOLUTE_JOINTS)
def test_revolute_joint_exists_and_type(urdf_root: ET.Element, joint_name: str) -> None:
    """Each of the 4 required revolute joints must be present with type='revolute'."""
    j = _get_joint(urdf_root, joint_name)
    assert (
        j.get("type") == "revolute"
    ), f"Joint '{joint_name}' should be 'revolute', got '{j.get('type')}'"


@pytest.mark.parametrize("joint_name", _REVOLUTE_JOINTS)
def test_revolute_joint_axis_is_unit_vector(urdf_root: ET.Element, joint_name: str) -> None:
    """The <axis xyz='...'> for each revolute joint must be a unit vector.

    The URDF spec allows any non-zero axis; physics engines normalise it
    internally, but an off-by-magnitude error often indicates a copy-paste
    mistake, so we enforce it here.
    """
    j = _get_joint(urdf_root, joint_name)
    axis_el = j.find("axis")
    assert axis_el is not None, f"Joint '{joint_name}': missing <axis>"

    raw = axis_el.get("xyz", "1 0 0")
    vec = np.array([float(v) for v in raw.split()])
    norm = float(np.linalg.norm(vec))
    assert (
        abs(norm - 1.0) < 1e-6
    ), f"Joint '{joint_name}': axis {vec.tolist()} has norm {norm:.6f}, expected 1.0"


@pytest.mark.parametrize("joint_name", _REVOLUTE_JOINTS)
def test_revolute_joint_limits_lower_less_than_upper(
    urdf_root: ET.Element, joint_name: str
) -> None:
    """Each revolute joint must have lower < upper limits."""
    j = _get_joint(urdf_root, joint_name)
    limit_el = j.find("limit")
    assert limit_el is not None, f"Joint '{joint_name}': missing <limit>"

    lower = float(limit_el.get("lower", 0))
    upper = float(limit_el.get("upper", 0))
    assert lower < upper, f"Joint '{joint_name}': lower ({lower}) must be < upper ({upper})"


@pytest.mark.parametrize("joint_name", _REVOLUTE_JOINTS)
def test_revolute_joint_effort_positive(urdf_root: ET.Element, joint_name: str) -> None:
    """Effort (torque) limit must be positive -- a zero means the joint cannot move."""
    j = _get_joint(urdf_root, joint_name)
    limit_el = j.find("limit")
    assert limit_el is not None, f"Joint '{joint_name}': missing <limit>"
    effort = float(limit_el.get("effort", 0))
    assert effort > 0, f"Joint '{joint_name}': effort={effort} must be > 0"


@pytest.mark.parametrize("joint_name", _REVOLUTE_JOINTS)
def test_revolute_joint_velocity_positive(urdf_root: ET.Element, joint_name: str) -> None:
    """Velocity limit must be positive -- a zero means the joint cannot move."""
    j = _get_joint(urdf_root, joint_name)
    limit_el = j.find("limit")
    assert limit_el is not None, f"Joint '{joint_name}': missing <limit>"
    velocity = float(limit_el.get("velocity", 0))
    assert velocity > 0, f"Joint '{joint_name}': velocity={velocity} must be > 0"


# ---------------------------------------------------------------------------
# 8. Inertia tensor checks
# ---------------------------------------------------------------------------


def _get_inertial_links(root: ET.Element) -> list[tuple[str, ET.Element]]:
    """Return (link_name, inertial_element) pairs for links that have <inertial>."""
    result = []
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is not None:
            result.append((link.get("name"), inertial))
    return result


def _parse_inertia(inertia_el: ET.Element) -> np.ndarray:
    """Return the 3x3 symmetric inertia matrix from an <inertia> element."""
    ixx = float(inertia_el.get("ixx", 0))
    ixy = float(inertia_el.get("ixy", 0))
    ixz = float(inertia_el.get("ixz", 0))
    iyy = float(inertia_el.get("iyy", 0))
    iyz = float(inertia_el.get("iyz", 0))
    izz = float(inertia_el.get("izz", 0))
    return np.array(
        [
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ]
    )


def test_all_inertials_positive_definite(urdf_root: ET.Element) -> None:
    """Every <inertial> link must have a positive-definite inertia tensor.

    A positive-definite tensor means all eigenvalues are strictly positive,
    which is required by physics for a rigid body with non-zero extent.
    """
    links_with_inertial = _get_inertial_links(urdf_root)
    assert links_with_inertial, "No links with <inertial> found -- expected at least one"

    for link_name, inertial in links_with_inertial:
        inertia_el = inertial.find("inertia")
        assert inertia_el is not None, f"Link '{link_name}': <inertial> missing <inertia>"
        inertia_tensor = _parse_inertia(inertia_el)
        eigenvalues = np.linalg.eigvalsh(inertia_tensor)
        assert np.all(eigenvalues > 0), (
            f"Link '{link_name}': inertia tensor is not positive-definite. "
            f"Eigenvalues: {eigenvalues.tolist()}"
        )


def test_all_inertials_triangle_inequality(urdf_root: ET.Element) -> None:
    """Diagonal inertia moments must satisfy the triangle inequality.

    For a rigid body the principal moments Ixx, Iyy, Izz must satisfy:
        Ixx + Iyy >= Izz
        Iyy + Izz >= Ixx
        Ixx + Izz >= Iyy
    Violation indicates an impossible (non-physical) mass distribution.
    """
    _EPS = 1e-6  # small tolerance for floating-point rounding

    for link_name, inertial in _get_inertial_links(urdf_root):
        inertia_el = inertial.find("inertia")
        assert inertia_el is not None, f"Link '{link_name}': <inertial> missing <inertia>"

        ixx = float(inertia_el.get("ixx", 0))
        iyy = float(inertia_el.get("iyy", 0))
        izz = float(inertia_el.get("izz", 0))

        assert (
            ixx + iyy + _EPS >= izz
        ), f"Link '{link_name}': Ixx+Iyy < Izz ({ixx:.3e}+{iyy:.3e} < {izz:.3e})"
        assert (
            iyy + izz + _EPS >= ixx
        ), f"Link '{link_name}': Iyy+Izz < Ixx ({iyy:.3e}+{izz:.3e} < {ixx:.3e})"
        assert (
            ixx + izz + _EPS >= iyy
        ), f"Link '{link_name}': Ixx+Izz < Iyy ({ixx:.3e}+{izz:.3e} < {iyy:.3e})"


# ---------------------------------------------------------------------------
# 9. PyBullet headless load test
# ---------------------------------------------------------------------------


def test_pybullet_load_urdf() -> None:
    """Load buddy_jr.urdf in PyBullet DIRECT mode (headless) and check joint count.

    This is the most thorough integration check: PyBullet validates link
    geometry, joint parent/child references, and mass properties together.

    The test also asserts that PyBullet reports the expected total number of
    joints (revolute + fixed = 4 + 2 = 6).  PyBullet's getNumJoints returns
    ALL joint types; the count must match _EXPECTED_TOTAL_JOINTS_PYBULLET so
    that the downstream joint-index look-ups in rl_lab.sim.pybullet_backend
    remain in sync with the URDF.

    The test is automatically skipped if pybullet is not installed so that
    the pure-Python CI environment can still run the rest of the suite.
    (PyBullet has no macOS arm64 wheel; CI tests this section on Linux only.)
    """
    p = pytest.importorskip("pybullet", reason="pybullet not installed")

    cid = p.connect(p.DIRECT)
    try:
        robot_id = p.loadURDF(str(_URDF_PATH), useFixedBase=True, physicsClientId=cid)
        assert robot_id >= 0, "PyBullet loadURDF returned negative id -- unknown error"

        num_joints = p.getNumJoints(robot_id, physicsClientId=cid)
        assert num_joints == _EXPECTED_TOTAL_JOINTS_PYBULLET, (
            f"PyBullet reports {num_joints} joints; expected {_EXPECTED_TOTAL_JOINTS_PYBULLET} "
            f"(4 revolute + 2 fixed).  If the URDF gained or lost joints, update "
            f"_EXPECTED_TOTAL_JOINTS_PYBULLET in this file AND the joint-index mapping in "
            f"rl_lab/sim/pybullet_backend.py."
        )
    finally:
        p.disconnect(cid)

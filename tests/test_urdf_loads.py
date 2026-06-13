"""tests/test_urdf_loads.py

Pytest suite for the Buddy Jr URDF (urdf/buddy_jr.urdf).

Covers:
- XML is parseable and the root element is <robot name='buddy_jr'>.
- Exactly one root link (base_link); the joint-link graph is acyclic.
- Each of the 4 revolute joints has a unit-norm axis, lower < upper,
  effort > 0, velocity > 0  (parametrized over the joint names).
- Every <inertial> link has a positive-definite inertia tensor and its
  principal diagonal moments satisfy the triangle inequality.
- PyBullet DIRECT-mode load (skipped if pybullet is not installed).
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
# 3. Revolute joint checks (parametrized)
# ---------------------------------------------------------------------------

_REVOLUTE_JOINTS = ["base_yaw", "shoulder_pitch", "elbow_pitch", "camera_tilt"]


def _get_joint(root: ET.Element, joint_name: str) -> ET.Element:
    """Return the <joint> element with the given name, or fail."""
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
# 4. Inertia tensor checks
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
# 5. PyBullet load test
# ---------------------------------------------------------------------------


def test_pybullet_load_urdf() -> None:
    """Load buddy_jr.urdf in PyBullet DIRECT mode (headless).

    This is the most thorough integration check: PyBullet validates link
    geometry, joint parent/child references, and mass properties together.
    The test is automatically skipped if pybullet is not installed so that
    the pure-Python CI environment can still run the rest of the suite.
    """
    p = pytest.importorskip("pybullet", reason="pybullet not installed")

    cid = p.connect(p.DIRECT)
    try:
        robot_id = p.loadURDF(str(_URDF_PATH), useFixedBase=True, physicsClientId=cid)
        assert robot_id >= 0, "PyBullet loadURDF returned negative id -- unknown error"
    finally:
        p.disconnect(cid)

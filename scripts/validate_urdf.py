#!/usr/bin/env python3
"""validate_urdf.py -- Standalone validator for Buddy Jr's URDF.

Usage
-----
    python scripts/validate_urdf.py                   # default: urdf/buddy_jr.urdf
    python scripts/validate_urdf.py path/to/other.urdf

What it checks
--------------
1. The XML is well-formed and parseable.
2. There is exactly one root link (base_link), the joint graph is acyclic
   (forms a valid kinematic tree).
3. Each of the 4 revolute joints has:
   - A unit-norm axis vector (|axis| ~= 1.0).
   - lower < upper limits.
   - effort > 0 and velocity > 0.
4. Every link with an <inertial> element has a positive-definite 3x3 inertia
   tensor and satisfies the triangle inequality on principal moments:
     Ixx + Iyy >= Izz,  Iyy + Izz >= Ixx,  Ixx + Izz >= Iyy.
5. (Optional) PyBullet DIRECT-mode load -- skipped gracefully if pybullet
   is not installed.

Exit code: 0 on full pass, non-zero on any failure.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Tolerance for floating-point comparisons.
_EPS = 1e-6

# Expected revolute joint names (order doesn't matter for XML checks).
EXPECTED_REVOLUTE_JOINTS = {
    "base_yaw",
    "shoulder_pitch",
    "elbow_pitch",
    "camera_tilt",
}


def _fail(msg: str) -> None:
    """Print a failure line and record that at least one check failed."""
    print(f"  FAIL  {msg}")
    _FAILURES.append(msg)


def _pass(msg: str) -> None:
    """Print a passing line."""
    print(f"  ok    {msg}")


# Accumulate failures so we print a summary at the end.
_FAILURES: list[str] = []


# ---------------------------------------------------------------------------
# XML structural checks
# ---------------------------------------------------------------------------


def check_single_root(root: ET.Element) -> str | None:
    """Return the name of the single root link, or None on failure.

    A root link is one that never appears as a <child> in any joint.
    """
    all_links = {link.get("name") for link in root.findall("link")}
    child_links = {
        j.find("child").get("link") for j in root.findall("joint") if j.find("child") is not None
    }  # noqa: E501
    root_links = all_links - child_links

    if len(root_links) != 1:
        _fail(f"Expected exactly 1 root link, found {len(root_links)}: {root_links}")
        return None

    (root_link_name,) = root_links
    if root_link_name != "base_link":
        _fail(f"Root link should be 'base_link', got '{root_link_name}'")
        return None

    _pass(f"Single root link: '{root_link_name}'")
    return root_link_name


def check_acyclic(root: ET.Element) -> bool:
    """Verify the joint-link graph is a proper tree (no cycles).

    Performs a depth-first traversal from base_link; any unvisited link
    after traversal indicates a disconnected or cyclic structure.
    """
    # Build parent->children map (link name -> [child link names])
    children: dict[str, list[str]] = {link.get("name"): [] for link in root.findall("link")}
    for joint in root.findall("joint"):
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            continue
        p = parent_el.get("link")
        c = child_el.get("link")
        if p in children:
            children[p].append(c)

    all_links = set(children.keys())
    visited: set[str] = set()

    # DFS from base_link
    stack = ["base_link"]
    while stack:
        node = stack.pop()
        if node in visited:
            _fail(f"Cycle detected at link '{node}'")
            return False
        visited.add(node)
        stack.extend(children.get(node, []))

    if visited != all_links:
        unreachable = all_links - visited
        _fail(f"Unreachable links (disconnected tree?): {unreachable}")
        return False

    _pass("Joint-link tree is acyclic and fully connected")
    return True


# ---------------------------------------------------------------------------
# Revolute joint checks
# ---------------------------------------------------------------------------


def check_revolute_joints(root: ET.Element) -> None:
    """Assert all expected revolute joints are present and well-formed."""
    rev_joints = {j.get("name"): j for j in root.findall("joint") if j.get("type") == "revolute"}

    found = set(rev_joints.keys())
    missing = EXPECTED_REVOLUTE_JOINTS - found
    extra = found - EXPECTED_REVOLUTE_JOINTS
    if missing:
        _fail(f"Missing expected revolute joint(s): {missing}")
    if extra:
        _fail(f"Unexpected revolute joint(s): {extra}")

    for name in sorted(EXPECTED_REVOLUTE_JOINTS):
        if name not in rev_joints:
            continue
        j = rev_joints[name]
        _check_joint_axis(name, j)
        _check_joint_limits(name, j)


def _check_joint_axis(name: str, joint: ET.Element) -> None:
    """Axis vector must be a unit vector (L2-norm == 1)."""
    axis_el = joint.find("axis")
    if axis_el is None:
        _fail(f"Joint '{name}': missing <axis> element")
        return

    raw = axis_el.get("xyz", "1 0 0")  # URDF default is 1 0 0
    try:
        vec = np.array([float(v) for v in raw.split()])
    except ValueError:
        _fail(f"Joint '{name}': <axis xyz='{raw}'> is not parseable as floats")
        return

    norm = float(np.linalg.norm(vec))
    if abs(norm - 1.0) > _EPS:
        _fail(f"Joint '{name}': axis {vec.tolist()} has norm {norm:.6f}, expected 1.0")
    else:
        _pass(f"Joint '{name}': axis {vec.tolist()} is unit-norm (norm={norm:.6f})")


def _check_joint_limits(name: str, joint: ET.Element) -> None:
    """lower < upper, effort > 0, velocity > 0."""
    limit_el = joint.find("limit")
    if limit_el is None:
        _fail(f"Joint '{name}': missing <limit> element")
        return

    try:
        lower = float(limit_el.get("lower", 0))
        upper = float(limit_el.get("upper", 0))
        effort = float(limit_el.get("effort", 0))
        velocity = float(limit_el.get("velocity", 0))
    except ValueError as exc:
        _fail(f"Joint '{name}': cannot parse <limit> attributes: {exc}")
        return

    if lower < upper:
        _pass(f"Joint '{name}': limits [{lower}, {upper}] rad -- lower < upper")
    else:
        _fail(f"Joint '{name}': lower ({lower}) >= upper ({upper})")

    if effort > 0:
        _pass(f"Joint '{name}': effort={effort} > 0")
    else:
        _fail(f"Joint '{name}': effort={effort} must be > 0")

    if velocity > 0:
        _pass(f"Joint '{name}': velocity={velocity} > 0")
    else:
        _fail(f"Joint '{name}': velocity={velocity} must be > 0")


# ---------------------------------------------------------------------------
# Inertia checks
# ---------------------------------------------------------------------------


def check_inertials(root: ET.Element) -> None:
    """For every link with <inertial>, verify the inertia tensor is valid."""
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        link_name = link.get("name")
        _check_single_inertial(link_name, inertial)


def _check_single_inertial(link_name: str, inertial: ET.Element) -> None:
    """Positive-definite check + triangle inequality on principal moments."""
    inertia_el = inertial.find("inertia")
    if inertia_el is None:
        _fail(f"Link '{link_name}': <inertial> has no <inertia> child")
        return

    try:
        ixx = float(inertia_el.get("ixx", 0))
        ixy = float(inertia_el.get("ixy", 0))
        ixz = float(inertia_el.get("ixz", 0))
        iyy = float(inertia_el.get("iyy", 0))
        iyz = float(inertia_el.get("iyz", 0))
        izz = float(inertia_el.get("izz", 0))
    except ValueError as exc:
        _fail(f"Link '{link_name}': cannot parse <inertia> attributes: {exc}")
        return

    # Reconstruct the symmetric 3x3 tensor
    inertia_tensor = np.array(
        [
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ]
    )

    # Positive-definite: all eigenvalues > 0
    eigenvalues = np.linalg.eigvalsh(inertia_tensor)  # stable for symmetric matrices
    if np.all(eigenvalues > 0):
        _pass(
            f"Link '{link_name}': inertia tensor positive-definite (min eigenvalue={eigenvalues.min():.3e})"
        )
    else:
        _fail(
            f"Link '{link_name}': inertia tensor NOT positive-definite "
            f"(eigenvalues={eigenvalues.tolist()})"
        )

    # Triangle inequality on the *diagonal* (physical principal moments).
    # For a diagonal-dominant tensor (as produced by CAD/physics engines for
    # primitive shapes) this is equivalent to the bounding-sphere test:
    #   no single principal moment may exceed the sum of the other two.
    if (ixx + iyy + _EPS >= izz) and (iyy + izz + _EPS >= ixx) and (ixx + izz + _EPS >= iyy):
        _pass(f"Link '{link_name}': principal moments satisfy triangle inequality")
    else:
        _fail(
            f"Link '{link_name}': triangle inequality violated "
            f"(Ixx={ixx:.3e} Iyy={iyy:.3e} Izz={izz:.3e})"
        )


# ---------------------------------------------------------------------------
# PyBullet load check
# ---------------------------------------------------------------------------


def check_pybullet_load(urdf_path: Path) -> None:
    """Attempt to load the URDF in PyBullet DIRECT mode.

    Falls back to a skip (not a failure) if pybullet is not installed.
    This is the most comprehensive sanity check -- PyBullet will reject
    malformed geometry, bad joint parent/child references, etc.
    """
    try:
        import pybullet as p  # type: ignore[import]
    except ImportError:
        print(
            "  skip  PyBullet load test (pybullet not installed -- install with: pip install pybullet)"
        )
        return

    cid = p.connect(p.DIRECT)
    try:
        robot_id = p.loadURDF(str(urdf_path), useFixedBase=True, physicsClientId=cid)
        if robot_id >= 0:
            _pass(f"PyBullet loaded URDF successfully (bodyUniqueId={robot_id})")
        else:
            _fail("PyBullet loadURDF returned a negative id (unknown error)")
    except Exception as exc:  # noqa: BLE001
        _fail(f"PyBullet loadURDF raised: {exc}")
    finally:
        p.disconnect(cid)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _find_repo_root() -> Path:
    """Walk up from this script's location to find the repo root.

    We look for the directory that contains both 'urdf/' and 'pyproject.toml'
    so the script works when run from any subdirectory.
    """
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent, here.parent.parent]:
        if (candidate / "urdf").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    # Last resort: cwd
    return Path.cwd()


def main() -> None:
    repo_root = _find_repo_root()
    default_urdf = repo_root / "urdf" / "buddy_jr.urdf"

    parser = argparse.ArgumentParser(
        description="Validate the Buddy Jr URDF file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Checks performed:\n"
            "  1. Well-formed XML\n"
            "  2. Single root link named base_link; acyclic joint tree\n"
            "  3. 4 revolute joints with unit-norm axes and valid limits\n"
            "  4. Positive-definite inertia tensors satisfying triangle inequality\n"
            "  5. PyBullet DIRECT-mode load (skipped if pybullet not installed)\n"
        ),
    )
    parser.add_argument(
        "urdf",
        nargs="?",
        default=str(default_urdf),
        help=f"Path to URDF file (default: {default_urdf})",
    )
    args = parser.parse_args()

    urdf_path = Path(args.urdf).resolve()
    if not urdf_path.is_file():
        print(f"ERROR: URDF not found: {urdf_path}")
        sys.exit(2)

    print(f"\nValidating: {urdf_path}\n")

    # 1. Parse XML
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        _pass("XML is well-formed")
    except ET.ParseError as exc:
        _fail(f"XML parse error: {exc}")
        _print_summary()
        sys.exit(1)

    # Confirm this is a <robot> element
    if root.tag != "robot":
        _fail(f"Root element should be <robot>, got <{root.tag}>")
    else:
        _pass(f"Root element is <robot name='{root.get('name')}'>")

    # 2. Single root link + acyclic tree
    print()
    check_single_root(root)
    check_acyclic(root)

    # 3. Revolute joints
    print()
    print("Revolute joint checks:")
    check_revolute_joints(root)

    # 4. Inertia tensors
    print()
    print("Inertia tensor checks:")
    check_inertials(root)

    # 5. PyBullet
    print()
    print("PyBullet load check:")
    check_pybullet_load(urdf_path)

    _print_summary()

    if _FAILURES:
        sys.exit(1)


def _print_summary() -> None:
    print()
    if _FAILURES:
        print(f"RESULT: FAIL -- {len(_FAILURES)} check(s) failed:")
        for f in _FAILURES:
            print(f"  - {f}")
    else:
        print("RESULT: PASS -- all checks passed.")


if __name__ == "__main__":
    main()

# Releasing rl-lab to PyPI

This page explains the versioning policy, the tag-to-release workflow, and
how to set up the one-time PyPI trusted-publisher configuration.  You do not
need an API token — authentication uses OpenID Connect (OIDC).

---

## Versioning policy

rl-lab follows [Semantic Versioning 2.0.0](https://semver.org/):

```
MAJOR.MINOR.PATCH[-pre]
```

| Segment | When to bump |
|---------|--------------------------------------------------------------|
| `MAJOR` | Breaking changes to the public API (env IDs, CLI flags, URDF joint names, deploy format). |
| `MINOR` | New experiments, new algorithms, new env variants, backwards-compatible CLI additions. |
| `PATCH` | Bug fixes, doc corrections, dependency pin updates, typos. |
| `-pre`  | Optional pre-release suffix: `-alpha1`, `-beta2`, `-rc1`. These are published to PyPI but pip does not install them by default unless you pass `--pre`. |

Examples:

- `v0.1.0` — first public release (all M1-M6 milestones)
- `v0.2.0` — adds a new experiment (Experiment 13)
- `v0.2.1` — fixes a bug in the SAC checkpoint path
- `v1.0.0` — stable API declared after the first real-world deployment
- `v1.0.0-rc1` — release candidate for community testing

> **Note:** The version string in `pyproject.toml` must match the tag
> exactly (without the leading `v`).  For example, tag `v0.1.0` requires
> `version = "0.1.0"` in `pyproject.toml`.

---

## Pre-release checklist

Before pushing a tag, work through this list on your local machine:

1. **CI is green.**  Check the Actions tab — all `CI` workflow jobs must
   pass on `main`.

2. **Update the version in `pyproject.toml`.**

   ```toml
   [project]
   version = "0.2.0"   # <-- change this
   ```

3. **Update `CHANGELOG.md`.**  Add a `## [0.2.0] — YYYY-MM-DD` section
   summarising what changed.  Follow the [Keep a Changelog](https://keepachangelog.com/)
   format already used in the file.

4. **Commit the version bump.**

   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "chore: bump version to 0.2.0"
   git push origin main
   ```

5. **Wait for CI to pass** on the bump commit before tagging.

---

## Pushing a release tag

Tags must be annotated (`-a`) so GitHub records them as releases:

```bash
# Replace 0.2.0 with your version.
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

Pushing the tag triggers the `Release` GitHub Actions workflow
(`.github/workflows/release.yml`).  You can watch it under
**Actions > Release**.

---

## What the workflow does

```
push tag v*
    │
    ▼
┌─────────────────────────────────────────────┐
│  Job: build  (ubuntu-latest, Python 3.12)   │
│                                             │
│  1. actions/checkout@v4                     │
│  2. python -m pip install build             │
│  3. python -m build  →  dist/*.whl + .tar.gz│
│  4. actions/upload-artifact@v4  (name=dist) │
└──────────────────┬──────────────────────────┘
                   │  needs: build
                   ▼
┌─────────────────────────────────────────────┐
│  Job: publish  (environment: pypi)          │
│                                             │
│  1. actions/download-artifact@v4            │
│  2. pypa/gh-action-pypi-publish@release/v1  │
│     → OIDC token, no API key required       │
└─────────────────────────────────────────────┘
```

The `publish` job runs inside the **`pypi`** GitHub Actions environment,
which you can protect with required reviewers or tag-pattern rules.  The
`id-token: write` permission is the only special permission required —
it lets the runner request a short-lived OIDC token from GitHub and
present it to PyPI instead of a long-lived secret.

---

## One-time PyPI trusted-publisher setup

You only need to do this once, the first time you release.

1. Log in to [pypi.org](https://pypi.org) as `kevinmcaleer`.
2. Go to **Your projects > rl-lab > Manage > Publishing**.
   - If the project does not exist yet on PyPI, use the
     ["pending publisher"](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
     flow instead (same form, no project needed first).
3. Click **Add a new publisher** and fill in:

   | Field | Value |
   |---|---|
   | Owner | `kevinmcaleer` |
   | Repository | `rl_lab` |
   | Workflow filename | `release.yml` |
   | Environment name | `pypi` |

4. Save.  That is all — no token to copy, rotate, or leak.

### GitHub environment

Create the `pypi` environment in your repository:

1. **Settings > Environments > New environment**, name it `pypi`.
2. Optionally add yourself as a **required reviewer** so every publish
   needs manual approval before it hits PyPI.
3. Optionally restrict it to tags matching `v*` under
   **Deployment branches and tags**.

---

## After the workflow succeeds

1. **Verify the release on PyPI:**
   <https://pypi.org/project/rl-lab/>

2. **Create a GitHub Release** (optional but recommended):

   ```bash
   gh release create v0.2.0 \
     --title "v0.2.0" \
     --notes-file CHANGELOG.md
   ```

   Or use the GitHub web UI: **Releases > Draft a new release**, select
   the tag you just pushed, and paste in the changelog section.

3. **Announce** on [kevsrobots.com](https://www.kevsrobots.com) and the
   community Discord if appropriate.

---

## Installing a specific release

```bash
# Latest stable
pip install rl-lab

# Exact version
pip install rl-lab==0.2.0

# Pre-release
pip install --pre rl-lab
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `403 Forbidden` from PyPI | Trusted publisher not configured, or environment name does not match | Re-check the four fields in the PyPI publisher form and the `environment: name:` in `release.yml` |
| `version already exists` from PyPI | Tag pushed for a version already on PyPI | Bump the version in `pyproject.toml`, commit, push a new tag |
| Workflow does not trigger | Tag does not start with `v` | Use `git tag v0.2.0` (note the `v`) |
| `dist/` is empty | `python -m build` failed silently | Check the build job logs; a missing `[build-system]` table in `pyproject.toml` is a common cause |
| `id-token: write` permission error | Workflow has `permissions: read-all` at the top level | The `publish` job sets its own `id-token: write` — remove any top-level permissions block that overrides job-level settings |

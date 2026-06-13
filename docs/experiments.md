# Experiments

The full, runnable curriculum lives in [`experiments/README.md`](../experiments/README.md).

## Difficulty presets

The reach environment ships three difficulty presets in `rl_lab/config/env/`,
varying the target distribution, success tolerance, episode length and reward
mode:

| Preset | Reward | Tolerance | Max steps | Target shell |
|--------|--------|-----------|-----------|--------------|
| `easy`   | dense  | 3 cm  | 150 | 0.06–0.10 m |
| `medium` | dense  | 2 cm  | 200 | 0.04–0.13 m |
| `hard`   | sparse | 1.5 cm| 300 | 0.04–0.155 m |

Load one in code (or, later, via the training CLI):

```python
from rl_lab.env.presets import make_env_from_preset
env = make_env_from_preset("easy")
```

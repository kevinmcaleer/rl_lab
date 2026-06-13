# The experiment ladder

Twelve progressive experiments, each runnable (`python experiments/NN_name.py`)
and each teaching one new idea. Work through them in order — every experiment
builds on the previous one.

| # | Experiment | You learn |
|---|------------|-----------|
| 1 | [Bandit base](experiments/01_bandit.md) | reward, action, explore vs. exploit, epsilon |
| 2 | [Build the world](experiments/02_world.md) | the environment as a first-class object; URDF + viewer |
| 3 | [Tabular Q-learning](experiments/03_qlearning.md) | the MDP, Q-tables, the Bellman update, gamma |
| 4 | [Reward shaping & the discretisation wall](experiments/04_reward_shaping.md) | reward design, reward hacking, the curse of dimensionality |
| 5 | [DQN](experiments/05_dqn.md) | function approximation, replay buffer, target networks |
| 6 | [Generalisation & domain randomisation](experiments/06_generalisation.md) | memorisation vs. generalisation; the first sim-to-real bridge |
| 7 | [REINFORCE from scratch](experiments/07_reinforce.md) | policy gradients, the log-prob trick, value baselines |
| 8 | [PPO](experiments/08_ppo.md) | actor-critic, GAE, the clipped surrogate objective |
| 9 | [Continuous PPO](experiments/09_ppo_continuous.md) | continuous actions, action scaling, smoothness penalties |
| 10 | [SAC + the full aim task](experiments/10_sac_aim.md) | off-policy continuous control, entropy, sample efficiency |
| 11 | [Closing the sim-to-real gap](experiments/11_robustify.md) | domain randomisation, noise, latency, safety clamps |
| 12 | [Deploy to real hardware](experiments/12_deploy.md) | inference-only export driving real SG90 servos |

Each experiment script takes `--quick` (a fast smoke run) and `--render foxglove`
(live 3D streaming). New to the ideas? Start with the [concept primers](concepts/mdps.md).

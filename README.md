
# pytorch_drl
## Model Free Deep Reinforcement Learning Algorithms

## Algorithms:
- [**Rainbow**](https://arxiv.org/abs/1710.02298) [[code](https://github.com/selim-karaduman/pytorch_drl/blob/master/pytorch_drl/algs/rainbow.py)] [[tests](https://github.com/selim-karaduman/pytorch_drl/blob/master/tests/rainbow_test.ipynb)]
    - [**DQN**](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) and
    - [**DDQN**](https://arxiv.org/abs/1509.06461) or
    - [**Dueling DQN**](https://arxiv.org/abs/1511.06581) or
    - [**Noisy Networks**](https://arxiv.org/abs/1706.10295) or
    - [**Prioritized Experience Replay**](https://arxiv.org/abs/1511.05952) or
    - [**N Step DQN**](https://arxiv.org/abs/1602.01783) or
    - [**Categorical DQN**](https://arxiv.org/abs/1707.06887) / [**Quantile Regression**](https://arxiv.org/abs/1710.10044)
- [**DDPG**](https://arxiv.org/abs/1509.02971) [[code](https://github.com/selim-karaduman/pytorch_drl/blob/master/pytorch_drl/algs/ddpg.py)] [[tests](https://github.com/selim-karaduman/pytorch_drl/blob/master/tests/ddpg_test.ipynb)]
- [**TD3**](https://arxiv.org/abs/1802.09477) [[code](https://github.com/selim-karaduman/pytorch_drl/blob/master/pytorch_drl/algs/td3.py)] [[tests](https://github.com/selim-karaduman/pytorch_drl/blob/master/tests/td3_test.ipynb)]
- [**SAC**](https://arxiv.org/abs/1801.01290) [[code](https://github.com/selim-karaduman/pytorch_drl/blob/master/pytorch_drl/algs/sac.py)] [[tests](https://github.com/selim-karaduman/pytorch_drl/blob/master/tests/sac_test.ipynb)]
- [**ACER**](https://arxiv.org/abs/1611.01224) [[code](https://github.com/selim-karaduman/pytorch_drl/blob/master/pytorch_drl/algs/acer.py)] [[tests](https://github.com/selim-karaduman/pytorch_drl/blob/master/tests/acer_test.ipynb)]
- [**A2C**](https://arxiv.org/abs/1602.01783) [[code](https://github.com/selim-karaduman/pytorch_drl/blob/master/pytorch_drl/algs/a2c.py)] [[tests](https://github.com/selim-karaduman/pytorch_drl/blob/master/tests/a2c_test.ipynb)]
- [**TRPO**](https://arxiv.org/abs/1502.05477) [[code](https://github.com/selim-karaduman/pytorch_drl/blob/master/pytorch_drl/algs/trpo.py)] [[tests](https://github.com/selim-karaduman/pytorch_drl/blob/master/tests/trpo_test.ipynb)]
- [**ACKTR**](https://arxiv.org/abs/1708.05144) [[code](https://github.com/selim-karaduman/pytorch_drl/blob/master/pytorch_drl/algs/acktr.py)] [[tests](https://github.com/selim-karaduman/pytorch_drl/blob/master/tests/acktr_test.ipynb)]
- [**PPO**](https://arxiv.org/abs/1707.06347) [[code](https://github.com/selim-karaduman/pytorch_drl/blob/master/pytorch_drl/algs/ppo.py)] [[tests](https://github.com/selim-karaduman/pytorch_drl/blob/master/tests/ppo_test.ipynb)]
- [**GAIL**](https://arxiv.org/abs/1606.03476) [[code](https://github.com/selim-karaduman/pytorch_drl/blob/master/pytorch_drl/algs/gail.py)] [[tests](https://github.com/selim-karaduman/pytorch_drl/blob/master/tests/gail_test.ipynb)]
- [**HER**](https://arxiv.org/abs/1707.01495) [[code](https://github.com/selim-karaduman/pytorch_drl/blob/master/pytorch_drl/algs/her.py)] [[tests](https://github.com/selim-karaduman/pytorch_drl/blob/master/tests/her_test.ipynb)]
- [**MADDPG**](https://arxiv.org/abs/1706.02275) [[code](https://github.com/selim-karaduman/pytorch_drl/blob/master/pytorch_drl/algs/maddpg.py)] [[tests](https://github.com/selim-karaduman/pytorch_drl/blob/master/tests/maddpg_test.ipynb)]

**Note:** Not tested thoroughly. Gail had problems. For tests check [**/tests**](https://github.com/selim-karaduman/pytorch_drl/tree/master/tests)

---
***

### Packages
- pytorch == 1.5
- gym ==  0.15.6
- [**multiagent particle environment**](https://github.com/openai/multiagent-particle-envs) (for MADDPG)
- numpy == 1.18

---
***

### Acknowledgements/References
- ACKTR: [KFAC](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/kfac.py) implementation of [@ikostrikov](https://github.com/ikostrikov)
- TRPO: For line search [this](https://github.com/ikostrikov/pytorch-trpo) is used as reference. [@ikostrikov](https://github.com/ikostrikov)
- [Noisy Linear Layers](https://github.com/higgsfield/RL-Adventure) [@higgsfield](https://github.com/higgsfield)
- [Parallel Environment](https://github.com/dolhana/udacity-deep-reinforcement-learning/blob/master/pong/parallelEnv.py). [@dolhana](https://github.com/dolhana)
- For PER: [segment tree](https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/segment_tree.py) [@hill-a](https://github.com/hill-a)
- For [policy_heads](https://github.com/selim-karaduman/pytorch_drl/blob/master/pytorch_drl/models/policy_head.py): [distribution modules](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/distributions.py) of [@ikostrikov](https://github.com/ikostrikov) is used.


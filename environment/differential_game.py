
import numpy as np
import gym, os, sys
sys.path.append(os.getcwd())
class DifferentialGame(gym.Env):

    def __init__(self, action_low=-10, action_high=10):

        self.timestep = 0
        self.num_agent = 2
        self.observation_spaces = gym.spaces.MultiDiscrete([1] * 2)
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, shape=(2,))

        assert self.num_agent == 2

    def calculate_reward(self, actions):
        a1, a2 = actions
        f1 = 0.8 * (-((a1 + 5) / 3) ** 2 - -((a2 + 5) / 3) ** 2)
        f2 = 1.0 * (-((a1 - 5) / 1) ** 2 - -((a2 - 5) / 1) ** 2) + 10
        return [max(f1, f2)] * 2

    def step(self, actions):
        assert len(actions) == self.num_agent
        actions = np.array(actions) * self.action_space.high
        rewards = self.calculate_reward(actions)
        obss = np.array(list([[0. * i] for i in range(self.num_agent)]))
        info = {}
        dones = np.array([True] * self.num_agent)
        self.timestep += 1
        return obss, rewards, dones, info

    def reset(self):
        self.timestep = 0
        return np.array(list([[0. * i] for i in range(self.num_agent)]))

if __name__ == '__main__':
    gym.envs.register(
        id='DiffGame-v0',
        entry_point='environment.differential_game:DifferentialGame',
        max_episode_steps=2,
    )
    env = gym.make('DiffGame-v0')
    obss = env.reset()
    actions = env.action_space.sample()
    print(actions, obss)
    obss, rews ,dones, info = env.step(actions)
    print(actions, obss, rews, dones)
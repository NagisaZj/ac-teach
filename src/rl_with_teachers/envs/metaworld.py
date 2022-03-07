import gym
from gym import spaces
from gym.envs.registration import register
import rl_with_teachers.teachers.metaworld as metateacher
import random
import numpy as np
import metaworld, pickle
import metaworld.policies  as p


class MetaWorldEnv:
    """
    A mujoco task with a fetch robot that needs to pick up a cube and move it to a
    goal point (or just push it there).
    """

    def __init__(self,task_name='push-v2',seed=0):
        print(task_name)
        mt1 = metaworld.MT1(task_name)  # Construct the benchmark, sampling tasks

        env = mt1.train_classes[task_name]()  # Create an environment with task `pick_place`
        task = random.choice(mt1.train_tasks)
        env.set_task(task)  # Set task
        env.seed(seed)
        self.env = env
        self.unwrapped = env
        self.random_seed=seed
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.spec = self.env
        self.spec.max_episode_steps = 500
        self.spec.id=task_name

    def seed(self,seed):
        return self.env.seed(seed)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return obs

    def _reset_sim(self):
        self.env._reset_sim()
        return True

    def _sample_goal(self):
        return None

    def make_teachers(self, type, noise=None, drop=0, num_random=0):
        paths = ['/data2/zj/NewMTRL/logs/90f2497ff4cee27c0d30fbc66e6ba205f94808ba4ea16e057df58e73_issue_None_seed_43_2/model',
                 '/data2/zj/NewMTRL/logs/90f2497ff4cee27c0d30fbc66e6ba205f94808ba4ea16e057df58e73_issue_None_seed_43_2/model',
                 '/data2/zj/NewMTRL/logs/90f2497ff4cee27c0d30fbc66e6ba205f94808ba4ea16e057df58e73_issue_None_seed_253/model']
        teachers = []
        with open('./config.txt', 'rb') as f:
            config = pickle.load(f)
        for i, p in enumerate(paths):
            if i==1:
                agent = metateacher.MTRLAgent(index=i, path=p,action_dim=self.action_space.shape[0],features=self.observation_space.shape[0],config=config)
                teachers.append(agent)


        # teachers = []
        # teachers.append(metateacher.ScriptAgent(p.SawyerPushV2Policy()))

        return teachers




register(
        id='MetaWorldEnv-v0',
        entry_point='rl_with_teachers.envs:MetaWorldEnv',
        max_episode_steps=500,
        )


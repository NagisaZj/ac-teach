from rl_with_teachers.teachers.base import TeacherPolicy
import numpy as np
# import torch, mtrl,hydra



class MTRLAgent(TeacherPolicy):
    def __init__(self, index, path,action_dim,features,config):
        self.id = index
        self.agent = hydra.utils.instantiate(
            config.agent.builder,
            env_obs_shape=(features,),
            action_shape=(action_dim,),
            action_range=[
                np.ones([action_dim]) * -1,
                np.ones([action_dim]),
            ],
            device=torch.device("cuda:0"),
        )
        self.agent.load_latest_step(model_dir=path)

    def __call__(self, obs):
        task_encoding = None  # type: ignore[assignment]
        obs = np.array(obs)[None]
        obs = torch.Tensor(obs).to(self.agent.device)
        id = torch.Tensor([self.id]).int()
        id = torch.unsqueeze(id, 0).to(self.agent.device)
        task_info = self.agent.get_task_info(
            task_encoding=task_encoding,
            component_name="critic",
            env_index=id,
        )

        obs = mtrl.agent.ds.mt_obs.MTObs(env_obs=obs, task_obs=id, task_info=task_info)
        mu_e, pi_e, log_pi_e, log_std_e = self.agent.actor.forward(mtobs=obs, detach_encoder=True)
        return pi_e.data.cpu().numpy()[0]
        # cube_pos = obs[3:6]
        # goal_pos = obs[-6:-3]
        # achieved = np.linalg.norm(goal_pos - cube_pos) < 0.01
        # if not self.picker.is_grasping(obs) and not achieved:
        #     action = self.picker(obs)
        # else:
        #     action = self.placer(obs)
        #
        # noisy_action = self.apply_noise(np.array(action))
        # noisy_action[-1] = action[-1]#0.95*action[-1]+0.05*noisy_action[-1]
        # return noisy_action

    def reset(self):
        return


class ScriptAgent(TeacherPolicy):
    def __init__(self, policy):
        self.policy = policy


    def __call__(self, obs):
        return self.policy.get_action(obs)

    def reset(self):
        return
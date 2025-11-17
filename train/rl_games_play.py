import yaml
import os
from argparse import ArgumentParser

from vlearn.gym import get_working_directory
import rl_games.torch_runner
from rl_games.common import env_configurations, vecenv
from rl_games.common.ivecenv import IVecEnv
import gym.spaces
import numpy as np
import time

import torch

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'


class ProxyEnv(IVecEnv):
    def __init__(self, config: dict):
        super().__init__()

        def create_box(space: dict):
            assert "shape" in space, "You must define the shape of the space in the yml file"
            shape = space["shape"]
            if "low" in space:
                low = np.array(space["low"])
            else:
                low = np.array([np.finfo('f').min] * shape, dtype=np.float32)
            if "high" in space:
                high = np.array(space["high"])
            else:
                high = np.array([np.finfo('f').max] * shape, dtype=np.float32)
            return gym.spaces.Box(low=low, high=high, shape=(shape, ))

        self.observation_space = create_box(config["observation_space"])
        self.action_space = create_box(config["action_space"])
        if "state_space" in config:
            self.state_space = create_box(config["state_space"])
        else:
            self.state_space = None

        self.env_info = {}
        self.env_info["observation_space"] = self.observation_space
        self.env_info["action_space"] = self.action_space
        self.env_info["state_space"] = self.state_space

    def get_env_info(self):
        return self.env_info


class ONNX_RLGames_Wrapper(torch.nn.Module):
    def __init__(self, model, is_deterministic, has_batch_dimension, clip_actions, actions_low, actions_high):
        super().__init__()

        self.model = model
        self.is_deterministic = is_deterministic
        self.has_batch_dimension = has_batch_dimension
        self.clip_actions = clip_actions
        self.actions_low = actions_low
        self.actions_high = actions_high

    def rescale_actions(self, low, high, action):
        d = (high - low) / 2.0
        m = (high + low) / 2.0
        scaled_action = action * d + m
        return scaled_action

    def forward(self, obs):
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': None
            }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        if self.is_deterministic:
            action = mu
        else:
            action = action
        if not self.has_batch_dimension:
            action = torch.squeeze(action.detach())
        if self.clip_actions:
            return self.rescale_actions(self.actions_low, self.actions_high, torch.clamp(action, -1.0, 1.0))
        else:
            return action


class RLGamesInference():
    # WARN: stripped out part, their interface may change
    def __init__(self, config, ckpt):
        print('Creating RL Games torch runner...')
        self.config = config

        name = config["params"]["config"]["name"]
        env_configurations.register(name, {
            "vecenv_type": "VLEARN",
            "env_creator": lambda: self.create_my_envs()})

        vecenv.register("VLEARN", lambda: ProxyEnv(env_configurations.configurations))

        run_args = {'train': False, 'play': True, 'profile': False, 'checkpoint': ckpt}

        self.runner = rl_games.torch_runner.Runner()
        self.runner.load(config)
        self.player = self.runner.create_player()
        self.player.has_batch_dimension = True
        rl_games.torch_runner._restore(self.player, run_args)
        rl_games.torch_runner._override_sigma(self.player, run_args)
        print("RL Games torch runner created.")

        self.wrapper_model = ONNX_RLGames_Wrapper(self.player.model, True, self.player.has_batch_dimension, self.player.clip_actions, self.player.actions_low, self.player.actions_high)

    def create_my_envs(self):
        return ProxyEnv(self.config)

    def __call__(self, obs):
        return self.wrapper_model(obs)

    @property
    def device(self):
        return self.player.device

    def export_onnx(self, x, name: str):
        # Export to ONNX
        torch.onnx.export(
            self.wrapper_model,                      # model being run
            x,                # model input (or tuple for multiple inputs)
            name,        # where to save the model (path)
            export_params=True,         # store trained parameter weights inside the model file
            opset_version=17,           # ONNX version (17 recommended for 2025+)
            do_constant_folding=True,   # optimize constants
            input_names=['input'],   # 👈 ensures the input node name is 'input'
            output_names=['output'],  # 👈 ensures the output node name is 'output'
            dynamic_axes={              # variable-length axes
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
                },
            # verbose=True,
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("yml", nargs="?", default=None, help="the name of the yml config file")
    parser.add_argument("ckpt", nargs="?", default=None, help="ckeckpoint path")
    parser.add_argument("use_control_policy", nargs="?", default=None, type=int, help="use_control_policy")
    args = parser.parse_args()

    use_control_policy = args.use_control_policy
    onnx_name = "duck_ctrl.onnx" if use_control_policy else "duck_gait.onnx"

    with open(os.path.join(get_working_directory(), 'envs/rl_games_config/', args.yml), 'r') as stream:
        config = yaml.safe_load(stream)

    agent = RLGamesInference(config, args.ckpt)

    # TODO: generalize, it's only the duck env atm
    from envs.duck_backlash import DuckEnv
    envs = DuckEnv(num_envs=6, max_episode_length=10000, apply_external_force=False, env_offset_mode="line_z", use_control_policy=use_control_policy)
    obs, info = envs.reset()

    if not use_control_policy:
        envs.vel_xy_cmd[:, 0] = torch.tensor([0.0] * 3 + [-0.15] * 3)
        envs.yaw_rate_cmd[:, 0] = torch.tensor([-0.3, 0.0, 0.3] * 2)

    def get_obs(obs):
        if isinstance(obs, dict):
            return obs["obs"]
        elif isinstance(obs, torch.Tensor):
            return obs
        else:
            raise NotImplementedError

    # Export
    if True:
        agent.export_onnx(get_obs(obs)[0:1, :], onnx_name)

    # Rendering
    render = envs.gym.get_render()
    if render is not None:
        render.capped_step = True

    # Solver loop
    finished = False
    idx = 0

    t_sim = 0.0
    t_end = 5.0

    obs_all = get_obs(obs).clone().cpu()
    t_rec = [0.0]
    recording_done = False

    # from apps.duck_sim2real_backlash import LowPassActionFilter
    # action_filter = LowPassActionFilter(control_freq=50.0, cutoff_frequency=30.0)

    while not finished:
        # Step
        obs_d = get_obs(obs).to(agent.device)

        ts = time.perf_counter()
        action = agent(obs_d)
        print("inference time: ", time.perf_counter() - ts)

        action = action.to(envs.device)

        # action *= 0.8

        # targets = action + envs.home_pos_active
        # action_filter.push(targets)
        # targets = action_filter.get_filtered_action()
        # action = targets - envs.home_pos_active

        obs, rew, reset, timeout, info = envs.step(action)
        idx += 1
        t_sim += envs.dt

        # print("yaw_rate_cmd", envs.yaw_rate_cmd[0])
        # print("vel_xy_cmd", envs.vel_xy_cmd[0, 0])

        # if not recording_done and envs.total_num_envs == 1:
        #     obs_all = torch.cat((obs_all, obs["obs"].clone().cpu()), dim=0)
        #     t_rec.append(t_sim)

        #     if t_sim > t_end:
        #         import sys
        #         sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../deploy"))
        #         from duck_sim2real import save_observations
        #         # obs_all[:, 8:18] += envs.home_pos_active.view(1, -1).cpu().view(1, -1)
        #         save_observations(obs_all, t_rec)
        #         recording_done = True

        if idx >= envs.max_episode_length:
            render.set_paused(True)
            idx = 0

        finished = envs.render_finished

import torch
import numpy as np
import os

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

## Process command-line arguments ##
from argparse import ArgumentParser
from torch.profiler import profile, record_function, ProfilerActivity

parser = ArgumentParser()

parser.add_argument("environment", nargs="?",
                    choices=[
                        "allegro_custom",
                        "ant_custom",
                        "ant_determinism",
                        "ant_domain_randomization",
                        "ant_rigid_material_domain_randomization",
                        "ant_isaac_gym",
                        "ant_record_mask",
                        "cartpole_custom",
                        "cartpole_vision",
                        "cartpole_classifier_reward",
                        "franka_box_ik",
                        "h1_custom",
                        "h1_humanoid_gym",
                        "h1_isaac_gym",
                        "h1_vsim",
                        "duck_ppo",
                        "duck_ppo_control",
                        "humanoid_custom",
                        "humanoid_isaac_gym",
                        "minimal",
                        "piper_reach_space",
                        "trifinger_custom",
                        ],
                    default="ant_custom", help="environment to be trained")

parser.add_argument(
    "mode",
    nargs="?",
    choices=[
        "train",
        "play",
        "profile"],
    default="train",
    help="train network, play from a given checkpoint, or profile training")

parser.add_argument("checkpoint", nargs="?",
                    default=None, help="file containing saved state of policy network")

parser.add_argument("--seed", type=int,
                    help="seed for pytorch, numpy, etc")

parser.add_argument("--headless", choices=["True", "False"], default=None,
                    help="run without rendering")

parser.add_argument("--num_envs", type=int,
                    help="number of environments")

parser.add_argument("--max_epochs", type=int,
                    help="maximum number of epochs")

parser.add_argument("--horizon_length", type=int,
                    help="number of time steps per epoch")

parser.add_argument("--games_num", type=int,
                    help="number of games in play mode")

parser.add_argument("--learning_rate", type=float,
                    help="learning rate of RL algorithm")

parser.add_argument("--kl_threshold", type=float,
                    help="KL threshold of adaptive learning rate schedule")

parser.add_argument("--max_contact_pairs_per_env", type=int,
                    help="maximum number of contact pairs per environment")

parser.add_argument("--env_args", help="arguments passed to environment constructor in format"
                    " KEY1=VAL1,KEY2=VAL2,...")

parser.add_argument(
    "--record",
    type=str,
    nargs='?',
    const="output.vsim_anim",
    help="""record the running environments. If no file path is specified, records into
output.vsim_anim. In order to save only selected environments, accept the `record_mask_buf:
torch.Tensor` argument in the initializer of your environment class.  Only environments set to True
will be saved.""")

parser.add_argument("--record_max_frames", type=int, default=64,
                    help="Maximum number of frames to record (default 64)")

parser.add_argument("--record_max_file_size", type=int, default=1024 * 1024 * 1024,
                    help="Maximum size of output file for saving record result (default 1GiB)")

parser.add_argument("--experiment_name", type=str,
                    help="name of directory used to store results")

args = parser.parse_args()

env = args.environment
mode = args.mode
checkpoint = args.checkpoint

## Implement rl_games vectorized environment interface ##
from rl_games.common.ivecenv import IVecEnv
from vlearn.spaces import Box, Discrete
import gym.spaces

# Process arguments
class_kwargs = {}

if env == "allegro_custom":
    from envs.allegro_environment_gpu import AllegroEnvironment as EnvClass
    env_name = "allegro-env"
    yml_file = "allegro_ppo.yml"

elif env == "ant_custom":
    from envs.ant_environment_gpu import AntEnvironmentGpu as EnvClass
    env_name = "ant-env"
    yml_file = "ant_ppo.yml"

elif env == "ant_isaac_gym":
    from envs.ant_environment_isaac_gym import AntEnvironmentIsaacGym as EnvClass
    env_name = "ant-env"
    yml_file = "ant_ppo.yml"

elif env == "ant_domain_randomization":
    from envs.ant_environment_domain_randomization import AntEnvironmentDomainRandomization as \
        EnvClass
    env_name = "ant-env"
    yml_file = "ant_ppo.yml"

elif env == "ant_rigid_material_domain_randomization":
    from envs.ant_environment_rigid_material_domain_randomization import AntEnvironmentRigidMaterialDomainRandomization as \
        EnvClass
    env_name = "ant-env"
    yml_file = "ant_ppo.yml"

elif env == "ant_record_mask":
    from envs.ant_environment_record_mask import AntEnvironmentRecordMask as EnvClass
    env_name = "ant-env"
    yml_file = "ant_ppo.yml"

elif env == "ant_determinism":
    from envs.ant_environment_determinism import AntEnvironmentDeterminism as EnvClass
    env_name = "ant-env"
    yml_file = "ant_ppo.yml"

elif env == "cartpole_custom":
    from envs.cartpole_environment_gpu import CartpoleEnvironmentGpu as EnvClass
    env_name = "cartpole-env"
    yml_file = "cartpole_ppo.yml"

elif env == "cartpole_vision":
    from envs.cartpole_environment_vision import CartpoleEnvironmentVision as EnvClass
    env_name = "cartpole-env"
    yml_file = "cartpole_vision_ppo.yml"

elif env == "cartpole_classifier_reward":
    from envs.cartpole_environment_gpu_classifier import CartpoleVisionClassifier as EnvClass
    env_name = "cartpole-env"
    yml_file = "cartpole_vision_classifier_ppo.yml"

elif env == "minimal":
    from envs.minimal_environment import MinimalEnvironment as EnvClass
    env_name = "minimal-env"
    yml_file = "minimal_ppo.yml"

elif env == "franka_box_ik":
    from python.train.envs.franka_box_pickup import FrankaBoxPickup as EnvClass
    env_name = "franka_box-env"
    yml_file = "franka_box_ik_ppo.yml"

elif env == "humanoid_custom":
    from envs.humanoid_environment_gpu import HumanoidEnvironmentGpu as EnvClass
    env_name = "humanoid-env"
    yml_file = "humanoid_ppo.yml"

elif env == "humanoid_isaac_gym":
    from envs.humanoid_environment_isaac_gym import HumanoidEnvironmentIsaacGym as EnvClass
    env_name = "humanoid-env"
    yml_file = "humanoid_ppo.yml"

elif env == "h1_custom":
    from envs.h1_environment_gpu import H1EnvironmentGpu as EnvClass
    env_name = "h1-env"
    yml_file = "h1_ppo.yml"

elif env == "h1_isaac_gym":
    from envs.h1_environment_isaac_gym import H1EnvironmentIsaacGym as EnvClass
    env_name = "h1-env"
    yml_file = "h1_ppo.yml"

elif env == "h1_humanoid_gym":
    from envs.h1_environment_humanoid_gym import H1EnvironmentHumanoidGym as EnvClass
    env_name = "h1-env"
    yml_file = "h1_humanoid_gym_ppo.yml"

elif env == "h1_vsim":
    from envs.h1_environment_vsim import H1LocomotionEnv as EnvClass
    env_name = "h1-env"
    yml_file = "h1_ppo_vsim.yml"

elif env == "duck_ppo":
    from envs.duck_backlash import DuckEnv as EnvClass
    env_name = "duck_ppo"
    yml_file = "duck_ppo.yml"

elif env == "duck_ppo_control":
    from envs.duck_backlash import DuckEnv as EnvClass
    env_name = "duck_ppo_control"
    yml_file = "duck_ppo_control.yml"

    class_kwargs["use_control_policy"] = True
    class_kwargs["max_episode_length"] = 800
    class_kwargs["apply_external_force"] = False


elif env == "piper_reach_space":
    from envs.piper_box_pickup import PiperBoxPickup as EnvClass
    env_name = "piper-reach-space"
    yml_file = "piper_reach_space.yml"

    class_kwargs["task"] = "reach-space"
    class_kwargs["vision"] = False
    class_kwargs["rgb_cameras"] = []
    class_kwargs["depth_cameras"] = []
    class_kwargs["object"] = "Box"
    class_kwargs["piper_venv"] = "assets/donotship/piper/piper.venv"
    class_kwargs["gripper_limit"] = 0.035
    class_kwargs["initial_target_pose"] = [0, 0, 0, 1, 0.4, 0, 0.82]
    class_kwargs["max_episode_length"] = 180
    class_kwargs["frame_skip"] = 1

elif env == "trifinger_custom":
    from envs.trifinger_environment_gpu import TrifingerEnvironmentGpu as EnvClass
    env_name = "trifinger-env"
    yml_file = "trifinger_ppo.yml"

else:
    Exception("This should not be reached")

## Read config file ##
import yaml
from vlearn.gym import get_working_directory
with open(os.path.join(get_working_directory(), 'envs/rl_games_config/', yml_file), 'r') as stream:
    config = yaml.safe_load(stream)

if 'player' not in config['params']['config']:
    config['params']['config']['player'] = {}

# Use vectorized environments in play mode
config['params']['config']['player']['use_vecenv'] = True

if args.seed is not None:
    config['params']['seed'] = args.seed

    # TODO: remove when Vsim's nondeterminism has been resolved
    # The following should not be necessary but I'm putting it here to guarantee torch determinism
    # as much as possible
    torch.cuda.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    # Set the number of threads for intra-op parallelism
    torch.set_num_threads(1)

    # Set the number of threads for inter-op parallelism
    torch.set_num_interop_threads(1)

    torch.cuda.set_device(0)  # Select the GPU you want to use
    torch.cuda.set_per_process_memory_fraction(1.0)  # Adjust memory allocation (if needed)

if args.num_envs is not None:
    config['params']['config']['num_actors'] = args.num_envs

if args.max_epochs is not None:
    config['params']['config']['max_epochs'] = args.max_epochs

if args.horizon_length is not None:
    config['params']['config']['horizon_length'] = args.horizon_length

if args.games_num is not None:
    config['params']['config']['player']['games_num'] = args.games_num

if args.learning_rate is not None:
    config['params']['config']['learning_rate'] = args.learning_rate

if args.kl_threshold is not None:
    config['params']['config']['kl_threshold'] = args.kl_threshold

if args.experiment_name is not None:
    config['params']['config']['full_experiment_name'] = args.experiment_name

# Adjust minibatch sizes

num_envs = config['params']['config']['num_actors']
horizon_len = config['params']['config']['horizon_length']


def adjust_minibatch_size(config_dict, num_envs, horizon_len):
    mb_size = config_dict['minibatch_size']

    batch_size = horizon_len * num_envs
    num_batches = (batch_size + mb_size - 1) // mb_size
    if num_batches > 1:
        mb_size = batch_size // num_batches
    else:
        mb_size = horizon_len * num_envs

    if (batch_size % mb_size) != 0:
        print("Error: batch size ({}) is not divisible by minibatch size ({})".format(batch_size,
                                                                                      mb_size))
        print("Batch size = horizon length ({}) x number of environments ({})".format(horizon_len,
                                                                                      num_envs))
        exit(1)

    config_dict['minibatch_size'] = mb_size


adjust_minibatch_size(config['params']['config'], num_envs, horizon_len)

if 'central_value_config' in config['params']['config'].keys():
    adjust_minibatch_size(config['params']['config']['central_value_config'], num_envs, horizon_len)

# Process remaining environment args
if args.headless is None:
    if mode == "train":
        args.headless = "True"
    elif mode == "play":
        args.headless = "False"
    elif mode == "profile":
        args.headless = "True"
    else:
        Exception("This should not be reached")

from vlearn.utils.str_to_env_args import str_to_env_args, str_to_bool
from typing import get_type_hints

class_kwargs['rendering'] = not str_to_bool(args.headless)
class_kwargs['send_interrupt'] = class_kwargs['rendering']

if args.max_contact_pairs_per_env is not None:

    class_kwargs['max_contact_pairs_per_env'] = args.max_contact_pairs_per_env

if args.env_args is not None:

    d = str_to_env_args(EnvClass, args.env_args)
    class_kwargs.update(d)

if args.record is not None:

    if "record_mask_buf" in get_type_hints(EnvClass.__init__):
        record_mask_buf = torch.full((num_envs,), True, device=torch.device("cuda:0"),
                                     dtype=torch.bool)
        class_kwargs['record_mask_buf'] = record_mask_buf
    else:
        print(">>> Note: {}.__init__() does not accept 'record_mask_buf: torch.Tensor' keyword "
              "argument, so all environments will be recorded.".format(EnvClass.__name__))
        record_mask_buf = []


def convert_space(space):

    if isinstance(space, Box):
        return gym.spaces.Box(low=space.low, high=space.high,
                              shape=space.shape)  # , dtype=space.dtype)

    if isinstance(space, Discrete):
        return gym.spaces.Discrete(n=space.n)

    Exception("This should not be reached")


from vlearn import get_gym


class VlearnEnv(IVecEnv):

    def __init__(self, config_dict, config_name, num_actors, **kwargs):
        self.envs = config_dict[config_name]['env_creator'](num_actors, **kwargs)

        self.num_actors = num_actors
        if args.record is not None:
            get_gym().start_recording(max_num_frames=args.record_max_frames)

    def step(self, actions):
        return self.envs.step(actions)

    def reset(self):
        return self.envs.reset()

    def get_env_info(self):

        env_info = {}
        env_info["observation_space"] = convert_space(self.envs.single_observation_space)
        env_info["action_space"] = convert_space(self.envs.single_action_space)

        if hasattr(self.envs, "single_state_space"):
            env_info["state_space"] = convert_space(self.envs.single_state_space)

        return env_info


## Define environment builder ##
from vlearn.torch_utils.wrappers import NewToOldAPICompatilibity

multi_gpu = config['params']['config'].get('multi_gpu', False)


def create_my_envs(num_envs, **kwargs):

    assert torch.cuda.is_available()

    # This is the same logic that RL Games uses to set the cuda device
    if multi_gpu:
        cuda_device = int(os.getenv("LOCAL_RANK", "0"))
    else:
        cuda_device = 0

    device = torch.device("cuda:{}".format(cuda_device))

    # For multiple environment sets
    if "num_envs" in class_kwargs:
        assert isinstance(class_kwargs["num_envs"], list)
        # If only one number is given, the user is specifying the size of a single environment set,
        # so we need to do some extra work to fit the overall num_envs
        if len(class_kwargs["num_envs"]) == 1:
            num_env_sets = num_envs // class_kwargs["num_envs"][0]
            class_kwargs["num_envs"] *= num_env_sets
            remainder = num_envs - num_env_sets * class_kwargs["num_envs"][0]
            if remainder > 0:
                class_kwargs["num_envs"].append(remainder)
        assert sum(class_kwargs["num_envs"]) == num_envs, \
            "sum(class_kwargs['num_envs']) = {} should be equal to num_envs = {}".format(
            sum(class_kwargs['num_envs']), num_envs)
        num_envs = class_kwargs.pop("num_envs")

    envs = EnvClass(num_envs, device, **class_kwargs)

    return NewToOldAPICompatilibity(envs)


## Register environment type and ant environment ##
from rl_games.common import env_configurations, vecenv

env_configurations.register(env_name, {
    "vecenv_type": "VLEARN",
    "env_creator": lambda num_envs, **kwargs: create_my_envs(num_envs, **kwargs)})

vecenv.register("VLEARN",
                lambda config_name, num_actors, **kwargs:
                VlearnEnv(env_configurations.configurations, config_name, num_actors,
                          **kwargs))

## Prepare run arguments ##
if mode == "train" or mode == "test":
    run_args = {'train': True, 'play': False, 'profile': False}
elif mode == "play":
    run_args = {'train': False, 'play': True, 'profile': False}
elif mode == "profile":
    run_args = {'train': True, 'play': False, 'profile': True}

if checkpoint:
    run_args['checkpoint'] = checkpoint

## Configure and run runner ##
from rl_games.torch_runner import Runner

try:
    if run_args['profile']:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            runner = Runner()

            runner.load(config)
            runner.run(run_args)
        prof.export_chrome_trace("trace.json")
    else:
        runner = Runner()

        runner.load(config)
        runner.run(run_args)
except (KeyboardInterrupt, Exception) as e:
    if args.record is not None:
        get_gym().end_recording(args.record, mask=list(record_mask_buf))
    raise e

if args.record is not None:
    get_gym().end_recording(args.record, mask=list(record_mask_buf),
                            max_file_size=args.record_max_file_size)

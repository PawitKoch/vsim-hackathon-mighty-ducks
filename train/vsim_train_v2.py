import hydra
from hydra.utils import to_absolute_path
import os
from pathlib import Path
import shutil
os.environ["HYDRA_FULL_ERROR"] = "1"

from argparse import ArgumentParser

if __name__ == "__main__":
    # Step 1: parse
    parser = ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="The yaml file containing training config without .yaml")

    args, unknown = parser.parse_known_args()

    # Step 2: initialize Hydra
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reset any previous Hydra instances
    hydra.initialize(config_path="./train_config", version_base=None)
    cfg = hydra.compose(config_name=args.config, overrides=unknown)

    # Step 3: training
    agent = hydra.utils.instantiate(cfg.agent, _recursive_=True)

    cache_dir = Path(to_absolute_path(f"sam2_cache_"))
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    if agent.mode == "train":
        agent.train()

    if agent.mode == "play":
        agent.infer()

    # debug infer_io if needed
    if False and agent.mode == "play":
        envs = agent.envs

        obs, info = envs.reset()

        # Rendering
        render = envs.gym.get_render()
        if render is not None:
            render.capped_step = True

        # Solver loop
        finished = False
        idx = 0

        while not finished:
            imgs = []
            if "image" in info:
                imgs = info["images"]

            action = agent.infer_io(obs, imgs)

            # Step
            obs, rew, reset, timeout, info = envs.step(action)

            idx += 1

            if idx >= envs.max_episode_length:
                render.set_paused(True)
                idx = 0

            finished = envs.render_finished

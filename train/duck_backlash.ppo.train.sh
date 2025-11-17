#!/bin/sh

# python rl_games_train.py duck_ppo train policies/duck_v10/duck_ppo.pth --headless=False
python rl_games_train.py duck_ppo train runs/duck_ppo_15-13-26-22/nn/duck_ppo.pth --headless=False

#!/bin/bash
set -euo pipefail

# Minimal Python script to activate vlearn
python -c 'import vlearn; vlearn.create_gym(with_render=False)'

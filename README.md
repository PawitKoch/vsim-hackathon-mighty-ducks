# Vlearn

## Prerequisites

__Ubuntu 22.04 / 24.04__
- [conda](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)
- CUDA Drivers - run `nvidia-smi` and ensure your `CUDA Version:` is >= 12.9. If this value is less than 12.9, you will need to udpate your drivers using the command `sudo apt install nvidia-driver-580-open`
- CUDA Toolkit 12.9 - we have already downloaded the cuda toolkit 12.9 script you can run here with `sudo sh cuda_12.9.0_575.51.03_linux.run`. This script takes a while to run and does not provide any updates while it is running so please be patient. During the installation, you may be prompted by several messages, here is how you can respond to each:
    - Existing package manager found - select `Continue`
    - End user license agreement - `accept`
    - CUDA installer:
      - Ensure Driver is **unchecked**
      - Leave everything else as is and select `Install`
    - Update symlink - select `yes`
    - The script will continue to run in the background, after a while you will get a summary to signify this step has finished.
- After these, you will need to add the following environment variables to your `~/.bashrc` and restart your terminal for changes to take effect (and these will need to replace any previous cude version environment variables you have)
```
export PATH="$PATH:/usr/local/cuda-12.9/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-12.9/lib64"
```
- [CMake 3.24](https://cmake.org/download/) or later.  On Ubuntu 22.04, it is necessary to install
  CMake using
  Snap because `apt-get` installs CMake 3.22: `sudo snap install cmake
  --classic`
- SDL2-dev: `sudo apt install libsdl2-dev`
- Assimp: `sudo apt install libassimp-dev`
- GNU C++ Compiler: `sudo apt install g++`
- Mesa OpenGL utility library: `sudo apt install libglu1-mesa-dev`
- ZeroMQ library: `sudo apt install libzmq3-dev`
- UV: `sudo snap install astral-uv --classic`

## Installation on Ubuntu 22.04 / 24.04

First, create a new conda environment with Python version 3.10:

```
conda create --name vlearn python=3.10
conda activate vlearn
```

In these instructions, replace `python=3.10` with `python=3.11` if you are using Python 3.11.

Next, install the remaining dependencies:

```
pip install matplotlib pyyaml tensorboard rl-games h5py hydra-core onnx onnxruntime onnxruntime-gpu
```

We have also included pytorch wheel files:
```
pip install torch-wheel-files/*.whl
```

Finally, run `pip install` on the Linux wheel file:

```
pip install vlearn-0.2.7-cp310-cp310-linux_x86_64.whl
```

## Activation

In order to activate Vlearn, please make sure that 1) you are connected to the internet and 2) your
license key is saved in a file named `License.key` in the root directory. Then, run the command
`bash activate_vlearn.sh` (Linux) or `cmd /c activate_vlearn.bat` (Windows) from the root directory.
This will start a node-locked activation registered to your machine.

Once Vlearn is installed and activated, you can starting training!

## Documentation

Please unzip `docs/vlearn-docs.zip` and open `index.html` with a browser to
view Vlearn's documentation.

## Python stubs for Visual Studio Code auto-complete

In Visual Studio Code, add `python.analysis.extraPaths": ["./docs/stubs"]` to `settings.json` in
order for auto-complete to work correctly.

## Troubleshooting

### Cannot find `libTurboActivate.so`

If you get the following error,

```
OSError: libTurboActivate.so: cannot open shared object file: No such file or directory
```

then search for the Vlearn installation and add its `lib` subdirectory to
`LD_LIBRARY_PATH` as follows:

```
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/path/to/vlearn/installation/lib"
```

You can find the location of the Vlearn installation using the command

```
pip show vlearn | grep Location
```

For example, if this command returns `Location:
/home/user/anaconda3/envs/vlearn/lib/python3.10/site-packages`, then the
command should be

```
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/home/user/anaconda3/envs/vlearn/lib/python3.10/site-packages/vlearn/lib"
```

### CUDA or libGL error

If Vlearn crashes with the error message `Assert failed err == cudaSuccess`, a `torch.AcceleratorError: CUDA error: unknown error` or an `libGL` error
and you are using a machine with both an integrated GPU *and* a discrete NVIDIA
GPU, this is likely due to an issue with NVIDIA Optimus.

In order to resolve this issue, set the following environment variables:

```
export CUDA_VISIBLE_DEVICES=0
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
```

and run Vlearn again.  You can put these export statements in a `.bashrc` file to make them
persistent.  Alternatively, you can set these environment variables specifically in the scope of
your conda environment as follows:

```
conda activate vlearn
conda env config vars set CUDA_VISIBLE_DEVICES=0
conda env config vars set __NV_PRIME_RENDER_OFFLOAD=1
conda env config vars set __GLX_VENDOR_LIBRARY_NAME=nvidia
```

If this doesn't work, follow the instructions here to configure your machine to use NVIDIA graphics
only (reboot required):
[https://wiki.archlinux.org/title/NVIDIA_Optimus#Use_NVIDIA_graphics_only](https://wiki.archlinux.org/title/NVIDIA_Optimus#Use_NVIDIA_graphics_only).
Then try running Vlearn again.

### New Vlearn release produces error

If you update to a new release of Vlearn and it is producing errors, this may be due to your conda
environment continuing to use the old version of Vlearn.  In order to resolve this issue, please run
`pip uninstall vlearn` to uninstall the old version of Vlearn.

### License activation fails on virtual machine

It is currently not possible to run Vlearn in a virtual machine.  If you are using Windows 11, you
may run into this error because of Windows Hypervisor.  Please refer to
<https://wyday.com/limelm/help/faq/#in-vm> for troubleshooting.  If this does not resolve your
issue, please get in touch with us.

### C++ library mismatch

If you get the error `version `GLIBCXX_3.4.32' not found`, then there may be a
mismatch between the C++ library provided by your conda environment, and the
C++ library expected by `libassimp`.

In order to resolve this issue, set the following environment variable: `export
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`.

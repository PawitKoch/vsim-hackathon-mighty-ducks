# OpenDuckMini-deployment

## Hackathon
1. Turn on the duck:
- switch is on its back - down is on, up is off
- Wait ~10-20s for the duck to turn on

2. Connect to the duck's wifi hotspot:
- Find the duck's id on the back of its head
- Find the corresponding duck wifi and connect
- password: `ARi4-duck`

3. Open a terminal (No. 1):
- ssh into the duck's pi `ssh duck@10.42.0.1`
- password: `duck`
- if you get a warning "REMOTE HOST IDENTIFICATION CHANGED ...", run the command that it's suggesting

4. (Optional) Open a second terminal (No. 2):
- If you don't want to change the deployment script on the duck, skip this step.
- If you made any changes in the `vsim_edge_deploy.py`, you need to copy it over to the duck
- `rsync -avz --exclude='.git' /path/to/OpenDuckMini-deployment duck@10.42.0.1:~`
- password: `duck`

5. In terminal No. 1:
- Activate env:`workon open-duck-mini-runtime`
- `cd ~/OpenDuckMini-deployment/scripts/`
- run script with`python vsim_edge_deploy.py --duck_id=N` where `N = the duck's id`
- if you get an IO error/USB port not found or similar, power cycle the duck

## Installation - not needed for Hackathon
Based on https://github.com/apirrone/Open_Duck_Mini_Runtime
- you need vsim_endpoint to control the duck from a computer (not needed if you run the policy on the edge)
- run install.sh
- add this to the end of the .bashrc:
    ```
    export WORKON_HOME=$HOME/.virtualenvs
    export PROJECT_HOME=$HOME/Devel
    source /usr/share/virtualenvwrapper/virtualenvwrapper.sh
    ```
- Enable I2C: `sudo raspi-config` -> `Interface Options` -> `I2C`
- Set the usbserial latency timer
    ```
    cd  /etc/udev/rules.d/
    sudo touch 99-usb-serial.rules
    sudo nano 99-usb-serial.rules
    ```
 - copy the following line in the file
    `SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"`
- test main script: `cd scripts` and run `python vsim_edge_deploy.py`

rsync basics:
- rsync -avz duck@10.42.0.1:/remote/path /local/path
- rsync -avz --exclude='.git' /local/path duck@10.42.0.1:/remote/path

rsync transfer observation file:
- rsync -avz duck@10.42.0.1:~/OpenDuckMini-deployment/scripts/obs.json ~/transfer
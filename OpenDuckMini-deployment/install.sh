sudo apt update
sudo apt upgrade
sudo apt install git
sudo apt install python3-pip
sudo apt install python3-virtualenvwrapper

mkvirtualenv -p python3 open-duck-mini-runtime
workon open-duck-mini-runtime

pip install zmq rustypot==0.1.0 adafruit-circuitpython-bno055==5.4.13 adafruit-blinka RPi.GPIO
pip install git+https://github.com/pollen-robotics/pypot@support-feetech-sts3215

pip install onnxruntime
# Mastermind

Mastermind is an offensive security assessment agent.

## News
- 2025-12-23: Working on the Agentic RL framework 🚧.
- 2025-12-21: Mastermind Agent is released 🎉.

## Docker Setup

```bash
docker run -ti --privileged --network host --name mastermind-kali kalilinux/kali-rolling
```

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install kali-linux-headless
apt update && apt -y install kali-linux-headless

# Install OpenVPN
apt install openvpn
sudo openvpn --config ./labs_Elfsong.ovpn --daemon
ip addr show tun0
sudo killall openvpn

# Clone mastermind Repository
cd /home
git clone https://github.com/Elfsong/mastermind.git

# Create virtual environment named mastermind
cd mastermind
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# Run mastermind
cd src
python -m mastermind
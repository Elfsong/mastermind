# Mastermind

Mastermind is an offensive security assessment agent.

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
uv venv .mastermind
source .mastermind/bin/activate

# Set the environment variables
echo "OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>" >> .env
echo "TAVILY_API_KEY=<YOUR_TAVILY_API_KEY>" >> .env
echo "LANGSMITH_TRACING=true" >> .env
echo "LANGSMITH_ENDPOINT=https://api.smith.langchain.com" >> .env
echo "LANGSMITH_API_KEY=<YOUR_LANGSMITH_API_KEY>" >> .env
echo "LANGSMITH_PROJECT=<YOUR_LANGSMITH_PROJECT>" >> .env
echo "TARGET_IP=<YOUR_TARGET_IP>" >> .env
```
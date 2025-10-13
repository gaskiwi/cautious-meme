# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- (Optional) CUDA-capable GPU for faster training
- (Optional) Docker for containerized training
- (Optional) AWS account for cloud training

## Installation Methods

### Method 1: Virtual Environment (Recommended for Local Development)

```bash
# Clone the repository
git clone https://github.com/gaskiwi/cautious-meme.git
cd cautious-meme

# Create and activate virtual environment
python -m venv venv

# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import src; print('Installation successful!')"
```

### Method 2: Using Make

```bash
# Clone repository
git clone https://github.com/gaskiwi/cautious-meme.git
cd cautious-meme

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install using Make
make install

# Test installation
make test
```

### Method 3: Development Installation

```bash
# Clone repository
git clone https://github.com/gaskiwi/cautious-meme.git
cd cautious-meme

# Install in editable mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Method 4: Docker (No Python Setup Required)

```bash
# Clone repository
git clone https://github.com/gaskiwi/cautious-meme.git
cd cautious-meme

# Build Docker image
docker build -t rl-robotics:latest .

# Run training
docker-compose up training
```

## System-Specific Dependencies

### Ubuntu/Debian

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git
```

### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.10
```

### Windows

1. Install Python 3.8+ from [python.org](https://www.python.org/)
2. Install Visual C++ Build Tools from [Microsoft](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## GPU Support

### NVIDIA CUDA (for GPU acceleration)

#### Linux

```bash
# Install NVIDIA drivers
sudo apt-get install nvidia-driver-525

# Install CUDA Toolkit (if not using Docker)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

#### Windows

1. Install NVIDIA drivers from [NVIDIA website](https://www.nvidia.com/download/index.aspx)
2. Install CUDA Toolkit 11.8
3. Install PyTorch with CUDA:
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Docker with GPU Support

```bash
# Install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/yum.repos.d/nvidia-docker.repo

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test
docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Verification

### Quick Test

```bash
# Test configuration loading
python -c "from src.utils.config_loader import load_config; config = load_config('configs/training_config.yaml'); print('Config OK')"

# Test environment
python -c "from src.environments import SimpleRobotEnv; env = SimpleRobotEnv(); env.reset(); print('Environment OK')"

# Run comprehensive tests
python examples/quick_test.py
```

### Test Training (Quick Run)

```bash
# Train for 1000 steps to verify everything works
python train.py --timesteps 1000
```

## AWS Setup (Optional)

### AWS CLI Installation

```bash
# Linux/Mac
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure
aws configure
```

### AWS Credentials

```bash
# Set up credentials
aws configure
# Enter:
#   AWS Access Key ID
#   AWS Secret Access Key
#   Default region (e.g., us-east-1)
#   Default output format (json)
```

## Common Issues

### PyBullet Installation Issues

**Error:** `No module named 'pybullet'`

**Solution:**
```bash
pip install pybullet --upgrade
```

### OpenGL Issues (Linux)

**Error:** `GLIBCXX_3.4.XX not found`

**Solution:**
```bash
sudo apt-get install libgl1-mesa-glx
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Make sure you're in the project root
export PYTHONPATH="${PYTHONPATH}:/path/to/cautious-meme"

# Or install in development mode
pip install -e .
```

### CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
- Reduce batch size in `configs/training_config.yaml`
- Use smaller network architecture
- Close other GPU-using applications

## Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# AWS
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=your-account-id

# Training
TRAINING_CONFIG=configs/training_config.yaml

# Optional: Weights & Biases
WANDB_API_KEY=your-key
```

## Upgrading

### Update Dependencies

```bash
# Pull latest code
git pull

# Update dependencies
pip install -r requirements.txt --upgrade

# Or with Make
make install
```

### Update Docker Image

```bash
# Rebuild image
docker build -t rl-robotics:latest .

# Or with Make
make docker-build
```

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv/

# Remove Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Remove generated files
rm -rf models/ logs/ runs/ checkpoints/
```

## Getting Help

If you encounter issues:

1. Check this installation guide
2. Review [README.md](README.md) for usage instructions
3. Check [QUICKSTART.md](QUICKSTART.md) for common workflows
4. Look for similar issues in the repository
5. Verify all system dependencies are installed

## Next Steps

After installation:

1. Run quick test: `python examples/quick_test.py`
2. Try training: `python train.py --timesteps 10000`
3. Monitor with TensorBoard: `tensorboard --logdir runs/`
4. Read [QUICKSTART.md](QUICKSTART.md) for next steps

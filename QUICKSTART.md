# Quick Start Guide

Get started with RL Robotics training in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip
- (Optional) Docker for containerized training
- (Optional) AWS account for cloud training

## Installation

### Option 1: Virtual Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/gaskiwi/cautious-meme.git
cd cautious-meme

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using Make

```bash
make install
```

### Option 3: Development Install

```bash
pip install -e .
```

## Running Your First Training

### 1. Train Locally (CPU)

```bash
python train.py
```

This will:
- Create a simple cart-pole robot environment
- Train a PPO agent for 1M timesteps
- Save checkpoints every 50k steps
- Save the best model to `models/best_model`
- Log training progress to `logs/` and TensorBoard in `runs/`

### 2. Monitor Training

Open TensorBoard in another terminal:

```bash
tensorboard --logdir runs/
```

Visit http://localhost:6006 to see training metrics in real-time.

### 3. Evaluate Trained Model

```bash
# Evaluate best model
python evaluate.py models/best_model --episodes 10

# Evaluate with visualization
python evaluate.py models/best_model --render --episodes 5
```

## Customization

### Change Algorithm

Edit `configs/training_config.yaml`:

```yaml
agent:
  algorithm: "SAC"  # Try PPO, SAC, TD3, or A2C
```

### Adjust Training Duration

```bash
python train.py --timesteps 500000  # Train for 500k steps
```

### Use Different Config

```bash
python train.py --config my_custom_config.yaml
```

### Resume Training

```bash
python train.py --checkpoint models/checkpoint_interrupted
```

## Docker Training

### Build Image

```bash
docker build -t rl-robotics:latest .
```

### Run Training

```bash
docker-compose up training
```

### Run TensorBoard

```bash
docker-compose up tensorboard
```

## AWS Cloud Training

### 1. Set Up AWS

```bash
# Configure AWS CLI
aws configure

# Set environment variables
export VPC_ID="vpc-xxxxx"
export SUBNET_IDS="subnet-xxxxx,subnet-yyyyy"
export KEY_NAME="your-ec2-key"
```

### 2. Deploy Infrastructure

```bash
cd aws
./deploy.sh \
  --vpc-id $VPC_ID \
  --subnet-ids $SUBNET_IDS \
  --key-name $KEY_NAME \
  --fleet-size 4
```

### 3. Build and Push Docker Image

```bash
./push-image.sh
```

### 4. Monitor Training

Check CloudWatch logs or S3 bucket for outputs.

### 5. Download Trained Models

```bash
# Get S3 bucket name from CloudFormation outputs
aws s3 sync s3://YOUR-BUCKET-NAME/outputs/models/ ./models/
```

## Next Steps

### Customize Your Robot

1. Create a new environment in `src/environments/`:

```python
from src.environments.base_robot_env import BaseRobotEnv

class MyRobotEnv(BaseRobotEnv):
    # Implement required methods
    pass
```

2. Update `src/training/trainer.py` to use your environment

3. Adjust observation/action spaces in your environment

### Tune Hyperparameters

Key parameters to experiment with:

- `learning_rate`: 0.0001 - 0.001
- `batch_size`: 32, 64, 128, 256
- `n_steps`: 1024, 2048, 4096
- Network size: [128, 128], [256, 256], [512, 512]

### Try Different Algorithms

- **PPO**: Good general performance, stable
- **SAC**: Great for continuous control
- **TD3**: Stable for continuous actions
- **A2C**: Faster training, less sample efficient

## Troubleshooting

### Import Errors

```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### PyBullet Issues

```bash
# On Ubuntu/Debian
sudo apt-get install python3-opengl

# Test PyBullet
python -c "import pybullet; print('PyBullet OK')"
```

### CUDA Not Available

```bash
# Check PyTorch CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# For CPU-only training, everything still works!
```

## Common Commands

```bash
# Train
make train

# Evaluate (set MODEL variable)
make evaluate MODEL=models/best_model

# Clean outputs
make clean

# Run tests
make test

# Build Docker
make docker-build

# Format code
make format
```

## Tips

1. **Start Small**: Train for 10k steps first to verify everything works
2. **Use TensorBoard**: Monitor training progress visually
3. **Save Often**: Adjust `save_freq` in config for more checkpoints
4. **Experiment**: Try different algorithms and hyperparameters
5. **Scale Up**: Once local training works, move to AWS for faster training

## Getting Help

- Check [README.md](README.md) for detailed documentation
- Review [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Examine configuration files in `configs/`
- Look at example environments in `src/environments/`

Happy training! 🤖🚀

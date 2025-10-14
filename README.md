# RL Robotics Training

A scalable reinforcement learning framework for robot training using PyBullet simulation and AWS Spot Fleet infrastructure.

## Overview

This project provides a complete RL robotics training environment with:
- **PyBullet-based robot simulation** - Flexible environment framework
- **Stable Baselines3 integration** - State-of-the-art RL algorithms (PPO, SAC, TD3, A2C)
- **AWS Spot Fleet training** - Cost-effective distributed training on GPU instances
- **Docker containerization** - Consistent environments across local and cloud
- **Modular architecture** - Easy to customize for your specific robot and tasks

## Project Structure

```
.
├── src/
│   ├── agents/          # RL agent implementations
│   ├── environments/    # PyBullet environment wrappers
│   ├── training/        # Training and evaluation scripts
│   └── utils/           # Configuration and logging utilities
├── configs/             # Training configuration files
├── aws/                 # AWS infrastructure code
│   ├── cloudformation/  # CloudFormation templates
│   ├── deploy.sh        # Deployment script
│   └── push-image.sh    # Docker image deployment
├── docker/              # Docker configurations
├── train.py             # Main training entry point
├── evaluate.py          # Model evaluation script
└── requirements.txt     # Python dependencies
```

## Quick Start

### Local Setup

1. **Install dependencies:**
   ```bash
   # Create virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install requirements
   pip install -r requirements.txt
   ```

2. **Train locally:**
   ```bash
   # Start training with default config
   python train.py
   
   # Train with custom config
   python train.py --config configs/training_config.yaml
   
   # Resume from checkpoint
   python train.py --checkpoint models/checkpoint_interrupted
   ```

3. **Evaluate trained model:**
   ```bash
   # Evaluate with visualization
   python evaluate.py models/best_model --render
   
   # Evaluate multiple episodes
   python evaluate.py models/best_model --episodes 20
   ```

## Training Sessions

This project uses **progressive training sessions** to train increasingly capable agents. Each session focuses on a specific goal and builds upon previous sessions.

### Training Session 1: Height Maximization ✅

**Goal:** Get one of the robots as high (vertically) as possible.

This first session teaches agents to maximize vertical height. The environment supports multiple robots, and the reward is based on the highest robot's position.

**Quick Test:**
```bash
python3 examples/test_height_env.py
```

**Start Training:**
```bash
python3 train.py
```

The default configuration (`configs/training_config.yaml`) is already set up for Training Session 1.

**Key Features:**
- Multi-robot support (default: 3 robots)
- 6 DOF control per robot (forces and torques)
- Reward based on maximum height achieved
- Progressive learning with height improvement bonuses

For detailed information about all training sessions, see [TRAINING_SESSIONS.md](TRAINING_SESSIONS.md).

### Future Sessions
Additional training sessions will be defined and implemented in succession, each building on the skills learned in previous sessions.

### Docker Setup

1. **Build Docker image:**
   ```bash
   docker build -t rl-robotics:latest .
   ```

2. **Run training in container:**
   ```bash
   docker-compose up training
   ```

3. **Run TensorBoard:**
   ```bash
   docker-compose up tensorboard
   # Access at http://localhost:6006
   ```

## AWS Spot Fleet Training

### Prerequisites

- AWS CLI configured with appropriate credentials
- AWS VPC and subnets set up
- EC2 key pair created

### Deployment Steps

1. **Deploy infrastructure:**
   ```bash
   cd aws
   ./deploy.sh \
     --vpc-id vpc-xxxxx \
     --subnet-ids subnet-xxxxx,subnet-yyyyy \
     --key-name your-key-pair \
     --fleet-size 4 \
     --region us-east-1
   ```

2. **Build and push Docker image:**
   ```bash
   ./push-image.sh --region us-east-1
   ```

3. **Monitor training:**
   - Check CloudWatch logs for training progress
   - Models are automatically saved to S3
   - Instances auto-terminate when training completes

4. **Download trained models:**
   ```bash
   aws s3 sync s3://rl-robotics-models-{account-id}/outputs/models/ ./models/
   ```

### AWS Resources Created

- **S3 Bucket**: Model storage with versioning
- **ECR Repository**: Docker image storage
- **EC2 Spot Fleet**: Cost-effective GPU instances
- **IAM Roles**: Secure access management
- **Security Groups**: Network access control

## Configuration

Edit `configs/training_config.yaml` to customize:

### Environment Settings
```yaml
environment:
  max_episode_steps: 1000
  render_mode: null  # Set to "human" for visualization
```

### Agent Configuration
```yaml
agent:
  algorithm: "PPO"  # Options: PPO, SAC, TD3, A2C
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
```

### Network Architecture
```yaml
network:
  policy_layers: [256, 256]
  value_layers: [256, 256]
  activation: "relu"
```

### Training Parameters
```yaml
training:
  total_timesteps: 1000000
  eval_freq: 10000
  save_freq: 50000
```

### AWS Settings
```yaml
aws:
  region: "us-east-1"
  instance_type: "g4dn.xlarge"
  fleet_size: 4
  s3_bucket: "rl-robotics-models"
```

## Custom Robot Environment

The project includes a simple cart-pole example. To use your own robot:

1. **Create a new environment class** inheriting from `BaseRobotEnv`:
   ```python
   from src.environments.base_robot_env import BaseRobotEnv
   
   class MyRobotEnv(BaseRobotEnv):
       def _load_robot(self):
           # Load your URDF/robot model
           pass
       
       def _get_observation(self):
           # Return robot state
           pass
       
       def _compute_reward(self):
           # Calculate reward
           pass
       
       def _is_done(self):
           # Check termination conditions
           pass
       
       def _apply_action(self, action):
           # Apply action to robot
           pass
   ```

2. **Update the training script** to use your environment:
   ```python
   from src.environments.my_robot_env import MyRobotEnv
   
   env = MyRobotEnv(...)
   ```

3. **Define observation and action spaces** in your environment's `__init__` method

## Monitoring and Visualization

### TensorBoard
```bash
# Local
tensorboard --logdir runs/

# Docker
docker-compose up tensorboard
```
Access at http://localhost:6006

### Weights & Biases (Optional)
Enable in config:
```yaml
monitoring:
  use_wandb: true
  wandb_project: "rl-robotics"
  wandb_entity: "your-username"
```

## Algorithms Supported

- **PPO (Proximal Policy Optimization)** - Default, good general performance
- **SAC (Soft Actor-Critic)** - Continuous control tasks
- **TD3 (Twin Delayed DDPG)** - Continuous control with stability
- **A2C (Advantage Actor-Critic)** - Faster training, less sample efficient

## Cost Optimization

### Spot Fleet Pricing
- g4dn.xlarge: ~$0.35-0.53/hour (vs $0.526 on-demand)
- g4dn.2xlarge: ~$0.70-1.05/hour (vs $0.752 on-demand)
- p3.2xlarge: ~$0.90-1.50/hour (vs $3.06 on-demand)

### Tips
1. Use spot instances for training (70% cost savings)
2. Auto-terminate instances when training completes
3. Use S3 lifecycle policies to manage old checkpoints
4. Start with smaller fleet size and scale up

## Troubleshooting

### Common Issues

**PyBullet not rendering:**
- Ensure you have OpenGL libraries installed
- Use `render_mode=None` for headless training

**CUDA out of memory:**
- Reduce batch size in config
- Use smaller network architecture
- Use gradient accumulation

**Spot instances terminated:**
- Checkpoints are auto-saved to S3
- Training resumes from last checkpoint
- Consider using on-demand instances for critical runs

**Docker GPU issues:**
- Install nvidia-docker2
- Verify with: `docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

## Development

### Running Tests
```bash
# Run environment tests
python -m pytest tests/

# Test a specific environment
python -c "from src.environments import SimpleRobotEnv; env = SimpleRobotEnv(); env.reset()"
```

### Code Structure
- Keep environment code in `src/environments/`
- Add new algorithms in `src/agents/`
- Training logic in `src/training/`
- Configuration in `configs/`

## Contributing

When adding new features:
1. Follow the existing code structure
2. Update configuration files as needed
3. Add appropriate documentation
4. Test locally before deploying to AWS

## Next Steps

1. **Bring your robot model:**
   - Add URDF/robot files to the project
   - Create custom environment class
   - Define task-specific rewards

2. **Define your training task:**
   - Walking/locomotion
   - Manipulation (picking/placing)
   - Navigation
   - Custom objectives

3. **Tune hyperparameters:**
   - Experiment with learning rates
   - Adjust network architecture
   - Try different algorithms

4. **Scale training:**
   - Deploy to AWS Spot Fleet
   - Monitor with TensorBoard
   - Iterate on performance

## Resources

- [PyBullet Documentation](https://pybullet.org/)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [AWS Spot Fleet Guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-fleet.html)

## License

MIT License - feel free to use for your robotics projects!

## Support

For issues or questions:
- Check the troubleshooting section
- Review configuration files
- Examine logs in `logs/` directory
- Check TensorBoard for training metrics

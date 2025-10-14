"""Setup script for RL Robotics package."""

from setuptools import setup, find_packages

setup(
    name="rl-robotics",
    version="0.1.0",
    description="RL Robotics Training Framework with AWS Spot Fleet",
    author="RL Robotics Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "stable-baselines3>=2.0.0",
        "sb3-contrib>=2.0.0",
        "pybullet>=3.2.5",
        "gymnasium>=0.29.0",
        "boto3>=1.28.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "tensorboard>=2.13.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rl-train=src.training.trainer:main",
            "rl-eval=src.training.evaluator:main",
        ],
    },
)

.PHONY: help install train evaluate docker-build docker-run test clean aws-deploy

help:
	@echo "RL Robotics Training - Available Commands:"
	@echo ""
	@echo "  make install        - Install Python dependencies"
	@echo "  make train          - Start local training"
	@echo "  make evaluate       - Evaluate trained model"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run training in Docker"
	@echo "  make test           - Run tests"
	@echo "  make clean          - Clean up generated files"
	@echo "  make aws-deploy     - Deploy AWS infrastructure"
	@echo ""

install:
	pip install -r requirements.txt

train:
	python train.py --config configs/training_config.yaml

evaluate:
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make evaluate MODEL=path/to/model"; \
		exit 1; \
	fi
	python evaluate.py $(MODEL) --episodes 10

docker-build:
	docker build -t rl-robotics:latest .

docker-run:
	docker-compose up training

docker-tensorboard:
	docker-compose up tensorboard

test:
	python -c "from src.environments import SimpleRobotEnv; env = SimpleRobotEnv(); env.reset(); print('Environment test passed!')"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf logs/* models/* runs/* checkpoints/* 2>/dev/null || true

aws-deploy:
	@if [ -z "$(VPC_ID)" ] || [ -z "$(SUBNET_IDS)" ] || [ -z "$(KEY_NAME)" ]; then \
		echo "Usage: make aws-deploy VPC_ID=vpc-xxx SUBNET_IDS=subnet-xxx,subnet-yyy KEY_NAME=your-key"; \
		exit 1; \
	fi
	cd aws && ./deploy.sh --vpc-id $(VPC_ID) --subnet-ids $(SUBNET_IDS) --key-name $(KEY_NAME)

aws-push:
	cd aws && ./push-image.sh

# Development targets
format:
	black src/ *.py

lint:
	flake8 src/ *.py

type-check:
	mypy src/

.PHONY: help install train evaluate docker-build docker-run test clean aws-deploy pipeline

help:
	@echo "RL Robotics Training - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install             - Install Python dependencies"
	@echo ""
	@echo "Local Training:"
	@echo "  make train               - Start single session training"
	@echo "  make pipeline            - Run complete training pipeline"
	@echo "  make pipeline-resume     - Resume interrupted pipeline"
	@echo "  make evaluate            - Evaluate trained model"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build        - Build Docker image"
	@echo "  make docker-run          - Run training in Docker"
	@echo "  make docker-tensorboard  - Run TensorBoard in Docker"
	@echo ""
	@echo "AWS Deployment:"
	@echo "  make aws-deploy          - Deploy EC2 Spot Fleet infrastructure"
	@echo "  make aws-deploy-sm       - Deploy SageMaker infrastructure"
	@echo "  make aws-push            - Push Docker image to ECR"
	@echo "  make aws-pipeline        - Run pipeline on SageMaker"
	@echo ""
	@echo "Development:"
	@echo "  make test                - Run environment tests"
	@echo "  make format              - Format code with black"
	@echo "  make lint                - Lint code with flake8"
	@echo "  make clean               - Clean up generated files"
	@echo ""

install:
	pip install -r requirements.txt

train:
	python train.py --config configs/training_config.yaml

# Pipeline commands
pipeline:
	python3 run_pipeline.py --config configs/pipeline_config.yaml

pipeline-resume:
	python3 run_pipeline.py --config configs/pipeline_config.yaml --resume

pipeline-s3:
	@if [ -z "$(S3_BUCKET)" ]; then \
		echo "Usage: make pipeline-s3 S3_BUCKET=bucket-name"; \
		exit 1; \
	fi
	python3 run_pipeline.py --config configs/pipeline_config.yaml --use-s3 --s3-bucket $(S3_BUCKET)

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

aws-deploy-sm:
	@if [ -z "$(VPC_ID)" ] || [ -z "$(SUBNET_IDS)" ]; then \
		echo "Usage: make aws-deploy-sm VPC_ID=vpc-xxx SUBNET_IDS=subnet-xxx,subnet-yyy"; \
		exit 1; \
	fi
	cd aws && ./deploy_sagemaker.sh --vpc-id $(VPC_ID) --subnet-ids $(SUBNET_IDS)

aws-push:
	cd aws && ./push-image.sh

aws-pipeline:
	@if [ -z "$(ROLE_ARN)" ]; then \
		echo "Usage: make aws-pipeline ROLE_ARN=arn:aws:iam::123456789012:role/SageMakerRole"; \
		exit 1; \
	fi
	python3 run_pipeline.py --config configs/pipeline_config_aws.yaml \
		--platform sagemaker --role-arn $(ROLE_ARN)

# Development targets
format:
	black src/ *.py

lint:
	flake8 src/ *.py

type-check:
	mypy src/

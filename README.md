# Data Engineering II — Distributed ML Pipeline for Diabetes Prediction

A production-grade distributed machine learning system demonstrating end-to-end ML infrastructure using Python, TensorFlow, Docker, Celery, and cloud deployment on OpenStack.

## 🎯 Project Overview

This project implements a **distributed diabetes prediction pipeline** that:
- Collects GitHub repository features via GitHub API
- Trains a 3-layer neural network using TensorFlow/Keras
- Deploys the model as a production REST API with async workers
- Uses Celery + RabbitMQ for distributed task processing
- Demonstrates cloud infrastructure automation with Ansible and OpenStack

## 🏗️ Architecture

```
┌─────────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│    Client VM        │         │      Dev VM      │         │     Prod VM      │
│                     │         │                  │         │   (Docker)       │
│  Provisioning &     │ ──────►│ Data Collection  │ ──────►│  Flask API       │
│  Orchestration      │         │ Model Training   │         │  Celery Workers  │
│                     │         │                  │         │  RabbitMQ        │
└─────────────────────┘         └──────────────────┘         └──────────────────┘
      (Ansible)                  (TensorFlow)              (Docker Compose)
```

## 📋 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **ML Framework** | TensorFlow/Keras | Binary classification model (diabetes prediction) |
| **Task Queue** | Celery + RabbitMQ | Async job distribution and management |
| **Web Framework** | Flask | REST API endpoints for predictions |
| **Containerization** | Docker | Production deployment isolation |
| **IaC** | Ansible | VM configuration and dependency installation |
| **Cloud** | OpenStack (SSC/SNIC) | VM provisioning and infrastructure |
| **Scripting** | Python | All automation and ML code |

## 📦 Project Structure

```
data-engineering-II-project/
│
├── crawler/                          # Data Collection
│   ├── github_crawler.py            # GitHub API scraper
│   └── repos.csv                    # Output dataset
│
├── ci_cd/
│   ├── development_server/          # Model Training
│   │   ├── neural_net.py           # TensorFlow training script
│   │   ├── github-repository-data.csv
│   │   ├── model.h5                # Trained weights
│   │   └── model.json              # Model architecture
│   │
│   └── production_server/           # Production Deployment
│       ├── app.py                  # Flask REST API
│       ├── workerA.py              # Celery tasks
│       ├── docker-compose.yml
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── static/
│       │   └── Chart.min.js
│       └── templates/
│           ├── result.html
│           ├── chart.html
│           └── bar_chart.html
│
├── openstack-client/                # Infrastructure Setup
│   ├── start_dev_prod_instances.py
│   ├── ansible_configuration.yml
│   ├── constants.py
│   ├── setup_client_vm.sh
│   └── UPPMAX_2026_1-24_openrc.sh
│
├── project-report/                  # Documentation
│   ├── project-report.tex
│   └── sections/
│       ├── introduction.tex
│       ├── system_architecture.tex
│       ├── results.tex
│       └── conclusion.tex
│
└── doc/
    └── DE2_Project3_Implementation_Guide.md
```

## 🚀 Quick Start

### Prerequisites
- OpenStack account with API credentials (SSC/SNIC)
- GitHub personal access token (for API rate limit)
- SSH keypair
- Ansible installed

### Step 1: Provision Infrastructure

```bash
cd openstack-client/
source UPPMAX_2026_1-24_openrc.sh

# Generate SSH keys
ssh-keygen -t rsa -f ~/.ssh/cluster-key

# Update SSH public key in dev-cloud-cfg.txt and prod-cloud-cfg.txt
cat ~/.ssh/cluster-key.pub
# Paste into both config files

# Provision VMs
python3 start_dev_prod_instances.py
# Wait 2-3 minutes for cloud-init
```

### Step 2: Configure VMs with Ansible

```bash
ansible-playbook -i inventory.ini ansible_configuration.yml \
  --private-key=~/.ssh/cluster-key
```

### Step 3: Train Model on Dev VM

```bash
ssh -i ~/.ssh/cluster-key appuser@<DEV_IP>
cd /data-engineering-II-project

# Set GitHub token (optional)
export GITHUB_TOKEN=ghp_XXXXXXXXXXXX

# Collect data
python3 crawler/github_crawler.py

# Train model
cp crawler/repos.csv ci_cd/development_server/github-repository-data.csv
cd ci_cd/development_server/
python3 neural_net.py
```

### Step 4: Deploy to Production via Git Hook

Set up automated deployment using Git hooks:

```bash
# On Dev VM - authorize Prod SSH access
cat /home/appuser/.ssh/id_rsa.pub

# On Prod VM - add Dev's public key
echo "<Dev's public key>" >> /home/appuser/.ssh/authorized_keys
chmod 600 /home/appuser/.ssh/authorized_keys

# On Dev VM - create post-receive hook
cat > /opt/model_repo.git/hooks/post-receive << 'HOOK'
#!/bin/bash
PROD_IP="<PROD_IP>"
scp -i /home/appuser/.ssh/id_rsa \
    -o StrictHostKeyChecking=no \
    /data-engineering-II-project/ci_cd/development_server/model.h5 \
    appuser@${PROD_IP}:/data-engineering-II-project/ci_cd/production_server/model.h5

ssh -i /home/appuser/.ssh/id_rsa \
    -o StrictHostKeyChecking=no \
    appuser@${PROD_IP} \
    "cd /data-engineering-II-project/ci_cd/production_server && docker compose restart worker_1"
HOOK
chmod +x /opt/model_repo.git/hooks/post-receive

# On Dev VM - add deployment remote
cd /data-engineering-II-project
git remote add deployment /opt/model_repo.git

# Deploy model (hook triggers automatically)
git add ci_cd/development_server/model.h5 ci_cd/development_server/model.json
git commit -m "deploy: trained model"
git push deployment main
```

### Step 5: Test the API

```bash
# Access Flask app
curl http://<PROD_IP>:5100/

# Get predictions and accuracy
curl -X POST http://<PROD_IP>:5100/predictions
curl -X POST http://<PROD_IP>:5100/accuracy
```

## 📊 Model Details

### Training Dataset
- **Source**: Pima Indians Diabetes Dataset
- **Samples**: Variable (up to 1000 from GitHub repos)
- **Features**: 8 numerical attributes
- **Target**: Binary classification (diabetes: yes/no)

### Neural Network Architecture
```
Input (8 features)
    ↓
Dense(16, relu) + Dropout
    ↓
Dense(8, relu)
    ↓
Dense(1, sigmoid)  ← Binary output
```

### Training Configuration
- **Loss**: Binary crossentropy
- **Optimizer**: Adam
- **Epochs**: 250
- **Batch Size**: 10
- **Validation**: Accuracy metric

## 🐳 Production Stack

### Docker Containers

#### Web Service
```dockerfile
Flask API on port 5100
├── Listens for HTTP requests
├── Dispatches tasks to Celery
└── Returns predictions via JSON
```

#### RabbitMQ
```
Message broker on port 5672
├── AMQP queue management
├── Celery task distribution
└── Results backend
```

#### Celery Worker
```
Task executor (configurable instances)
├── Loads pre-trained model
├── Executes prediction tasks
└── Stores results
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome page |
| POST | `/accuracy` | Get model accuracy on test data |
| POST | `/predictions` | Get model predictions with visualization |

## 📈 Scalability Testing

Scale Celery workers and measure throughput:

```bash
# Scale to 4 workers
docker compose up -d --scale worker_1=4

# Scale back to 1
docker compose up -d --scale worker_1=1
```

Compare metrics:
- **Baseline (1 worker)**: Baseline throughput
- **Scaled (4 workers)**: Improved throughput and reduced latency

## 🔧 Celery Task Examples

```python
from workerA import add_nums, get_predictions, get_accuracy

# Simple addition task
result = add_nums.delay(3, 4)
print(result.get(timeout=10))  # 7

# Prediction task
result = get_predictions.delay()
predictions = result.get(timeout=30)
print(f"Predictions: {predictions['predicted']}")
print(f"Actual: {predictions['y']}")

# Accuracy task
result = get_accuracy.delay()
accuracy = result.get(timeout=30)
print(f"Accuracy: {accuracy:.2f}%")
```

## 📝 Key Features

✅ **End-to-End ML Pipeline**
- Data collection from GitHub API
- Feature engineering
- Model training and validation
- Production deployment

✅ **Distributed Computing**
- Celery for async task distribution
- RabbitMQ for message brokering
- Scalable worker pools

✅ **Cloud Infrastructure**
- OpenStack VM provisioning
- Ansible configuration management
- Docker containerization

✅ **REST API**
- Flask-based HTTP endpoints
- JSON request/response format
- Real-time predictions

✅ **Monitoring & Logging**
- Docker Compose logging
- Celery task tracking
- Model performance metrics

## 🛠️ Development Workflow

### Local Testing
```bash
# Test neural network locally
python3 ci_cd/development_server/neural_net.py

# Test API locally (before cloud deployment)
python3 ci_cd/production_server/app.py
```

### Cloud Deployment Pipeline
```bash
# 1. Provision infrastructure on OpenStack
python3 openstack-client/start_dev_prod_instances.py

# 2. Configure VMs with Ansible
ansible-playbook -i openstack-client/inventory.ini openstack-client/ansible_configuration.yml

# 3. Train model on Dev VM
python3 crawler/github_crawler.py
python3 ci_cd/development_server/neural_net.py

# 4. Deploy via Git Hook (automatic SCP + worker restart)
git push deployment main
```

The **post-receive hook** handles:
- Copying trained model (`model.h5`) to Prod VM via SCP
- Restarting Celery workers automatically
- No manual intervention needed after git push

## 📚 Documentation

- **[Implementation Guide](doc/DE2_Project3_Implementation_Guide.md)** - Detailed setup and deployment instructions
- **[Project Report](project-report/project-report.tex)** - Academic paper with results and analysis

## ⚠️ Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| OpenStack commands fail | Run: `source UPPMAX_2026_1-24_openrc.sh` |
| Ansible can't connect | Wait 2-3 minutes for cloud-init to complete |
| Celery tasks stay PENDING | Check: `docker compose ps` and restart RabbitMQ |
| Model file not found | Verify SCP paths match container volumes |
| GitHub API rate limit | Generate new PAT or increase `--min-stars` threshold |
| Port conflicts | Change ports in docker-compose.yml |

## 👥 Team

- Arnab Kumar Ghosh
- Eric Buxton
- Prodip Kumar Das
- Rahul Bhat

**Course**: Data Engineering II  
**Institution**: Uppsala University  
**Date**: May 2026

## 📄 License

This project is part of the Data Engineering II course at Uppsala University.

## 🔗 Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Celery User Guide](https://docs.celeryproject.io/)
- [Docker Documentation](https://docs.docker.com/)
- [Ansible Documentation](https://docs.ansible.com/)
- [OpenStack Documentation](https://docs.openstack.org/)
- [GitHub API Documentation](https://docs.github.com/en/rest)

---

**Last Updated**: May 21, 2026  
**Status**: Production Ready

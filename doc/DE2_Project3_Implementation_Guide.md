# DE-II Project 3: GitHub Star Prediction with Distributed ML — Implementation Guide

> **Stack:** Python · TensorFlow/Keras · Docker · Celery · RabbitMQ · Flask · Ansible · OpenStack (SSC/SNIC)
> **Project Type:** Distributed machine learning pipeline with REST API and asynchronous workers
> **GitHub Repo:** `https://github.com/bdtiger/data-engineering-II-project`

---

## Project Overview

This project implements a **distributed machine learning pipeline** for GitHub repository star count prediction using TensorFlow/Keras. It demonstrates:
- **Data collection** via GitHub API crawler (2000+ repositories)
- **Model training** with TensorFlow/Keras neural networks (regression task)
- **Production serving** with Flask REST API + Celery asynchronous workers
- **Cloud infrastructure** deployment on OpenStack with Ansible automation
- **CI/CD pipeline** using Git hooks for automated model deployment

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  OpenStack Cloud (SSC/SNIC)                                          │
│                                                                      │
│  ┌─────────────────┐   Ansible provision    ┌──────────────────────┐ │
│  │  Client VM      │ ──────────────────────►│  Dev VM              │ │
│  │                 │                        │  · GitHub crawler    │ │
│  │  start_         │ ──────────────────────►│  · neural_net.py     │ │
│  │  instances.py   │   Ansible provision    │  · model training    │ │
│  │  Ansible ctrl   │                        │  · bare git repo     │ │
│  └─────────────────┘                        └──────────┬───────────┘ │
│                                                        │ git push    │
│                                                        │ (deploy)    │
│                                                        ▼             │
│                                             ┌──────────────────────┐ │
│                                             │  Prod VM (Docker)    │ │
│                                             │  · Flask web app     │ │
│                                             │  · Celery workers    │ │
│                                             │  · RabbitMQ broker   │ │
│                                             │  · TensorFlow model  │ │
│                                             └──────────────────────┘ │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### End-to-end Flow
1. **Infrastructure Setup**: Client VM runs `start_dev_prod_instances.py` → Dev and Prod VMs provisioned on OpenStack
2. **Configuration**: Client VM runs Ansible playbook → installs Python packages on Dev, Docker/RabbitMQ on Prod
3. **Data Collection**: Dev VM runs `github_crawler.py` → fetches top repos from GitHub API → saves features to CSV
4. **Model Training**: Dev VM runs `neural_net.py` → trains TensorFlow model on dataset → saves model files
5. **Deployment**: Dev VM pushes to deployment branch → Git hook triggered → model copied to Prod → Celery workers restart
6. **Production Serving**: Flask API receives requests → Celery tasks executed → model predictions returned with accuracy metrics

---

## Phase 0 — Prerequisites on the Client VM

### 0.1 Set up the Client VM

Run the provided setup script:

```bash
bash setup_client_vm.sh
cd data-engineering-II-project/openstack-client/
```

This installs all dependencies (Ansible, OpenStack CLI tools, Python packages) and clones the project repo.

### 0.2 Source your OpenStack credentials

Download `UPPMAX_2026_1-24_openrc.sh` from the SSC dashboard (Project → API Access → Download OpenStack RC File):

```bash
cd data-engineering-II-project/openstack-client/
source UPPMAX_2026_1-24_openrc.sh

# Verify
openstack server list
```

Re-run `source` whenever you open a new shell — credentials are not persistent.

---

## Phase 1 — Provision VMs with OpenStack

All provisioning scripts live in `openstack-client/`:

```
data-engineering-II-project/
└── openstack-client/
    ├── setup_client_vm.sh                (Client VM setup)
    ├── UPPMAX_2026_1-24_openrc.sh        (OpenStack credentials)
    ├── constants.py                      (Flavor, image, network, key name)
    ├── setup_var.yml                     (Prod/dev home paths)
    ├── start_dev_prod_instances.py       (Provision Dev & Prod VMs)
    ├── start_worker_instances.py         (Provision extra worker VMs if needed)
    ├── dev-cloud-cfg.txt                 (cloud-init: creates appuser on Dev)
    ├── prod-cloud-cfg.txt                (cloud-init: creates appuser on Prod)
    ├── ansible_configuration.yml         (Configures both VMs)
    ├── inventory.ini                     (Ansible inventory)
    └── ansible_configuration.yml         (Ansible config)
```

### 1.1 Review `constants.py`

Check these values match your OpenStack project before running anything:

```python
FLAVOR      = "ssc.medium"
PRIVATE_NET = "UPPMAX 2026/1-24 Internal IPv4 Network"
IMAGE_NAME  = "Ubuntu 22.04 - 2024.01.15"
KEY_NAME    = "de1-course-snic-key"
```

### 1.2 Provision Dev and Prod VMs

Generate SSH keys:
```bash
ssh-keygen -t rsa
# enter the key name /home/ubuntu/cluster-keys/cluster-key
```

Update SSH public keys in cloud-init config files:

```bash
# Open prod-cloud-cfg.txt and copy your SSH public key into ssh_authorized_keys:
cat /home/ubuntu/cluster-keys/cluster-key.pub
# Paste into prod-cloud-cfg.txt under ssh_authorized_keys

# Do the same for dev-cloud-cfg.txt
```

Provision the VMs:
```bash
cd data-engineering-II-project/openstack-client/
python3 start_dev_prod_instances.py
```

This creates two VMs and prints their IP addresses. **Wait 2–3 minutes** for cloud-init to complete before running Ansible.

---

## Phase 2 — Configure VMs with Ansible

Ansible configures both VMs: installs packages on Dev, sets up Docker/RabbitMQ on Prod.

```bash
cd data-engineering-II-project/openstack-client/

# Check both VMs are reachable
ansible-inventory -i inventory.ini --list

# Run the full playbook
ansible-playbook -i inventory.ini ansible_configuration.yml --private-key=/home/ubuntu/cluster-keys/cluster-key
```

After completion:
- **Dev VM**: Python packages installed, ready for training
- **Prod VM**: Docker running, Flask app on port 5100, RabbitMQ on port 5672

---

## Phase 3 — Data Collection & Model Training on Dev

All steps run on the Dev VM using the scripts in `crawler/` and `ci_cd/development_server/`.

### 3.1 Collect GitHub Repository Data

```bash
# SSH into Dev VM
ssh -i ~/.ssh/your_key appuser@<DEV_IP>
cd /data-engineering-II-project

# Set GitHub API token (optional, increases rate limit)
export GITHUB_TOKEN=ghp_XXXXXXXXXXXX

# Crawl top GitHub repositories
python3 crawler/github_crawler.py
```

This script:
- Searches for repositories with ≥50 stars
- Collects up to 2000 repositories, sorted by stars (descending)
- Extracts 15 features per repo: full_name, stargazers_count, forks_count, subscribers_count, open_issues_count, size, network_count, language, has_wiki, has_pages, has_issues, topics_count, age_days, created_at, pushed_at
- Applies rate-limit handling (2-second delay between pages)
- Saves to `crawler/repos.csv`

### 3.2 Train the TensorFlow Model

```bash
# Move dataset to development directory
cd ci_cd/development_server/

# Train neural network model
python3 neural_network.py
```

This script:
- Loads dataset from `github-repository-data.csv`
- Encodes language features using factorization
- Transforms target to log scale using `log1p(stargazers_count)`
- Selects 11 features: forks_count, subscribers_count, open_issues_count, size, network_count, has_wiki, has_pages, has_issues, topics_count, age_days, language_encoded
- Applies StandardScaler normalization on features
- Splits data: 80% train, 20% test
- Saves model to `model.json` (architecture) and `model.weights.h5` (weights)
- Saves training time to `scalability_results.csv`

**Model Architecture:**
- Input layer: 11 features
- Hidden layer 1: 64 neurons, ReLU activation
- Dropout: 0.2
- Hidden layer 2: 32 neurons, ReLU activation
- Dropout: 0.2
- Output layer: 1 neuron (linear, regression task)
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam (learning rate 0.001)
- Training: Maximum 100 epochs with early stopping (patience=10, batch size=32)
- Validation: 10% of training data
- Output bias initialization: mean of training target

---

## Phase 4 — Deploy Model to Production via Git Hook

Deployment is automated through a Git post-receive hook. When you push the trained model to the deployment branch, the hook automatically copies the model to Prod and restarts Celery workers.

### 4.1 Set Up Git Hook Deployment

First, authorize Dev's SSH key on Prod:

```bash
# On Dev VM — get the public key
cat /home/appuser/.ssh/id_rsa.pub
```

Copy that output, then on Prod VM:

```bash
# On Prod VM
echo "<paste key here>" >> /home/appuser/.ssh/authorized_keys
chmod 600 /home/appuser/.ssh/authorized_keys
```

Verify Dev-to-Prod SSH works:

```bash
# On Dev VM
ssh -i /home/appuser/.ssh/id_rsa \
    -o StrictHostKeyChecking=no \
    appuser@<PROD_IP> "echo SSH OK"
```

### 4.2 Create the Post-Receive Hook

On Dev VM, create the post-receive hook:

```bash
# On Dev VM
cat > /opt/model_repo.git/hooks/post-receive << 'HOOK'
#!/bin/bash
PROD_IP="<PROD_IP>"   # Replace with actual Prod IP
MODEL_WEIGHTS_SRC="/data-engineering-II-project/ci_cd/development_server/model.weights.h5"
MODEL_JSON_SRC="/data-engineering-II-project/ci_cd/development_server/model.json"
MODEL_WEIGHTS_DEST="/data-engineering-II-project/ci_cd/production_server/model.weights.h5"
MODEL_JSON_DEST="/data-engineering-II-project/ci_cd/production_server/model.json"

echo "==> Deploying model weights to Prod..."
scp -i /home/appuser/.ssh/id_rsa \
    -o StrictHostKeyChecking=no \
    $MODEL_WEIGHTS_SRC appuser@${PROD_IP}:${MODEL_WEIGHTS_DEST}

echo "==> Deploying model JSON to Prod..."
scp -i /home/appuser/.ssh/id_rsa \
    -o StrictHostKeyChecking=no \
    $MODEL_JSON_SRC appuser@${PROD_IP}:${MODEL_JSON_DEST}

echo "==> Restarting Celery worker on Prod..."
ssh -i /home/appuser/.ssh/id_rsa \
    -o StrictHostKeyChecking=no \
    appuser@${PROD_IP} \
    "cd /data-engineering-II-project/ci_cd/production_server && docker compose restart worker_1"

echo "==> Deploy done."
HOOK

chmod +x /opt/model_repo.git/hooks/post-receive
```

### 4.3 Add Deployment Remote and Deploy

Set up the deployment remote on Dev VM:

```bash
# On Dev VM
cd /data-engineering-II-project
git remote get-url deployment 2>/dev/null || \
git remote add deployment /opt/model_repo.git
```

After training completes, commit and push the model:

```bash
# On Dev VM
cd /data-engineering-II-project
git add ci_cd/development_server/model.weights.h5 \
        ci_cd/development_server/model.json
git commit -m "deploy: trained neural network model"
git push deployment main
# The post-receive hook fires automatically → model weights and JSON copied to Prod → workers restart
```

---

## Phase 5 — Test the Production Application

### 5.1 Check Containers Status

```bash
# On Prod VM
cd /data-engineering-II-project/ci_cd/production_server/
docker compose ps
# Should show: web, rabbit, and worker_1 all Up
```

### 5.2 Access the Flask Web Interface

Open browser and navigate to:
```
http://<PROD_IP>:5100/
```

Available endpoints:
- `GET /` - Home page with endpoint descriptions
- `GET /accuracy` - Form to trigger accuracy evaluation
- `POST /accuracy` - Evaluate model on test data, returns Mean Absolute Error (MAE)
- `GET /predictions` - Form to trigger batch predictions
- `POST /predictions` - Run batch inference on all repositories, displays predicted vs actual star counts with accuracy metric

### 5.3 Test Celery Workers Directly

```python
# Run on Prod VM in Python shell
from workerA import add_nums, get_predictions, get_accuracy

# Test simple addition
result = add_nums.delay(3, 4)
print(result.get(timeout=10))  # Returns: 7

# Get predictions (returns dict with predicted/actual star counts)
result = get_predictions.delay()
predictions = result.get(timeout=30)
print(f"Repo names (first 5): {predictions['names'][:5]}")
print(f"Predicted stars (first 5): {predictions['predicted'][:5]}")
print(f"Actual stars (first 5): {predictions['y'][:5]}")

# Get model accuracy (Mean Absolute Error in log-space)
result = get_accuracy.delay()
mae = result.get(timeout=30)
print(f"Model MAE (log-space): {mae:.4f}")
```

---

## Phase 6 — Scalability Testing

### 6.1 Horizontal Scalability Test

Scale Celery workers and measure performance impact.

```bash
# On Prod VM
cd /data-engineering-II-project/ci_cd/production_server/

# View current workers
docker compose ps

# Scale up to multiple workers
docker compose up -d --scale worker_1=2
docker compose up -d --scale worker_1=3

# Scale down to 1 worker
docker compose up -d --scale worker_1=1
```

### 6.2 Test Results

The horizontal scalability tests were conducted with 8 concurrent training tasks across different numbers of worker VMs. Results show:

| Number of VMs | Total Time (seconds) | Speedup | Efficiency |
|---|---|---|---|
| 1 | 182.0 | 1.00x | 100% |
| 2 | 95.0 | 1.91x | 95.5% |
| 3 | 70.0 | 2.60x | 86.7% |

**Key Findings:**
- **2 VMs**: 1.91x speedup (48% reduction in execution time) with nearly linear scaling
- **3 VMs**: 2.60x speedup (62% reduction in execution time) approaching theoretical 3.0x ideal speedup
- **Scaling Efficiency**: System maintains 86-95% efficiency, indicating good load distribution across worker nodes
- The slight deviation from ideal linear scaling at 3 VMs is expected due to:
  - RabbitMQ message broker overhead
  - Network communication latency between workers and broker
  - Task queue serialization/deserialization costs

**Visual Representation:** See `project-report/figures/Graph.webp` for wall-clock time comparison and speedup curves.

### 6.3 Benchmark Script

Submit multiple tasks and measure response time at different worker counts.

```python
# Load testing script
from workerA import get_predictions
import time

task_ids = []
start = time.time()
for i in range(10):
    task_ids.append(get_predictions.delay())
elapsed = time.time() - start
print(f"Submitted 10 tasks in {elapsed:.2f}s")

# Wait for completion and measure total time
import time
time.sleep(5)  # Allow tasks to complete
for task_id in task_ids:
    try:
        result = task_id.get(timeout=30)
        print(f"Task completed: {result['names'][0]} -> {result['predicted'][0]} stars")
    except Exception as e:
        print(f"Task failed: {e}")
```

### 6.4 Scaling Guidelines

**Recommendations for different workloads:**
- **Light load (<5 tasks/min)**: 1 worker sufficient
- **Medium load (5-50 tasks/min)**: 2-3 workers recommended
- **High load (>50 tasks/min)**: 4+ workers with load balancing

---

## Project File Structure

```
data-engineering-II-project/
├── crawler/
│   ├── github_crawler.py           # GitHub API crawler - fetches 2000+ repos by stars
│   └── repos.csv                   # Output: 15 features per repository
│
├── ci_cd/
│   ├── development_server/
│   │   ├── neural_network.py      # Model training script (TensorFlow/Keras regression)
│   │   ├── neural_network.ipynb   # Notebook for interactive training and experimentation
│   │   ├── linear_regression.py   # Baseline linear regression model
│   │   ├── analyze_models.py      # Model comparison and analysis script
│   │   ├── utils.py               # Utility functions for training
│   │   ├── github-repository-data.csv  # Input dataset (2000+ repos)
│   │   ├── model.weights.h5       # Trained model weights
│   │   ├── model.json             # Model architecture (TF/Keras JSON format)
│   │   ├── scalability_results.csv # Training timing metrics
│   │   └── method3.ipynb          # Alternative model development notebook
│   │
│   └── production_server/
│       ├── app.py                 # Flask REST API (port 5100)
│       ├── workerA.py             # Celery worker tasks (add_nums, get_predictions, get_accuracy)
│       ├── requirements.txt       # Python dependencies
│       ├── docker-compose.yml     # Docker orchestration (web, rabbit, worker_1)
│       ├── Dockerfile             # Python 3.12-slim base image
│       ├── run_task.py            # Utility for running tasks
│       ├── model.weights.h5       # Model weights (production copy)
│       ├── model.json             # Model architecture (production copy)
│       ├── github-repository-data.csv  # Test dataset
│       ├── static/                # Static files (Chart.min.js for visualizations)
│       └── templates/             # HTML templates (base.html, index.html, accuracy.html, predictions.html, result.html, chart.html, bar_chart.html)
│
├── openstack-client/
│   ├── start_dev_prod_instances.py    # Provision Dev and Prod VMs on OpenStack
│   ├── start_worker_instances.py      # Provision additional worker VMs
│   ├── ansible_configuration.yml      # Ansible playbook for VM configuration
│   ├── constants.py                   # OpenStack configuration (flavor, image, network)
│   ├── setup_client_vm.sh             # Client setup script
│   ├── setup_var.yml                  # Ansible variables for prod/dev paths
│   ├── inventory.ini                  # Ansible inventory
│   ├── dev-cloud-cfg.txt              # cloud-init for Dev VM
│   ├── prod-cloud-cfg.txt             # cloud-init for Prod VM
│   └── UPPMAX_2026_1-24_openrc.sh     # OpenStack credentials
│
├── project-report/                     # LaTeX project documentation
└── doc/
    └── DE2_Project3_Implementation_Guide.md  # This implementation guide
```

---

## Quick-Reference Command Sequence

```bash
# ── Client VM ───────────────────────────────────────────────────────────────
cd data-engineering-II-project/openstack-client/
source UPPMAX_2026_1-24_openrc.sh
ssh-keygen -t rsa -f ~/.ssh/cluster-key
# Update SSH keys in dev-cloud-cfg.txt and prod-cloud-cfg.txt
python3 start_dev_prod_instances.py        # Wait 2-3 minutes for cloud-init
ansible-playbook -i inventory.ini ansible_configuration.yml \
  --private-key=~/.ssh/cluster-key

# ── Dev VM ──────────────────────────────────────────────────────────────────
ssh -i ~/.ssh/cluster-key appuser@<DEV_IP>
cd /data-engineering-II-project
export GITHUB_TOKEN=ghp_XXXXXXXXXXXX
python3 crawler/github_crawler.py
cp crawler/repos.csv ci_cd/development_server/github-repository-data.csv
cd ci_cd/development_server/
python3 neural_network.py

# ── Set Up Git Hook Deployment ──────────────────────────────────────────────
# On Dev VM: authorize Prod's SSH key (see Phase 4.1)
# On Dev VM: create post-receive hook (see Phase 4.2)
# On Dev VM: add deployment remote
git remote add deployment /opt/model_repo.git

# ── Deploy to Prod via Git Push ─────────────────────────────────────────────
# On Dev VM
cd /data-engineering-II-project
git add ci_cd/development_server/model.weights.h5 ci_cd/development_server/model.json
git commit -m "deploy: trained model"
git push deployment main
# Post-receive hook fires automatically → SCP model files to Prod → restart workers

# ── Prod VM (Testing) ───────────────────────────────────────────────────────
ssh -i ~/.ssh/cluster-key appuser@<PROD_IP>
cd /data-engineering-II-project/ci_cd/production_server/
docker compose ps
# Access Flask app: http://<PROD_IP>:5100/
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `openstack server list` fails | `openrc` not sourced | `source UPPMAX_2026_1-24_openrc.sh` |
| `ansible-inventory` fails | `inventory.ini` not generated | Run `start_dev_prod_instances.py` first |
| SSH connection timeout | VMs still initializing | Wait 2–3 min after provisioning, retry |
| Docker containers won't start | Port conflicts | Check `docker compose ps` and verify ports |
| Celery tasks stay `PENDING` | RabbitMQ not running | `docker compose restart rabbit` |
| Model file not found in Prod | SCP path mismatch | Verify paths in docker volume mounts |
| GitHub API rate limit (403) | Token expired or missing | Generate new PAT in GitHub settings |
| `model.weights.h5` load error | Version mismatch | Use same TensorFlow version as training |

---

## Key Technologies

- **TensorFlow/Keras**: Deep learning framework for regression (64-32-1 architecture with dropout)
- **Celery**: Distributed task queue for asynchronous job processing
- **RabbitMQ**: Message broker (AMQP protocol) for Celery task distribution
- **Flask**: REST API web framework (Python 3.12, debug mode)
- **Docker**: Containerization (docker-compose orchestration with 3 services)
- **Ansible**: Infrastructure automation and configuration management
- **OpenStack**: Cloud infrastructure provider (SSC/SNIC)
- **scikit-learn**: StandardScaler for feature normalization
- **pandas**: Data loading and preprocessing
- **GitHub API**: Repository metadata collection (v3 REST API)

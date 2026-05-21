# DE-II Project 3: Diabetes Prediction with Distributed ML — Implementation Guide

> **Stack:** Python · TensorFlow/Keras · Docker · Celery · RabbitMQ · Flask · Ansible · OpenStack (SSC/SNIC)
> **Project Type:** Distributed machine learning pipeline with REST API and asynchronous workers
> **GitHub Repo:** `https://github.com/bdtiger/data-engineering-II-project`

---

## Project Overview

This project implements a **distributed machine learning pipeline** for diabetes prediction using the Pima Indians Diabetes Dataset. It demonstrates:
- **Data collection** via GitHub API
- **Model training** with TensorFlow/Keras neural networks
- **Production serving** with Flask REST API + Celery workers
- **Cloud infrastructure** deployment on OpenStack with Ansible
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
- Searches for repos with 50+ stars
- Collects up to 1000 repos
- Extracts features: stars, forks, issues, size, language, creation date, etc.
- Saves to `crawler/repos.csv`

### 3.2 Train the TensorFlow Model

```bash
# Move dataset to development directory
cp crawler/repos.csv ci_cd/development_server/github-repository-data.csv
cd ci_cd/development_server/

# Train neural network model
python3 neural_net.py
```

This script:
- Loads dataset from `github-repository-data.csv`
- Creates a 3-layer neural network (16-8-1 neurons)
- Trains for 250 epochs with batch size 10
- Saves model to `model.h5` and `model.json`
- Prints accuracy metrics

**Model Architecture:**
- Input layer: 8 features
- Hidden layer 1: 16 neurons, ReLU activation
- Hidden layer 2: 8 neurons, ReLU activation
- Output layer: 1 neuron, sigmoid activation (binary classification)
- Loss: binary crossentropy
- Optimizer: Adam

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
MODEL_SRC="/data-engineering-II-project/ci_cd/development_server/model.h5"
MODEL_DEST="/data-engineering-II-project/ci_cd/production_server/model.h5"

echo "==> Deploying model.h5 to Prod..."
scp -i /home/appuser/.ssh/id_rsa \
    -o StrictHostKeyChecking=no \
    $MODEL_SRC appuser@${PROD_IP}:${MODEL_DEST}

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
git add ci_cd/development_server/model.h5 \
        ci_cd/development_server/model.json
git commit -m "deploy: trained neural network model"
git push deployment main
# The post-receive hook fires automatically → model copied to Prod → workers restart
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
- `GET /` - Welcome page
- `POST /accuracy` - Get model accuracy on test data
- `POST /predictions` - Get model predictions with accuracy visualization

### 5.3 Test Celery Workers Directly

```python
# Run on Prod VM
from workerA import add_nums, get_predictions, get_accuracy

# Test simple addition
result = add_nums.delay(3, 4)
print(result.get(timeout=10))  # Should print 7

# Get predictions
result = get_predictions.delay()
predictions = result.get(timeout=30)
print(f"Predictions: {predictions['predicted'][:10]}")
print(f"Actual: {predictions['y'][:10]}")

# Get accuracy
result = get_accuracy.delay()
accuracy = result.get(timeout=30)
print(f"Model Accuracy: {accuracy:.2f}%")
```

---

## Phase 6 — Scalability Testing

Scale Celery workers and measure performance impact.

```bash
# On Prod VM
cd /data-engineering-II-project/ci_cd/production_server/

# View current workers
docker compose ps

# Scale up to 4 workers
docker compose up -d --scale worker_1=4

# Scale down to 1 worker
docker compose up -d --scale worker_1=1
```

**Benchmark:** Submit multiple tasks and measure response time at different worker counts.

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
```

---

## Project File Structure

```
data-engineering-II-project/
├── crawler/
│   ├── github_crawler.py           # GitHub API crawler - fetches repo data
│   └── repos.csv                   # Output: collected repository features
│
├── ci_cd/
│   ├── development_server/
│   │   ├── neural_net.py          # Model training script (TensorFlow/Keras)
│   │   ├── github-repository-data.csv  # Input dataset
│   │   └── model.h5               # Trained model weights
│   │
│   └── production_server/
│       ├── app.py                 # Flask REST API
│       ├── workerA.py             # Celery worker tasks
│       ├── requirements.txt       # Python dependencies
│       ├── docker-compose.yml     # Docker orchestration
│       ├── Dockerfile             # Container image definition
│       ├── model.h5               # Model weights (production)
│       ├── model.json             # Model architecture
│       ├── static/                # Static files (Chart.min.js)
│       └── templates/             # HTML templates (result.html, etc.)
│
└── openstack-client/
    ├── start_dev_prod_instances.py    # Provision VMs
    ├── ansible_configuration.yml       # Configuration management
    ├── constants.py                    # VM configuration
    ├── setup_client_vm.sh             # Client setup script
    └── UPPMAX_2026_1-24_openrc.sh     # OpenStack credentials
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
python3 neural_net.py

# ── Set Up Git Hook Deployment ──────────────────────────────────────────────
# On Dev VM: authorize Prod's SSH key (see Phase 4.1)
# On Dev VM: create post-receive hook (see Phase 4.2)
# On Dev VM: add deployment remote
git remote add deployment /opt/model_repo.git

# ── Deploy to Prod via Git Push ─────────────────────────────────────────────
# On Dev VM
cd /data-engineering-II-project
git add ci_cd/development_server/model.h5 ci_cd/development_server/model.json
git commit -m "deploy: trained model"
git push deployment main
# Post-receive hook fires automatically → SCP model to Prod → restart workers

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
| `model.h5` load error | Version mismatch | Use same TensorFlow version as training |

---

## Key Technologies

- **TensorFlow/Keras**: Deep learning framework (3-layer neural network)
- **Celery**: Distributed task queue for async job processing
- **RabbitMQ**: Message broker for Celery
- **Flask**: REST API web framework
- **Docker**: Containerization for production deployment
- **Ansible**: Infrastructure automation
- **OpenStack**: Cloud infrastructure provider

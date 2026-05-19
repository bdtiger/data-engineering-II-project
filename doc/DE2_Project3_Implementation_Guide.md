# DE-II Project 3: GitHub Star Predictor — Implementation Guide

> **Stack:** Python · Docker · Celery · RabbitMQ · Ansible · OpenStack (SSC/SNIC) · Git Hooks
> **Base repo:** `https://github.com/bdtiger/data-engineering-II-project`

---

## Architecture overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  OpenStack Cloud (SSC/SNIC)                                          │
│                                                                      │
│  ┌─────────────────┐   Ansible provision    ┌──────────────────────┐ │
│  │  Client VM      │ ──────────────────────►│  Dev VM              │ │
│  │                 │                        │  · crawl GitHub API  │ │
│  │  start_         │ ──────────────────────►│  · train models      │ │
│  │  instances.py   │   Ansible provision    │  · bare git repo     │ │
│  │  Ansible ctrl   │                        │  · post-receive hook │ │
│  └─────────────────┘                        └──────────┬───────────┘ │
│                                                        │ git push    │
│                                                        │ (Git Hook)  │
│                                                        ▼             │
│                                             ┌──────────────────────┐ │
│                                             │  Prod VM             │ │
│                                             │  · best_model.pkl    │ │
│                                             │  · Celery workers    │ │
│                                             │  · RabbitMQ broker   │ │
│                                             └──────────────────────┘ │
│                                                                      │
│  VM 4 & 5: start ONLY if needed for scalability testing              │
└──────────────────────────────────────────────────────────────────────┘
```

### End-to-end flow
1. Client VM runs `start_dev_prod_instances.py` → Dev and Prod VMs appear on OpenStack.
2. Client VM runs Ansible → both VMs get the repo cloned, Dev gets Python packages + bare git repo, Prod gets Docker + RabbitMQ + Celery containers running.
3. Dev VM: crawl GitHub API → extract features → train models → compare R² → pick best.
4. `git push deployment main` on Dev → post-receive hook fires → SCP `best_model.pkl` to Prod → containers restart.
5. Celery worker on Prod loads the new model and serves predictions via RabbitMQ.

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

Download `openrc.sh` from the SSC dashboard (Project → API Access → Download OpenStack RC File):

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
    └── ansible_configuration.yml         (Configures both VMs)
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
```bash
ssh-keygen -t rsa
# enter the key name /home/ubuntu/cluster-keys/cluster-key
```

open prod-cloud-cfg.txt delete the old key from the section `ssh_authorized_keys:` and
copy the complete contents of `/home/ubuntu/cluster-keys/cluster-key.pub` in the
`prod-cloud-cfg.txt` file.

Open the `dev-cloud-cfg.txt`. Delete the old key from the section
`ssh_authorized_keys:` and copy the complete contents of
`/home/ubuntu/cluster-keys/cluster-key.pub` in the `dev-cloud-cfg.txt` file.

```bash
cd data-engineering-II-project/openstack-client/
python3 start_dev_prod_instances.py
```

This creates two VMs (`group_5_dev_server_XXXX` and `group_5_prod_server_XXXX`) and prints their IP addresses. Note both IPs — you will need them.

**Wait 2–3 minutes** before running Ansible. cloud-init needs time to create the `appuser` account on both VMs.

---

## Phase 2 — Configure VMs with Ansible

Ansible does all the heavy lifting: installs packages, clones the repo, sets up the bare git repo on Dev, installs Docker and starts containers on Prod.

```bash
cd data-engineering-II-project/openstack-client/

# Check both VMs are reachable
ansible-inventory -i inventory.ini --list

# Run the full playbook
ansible-playbook -i inventory.ini ansible_configuration.yml --private-key=/home/ubuntu/cluster-keys/cluster-key
```

After this completes:

- **Dev VM** has: Python packages for crawling and training, a bare git repo at `/opt/model_repo.git`, and an SSH keypair at `/home/appuser/.ssh/id_rsa`.
- **Prod VM** has: Docker running, RabbitMQ 3.12 and Celery worker containers up, models directory at `/data-engineering-II-project/ci_cd/production_server/models/`.

---

## Phase 3 — Set Up the Git Hook CI/CD Pipeline

The post-receive hook on Dev is what triggers deployment to Prod automatically when you push a trained model.

### 3.1 Authorize Dev's SSH key on Prod

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

### 3.2 Verify Dev-to-Prod SSH works

```bash
# On Dev VM
ssh -i /home/appuser/.ssh/id_rsa \
    -o StrictHostKeyChecking=no \
    appuser@<PROD_IP> "echo SSH OK"
```

### 3.3 Create the post-receive hook on Dev

```bash
# On Dev VM
cat > /opt/model_repo.git/hooks/post-receive << 'HOOK'
#!/bin/bash
PROD_IP="<PROD_IP>"   # replace with actual prod IP
MODEL_SRC="/data-engineering-II-project/ci_cd/production_server/models/best_model.pkl"
MODEL_DEST="/data-engineering-II-project/ci_cd/production_server/models/best_model.pkl"

echo "==> Deploying best_model.pkl to Prod..."
scp -i /home/appuser/.ssh/id_rsa \
    -o StrictHostKeyChecking=no \
    $MODEL_SRC appuser@${PROD_IP}:${MODEL_DEST}

echo "==> Restarting Celery worker on Prod..."
ssh -i /home/appuser/.ssh/id_rsa \
    -o StrictHostKeyChecking=no \
    appuser@${PROD_IP} \
    "cd /data-engineering-II-project/ci_cd/production_server && docker compose restart worker"

echo "==> Deploy done."
HOOK

chmod +x /opt/model_repo.git/hooks/post-receive
```

### 3.4 Add the deployment remote on Dev

```bash
# On Dev VM
cd /data-engineering-II-project
git remote get-url deployment 2>/dev/null || \
git remote add deployment /opt/model_repo.git
```

---

## Phase 4 — Data Collection & Training on Dev

All steps run directly on the Dev VM — no Docker, no Celery involved.

```bash
# SSH into Dev
ssh -i ~/.ssh/your_key appuser@<DEV_IP>
cd /data-engineering-II-project

# 1. Collect 1000 repos with >= 50 stars (needs a GitHub PAT)
python3 data_collection/collect_github.py \
    --token ghp_XXXXXXXXXXXX \
    --total 1000 \
    --min-stars 50 \
    --out data.csv

# 2. Train all models and pick the best by R²
python3 training/train_models.py \
    --data      data.csv \
    --model-dir ci_cd/production_server/models/

# 3. Review results
cat ci_cd/production_server/models/metadata.json
```

> **Note on commits:** The GitHub search API does not return commit count directly.
> The features used (forks, watchers, open issues, size, age, etc.) are all available
> from the search endpoint and are good proxies for activity. Commit count would require
> a separate API call per repo (1000 extra requests) and is not required.

---

## Phase 5 — Deploy Model to Prod via Git Hook

```bash
# On Dev VM
cd /data-engineering-II-project

git add ci_cd/production_server/models/best_model.pkl \
        ci_cd/production_server/models/metadata.json
git commit -m "deploy: <ModelName> R2=<score>"
git push deployment main
# post-receive fires → SCP model to Prod → worker container restarts
```

---

## Phase 6 — Test the Production App

### 6.1 Check containers are running

```bash
# On Prod VM
cd /data-engineering-II-project/ci_cd/production_server
docker compose ps
# Should show: rabbit and worker both Up
```

### 6.2 Submit a prediction task

Send a task directly to the Celery worker via the RabbitMQ broker:

```python
# Run this on the Prod VM (or any machine that can reach Prod's RabbitMQ)
from workerA import get_predictions
result = get_predictions.delay()
print(result.get(timeout=30))
```

Or test the add_nums task first to confirm the queue is working:

```python
from workerA import add_nums
result = add_nums.delay(3, 4)
print(result.get(timeout=10))  # should print 7
```

### 6.3 Rank 5 GitHub repositories

Provide feature values for 5 repos and compare their predicted star counts:

```python
# On Prod VM
import joblib, numpy as np

model = joblib.load(
    '/data-engineering-II-project/ci_cd/production_server/models/best_model.pkl'
)

FEATURE_COLS = [
    "forks", "watchers", "open_issues", "size_kb",
    "has_wiki", "has_projects", "has_downloads", "is_fork",
    "age_days", "days_since_push", "topics_count",
    "network_count", "subscribers_count", "language_enc",
]

repos = [
    {"name": "facebook/react",        "forks": 44000,  "watchers": 44000,  "open_issues": 800,  "size_kb": 210000,  "has_wiki": 1, "has_projects": 0, "has_downloads": 1, "is_fork": 0, "age_days": 3900,  "days_since_push": 1, "topics_count": 5,  "network_count": 44000,  "subscribers_count": 6700, "language_enc": 7},
    {"name": "torvalds/linux",         "forks": 172000, "watchers": 172000, "open_issues": 400,  "size_kb": 4500000, "has_wiki": 0, "has_projects": 0, "has_downloads": 1, "is_fork": 0, "age_days": 12800, "days_since_push": 0, "topics_count": 0,  "network_count": 172000, "subscribers_count": 8000, "language_enc": 3},
    {"name": "microsoft/vscode",       "forks": 29000,  "watchers": 29000,  "open_issues": 6000, "size_kb": 380000,  "has_wiki": 0, "has_projects": 1, "has_downloads": 1, "is_fork": 0, "age_days": 3200,  "days_since_push": 1, "topics_count": 10, "network_count": 29000,  "subscribers_count": 3400, "language_enc": 7},
    {"name": "tensorflow/tensorflow",  "forks": 88000,  "watchers": 88000,  "open_issues": 2000, "size_kb": 640000,  "has_wiki": 1, "has_projects": 0, "has_downloads": 1, "is_fork": 0, "age_days": 2900,  "days_since_push": 1, "topics_count": 8,  "network_count": 88000,  "subscribers_count": 7100, "language_enc": 2},
    {"name": "vuejs/vue",              "forks": 35000,  "watchers": 35000,  "open_issues": 500,  "size_kb": 30000,   "has_wiki": 0, "has_projects": 0, "has_downloads": 1, "is_fork": 0, "age_days": 3600,  "days_since_push": 2, "topics_count": 6,  "network_count": 35000,  "subscribers_count": 4200, "language_enc": 7},
]

results = []
for repo in repos:
    fv = np.array([repo[c] for c in FEATURE_COLS]).reshape(1, -1)
    predicted = int(np.expm1(model.predict(fv)[0]))
    results.append({"name": repo["name"], "predicted_stars": predicted})

for r in sorted(results, key=lambda x: x["predicted_stars"], reverse=True):
    print(f"{r['predicted_stars']:>10,}  {r['name']}")
```

---

## Phase 7 — Scalability Analysis

Scale Celery workers on Prod and measure throughput before and after.

```bash
# On Prod VM — check current worker count
cd /data-engineering-II-project/ci_cd/production_server
docker compose ps

# Baseline: 1 worker — run a batch of tasks and time them
# Then scale up:
docker compose up -d --scale worker=4

# Run the same batch and compare
```

For load testing from the Client VM, install `apache2-utils` and use `ab`, or write a simple Python script that submits N tasks via Celery and measures total time.

Record requests/sec and mean latency at 1 worker vs 4 workers — that table goes in the report.

---

## Phase 8 — Report Checklist

The report must be **4 pages** with these sections:

| Section | What to include |
|---|---|
| Introduction | Problem statement, why GitHub stars matter, your approach |
| Related work | 2–3 papers on GitHub repo popularity prediction |
| System architecture | Diagrams: data pipeline, CI/CD pipeline, prod serving stack |
| Results — model comparison | Table of all models with R² scores; which won and why |
| Results — scalability | Table: workers=1 vs workers=4, requests/sec and avg latency |

---

## Quick-reference command sequence

```bash
# ── Client VM ───────────────────────────────────────────────────────────────
cd data-engineering-II-project/openstack-client/
source UPPMAX_2026_1-24_openrc.sh
python3 start_dev_prod_instances.py        # wait 2-3 min
ansible-playbook -i inventory.ini ansible_configuration.yml

# ── Dev VM ──────────────────────────────────────────────────────────────────
ssh appuser@<DEV_IP>
cd /data-engineering-II-project
python3 data_collection/collect_github.py --token <TOKEN> --total 1000
python3 training/train_models.py --data data.csv --model-dir ci_cd/production_server/models/
git add ci_cd/production_server/models/best_model.pkl ci_cd/production_server/models/metadata.json
git commit -m "deploy: best model"
git push deployment main

# ── Prod VM ─────────────────────────────────────────────────────────────────
cd /data-engineering-II-project/ci_cd/production_server
docker compose ps        # verify rabbit and worker are Up
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `openstack server list` fails | `openrc` not sourced | `source UPPMAX_2026_1-24_openrc.sh` |
| `ansible-inventory` fails | `inventory.ini` not generated | Run `start_dev_prod_instances.py` first |
| Ansible connection errors | VMs still initializing | Wait 2–3 min after provisioning, then retry |
| SSH Dev→Prod fails | Public key not in Prod's `authorized_keys` | See Phase 3.1 |
| Celery task stays `PENDING` | RabbitMQ not running | `docker compose ps` on Prod; `docker compose restart rabbit` |
| `best_model.pkl` not found after deploy | SCP path mismatch in hook | Check paths in `/opt/model_repo.git/hooks/post-receive` |
| GitHub API 403 | Token missing or expired | Generate new PAT: GitHub → Settings → Developer settings → Personal access tokens |
| GitHub API 422 | Search result set too large | Raise `--min-stars` to 500 |

# Longest6 Evaluation Guide (CARLA 0.9.16 + RTX 5090)

This guide walks through **every step** needed to evaluate TransFuser (or your CrossViT / `crossvit_fusion` model) on the **Longest6** benchmark using an Ubuntu server with an **RTX 50-series GPU**.

The code in this repo avoids **mmcv/mmdet** during evaluation (`model_eval.py` + `submission_agent.py`), which is the main blocker on RTX 5090.

---

## What you will get at the end

After following this guide you will have:

1. CARLA 0.9.16 running on your GPU server
2. A Python environment that works on RTX 5090
3. Longest6 evaluation output JSON
4. Printed metrics:
   - **Driving Score** (`score_composed`)
   - **Route Completion** (`score_route`)
   - **Penalty multiplier** (`score_penalty`)
   - **Infraction counts** (collisions, red lights, route deviation, etc.)

---

## Hardware / software requirements

| Item | Requirement |
|------|-------------|
| OS | Ubuntu 22.04 or 24.04 |
| GPU | RTX 5090 (or other 50-series) |
| RAM | 32 GB+ recommended |
| Disk | ~30 GB free (CARLA + maps + models) |
| CARLA | 0.9.16 |
| Python | 3.10 or 3.11 |

---

## Part A — Connect to your Vast.ai instance

### Step A1 — Connect

From Windows, use either:

- **Vast.ai Web Terminal** (easiest), or
- **SSH** from PowerShell / Windows Terminal:

```powershell
ssh -p <PORT> root@<IP_ADDRESS>
```

### Step A2 — Confirm Linux shell

You should see a prompt like:

```text
root@xxxxxxxx:~#
```

If you cannot connect, stop here and fix the instance first.

### Step A3 — Check the GPU

```bash
nvidia-smi
```

Expected:

- GPU name: **RTX 5090** (or similar 50-series)
- Driver version shown
- CUDA version shown

If `nvidia-smi` fails, the instance has no working NVIDIA driver.

---

## Part B — Prepare the server

### Step B1 — Update Ubuntu

```bash
apt update
apt upgrade -y
```

Wait until it finishes.

### Step B2 — Install basic tools

```bash
apt install -y git wget unzip curl tmux htop python3 python3-venv python3-pip ffmpeg
```

### Step B3 — Create workspace

```bash
mkdir -p ~/workspace
cd ~/workspace
pwd
```

Expected output:

```text
/root/workspace
```

---

## Part C — Install CARLA 0.9.16

### Step C1 — Download CARLA (manual method)

Open in your browser:

https://github.com/carla-simulator/carla/releases/tag/0.9.16

Download:

- `CARLA_0.9.16.tar.gz`
- `AdditionalMaps_0.9.16.tar.gz`

### Step C2 — Download on the server

```bash
cd ~/workspace
wget "https://tiny.carla.org/carla-0-9-16-linux" -O CARLA_0.9.16.tar.gz
wget "https://tiny.carla.org/additional-maps-0-9-16-linux" -O AdditionalMaps_0.9.16.tar.gz
```

### Step C3 — Extract CARLA

```bash
tar -xzf CARLA_0.9.16.tar.gz
ls
```

You should see `CARLA_0.9.16/`.

### Step C4 — Import additional maps

```bash
mkdir -p CARLA_0.9.16/Import
tar -xzf AdditionalMaps_0.9.16.tar.gz -C CARLA_0.9.16/Import
cd CARLA_0.9.16
chmod +x ImportAssets.sh CarlaUE4.sh
./ImportAssets.sh
```

Wait until import completes.

### Step C5 — Verify CARLA folder

```bash
cd ~/workspace/CARLA_0.9.16
ls
```

You should see:

- `CarlaUE4.sh`
- `PythonAPI/`
- `Import/`

### Automated alternative

From the project root:

```bash
chmod +x scripts/setup_carla_0.9.16.sh
./scripts/setup_carla_0.9.16.sh ~/workspace/CARLA_0.9.16
```

---

## Part D — Upload / clone the evaluation code

### Step D1 — Upload `transfuser-2022`

Upload the project folder to the server, for example:

```text
~/workspace/transfuser-2022/
```

If using git:

```bash
cd ~/workspace
# upload your zip, then:
unzip transfuser-2022.zip
cd transfuser-2022
```

### Step D2 — Confirm project structure

```bash
cd ~/workspace/transfuser-2022
ls
```

You should see:

- `team_code_transfuser/`
- `leaderboard/`
- `scenario_runner/`
- `scripts/`
- `model_ckpt/`

---

## Part E — Download model weights

You can evaluate **either** the official TransFuser model **or** your CrossViT model.

### Option 1 — Official TransFuser model

```bash
cd ~/workspace/transfuser-2022
mkdir -p model_ckpt/official_transfuser
wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/models_2022.zip -O /tmp/models_2022.zip
unzip /tmp/models_2022.zip -d /tmp/models_2022
# Copy the transfuser checkpoint folder contents:
cp /tmp/models_2022/transfuser/model.pth model_ckpt/official_transfuser/
# args.txt is already provided in model_ckpt/official_transfuser/
```

### Option 2 — CrossViT / bi-attenfusion model (`model_50.pth`)

1. Download `transfuser-2022.zip` from Google Drive:
   https://drive.google.com/drive/folders/1yG9LbVtSLaneHKlB5GL5Vhzz5Miue5Me

2. Copy the checkpoint:

```bash
cp /path/to/bi-attenfusion/model_50.pth ~/workspace/transfuser-2022/model_ckpt/crossvit_fusion/model.pth
```

The `args.txt` in `model_ckpt/crossvit_fusion/` is already configured for `crossvit_fusion`.

### Step E3 — Verify model folder

```bash
ls model_ckpt/crossvit_fusion/
# expected: args.txt  model.pth

ls model_ckpt/official_transfuser/
# expected: args.txt  model.pth
```

To switch models later, only change the model directory passed to the evaluation script.

---

## Part F — Create Python environment (RTX 5090, no mmcv)

### Step F1 — Run setup script

```bash
cd ~/workspace/transfuser-2022
chmod +x scripts/*.sh
export CARLA_ROOT=~/workspace/CARLA_0.9.16
./scripts/setup_environment.sh
```

This will:

1. Create `.venv_eval/`
2. Install PyTorch with **CUDA 12.8** (`cu128`) — required for RTX 50-series (Blackwell / sm_120)
3. Install evaluation dependencies from `requirements_eval.txt`
4. Install the CARLA 0.9.16 Python wheel
5. Run `scripts/verify_environment.py`

> **Why cu128?** RTX 5090 is compute capability **sm_120**. Only PyTorch built with
> **CUDA 12.8** (PyTorch ≥ 2.7) ships sm_120 kernels. `cu124`/`cu126` wheels will fail with
> `CUDA error: no kernel image is available for execution on the device`.
>
> If the stable wheel isn't available for your Python, use the nightly:
> ```bash
> TORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/cu128 ./scripts/setup_environment.sh
> ```

### Step F2 — Activate environment

```bash
source ~/workspace/transfuser-2022/.venv_eval/bin/activate
```

### Step F3 — Verify GPU + imports

```bash
cd ~/workspace/transfuser-2022
python scripts/verify_environment.py --carla-root ~/workspace/CARLA_0.9.16
```

All lines should show `[OK]`.

If CARLA import fails:

```bash
pip install ~/workspace/CARLA_0.9.16/PythonAPI/carla/dist/carla-*.whl
```

---

## Part G — Start CARLA

Use **tmux** so CARLA keeps running in the background.

### Step G1 — Start tmux session

```bash
tmux new -s carla
```

### Step G2 — Launch CARLA headless

```bash
export CARLA_ROOT=~/workspace/CARLA_0.9.16
~/workspace/transfuser-2022/scripts/start_carla.sh
```

Wait until you see the simulator ready message (first launch can take 1–3 minutes).

### Step G3 — Detach tmux

Press:

```text
Ctrl+B, then D
```

### Step G4 — Confirm CARLA is listening

In another terminal:

```bash
ss -tlnp | grep 2000
```

You should see port `2000` in use.

---

## Part H — Run evaluation

Always activate the venv first:

```bash
source ~/workspace/transfuser-2022/.venv_eval/bin/activate
cd ~/workspace/transfuser-2022
export CARLA_ROOT=~/workspace/CARLA_0.9.16
```

### Step H1 — Smoke test (1 route, ~5–15 minutes)

```bash
./scripts/run_single_route_eval.sh model_ckpt/crossvit_fusion 0
```

If this works, proceed to the full benchmark.

### Step H2 — Full Longest6 evaluation (36 routes)

**CrossViT model:**

```bash
./scripts/run_longest6_eval.sh model_ckpt/crossvit_fusion
```

**Official TransFuser model:**

```bash
./scripts/run_longest6_eval.sh model_ckpt/official_transfuser
```

This can take **several hours** depending on GPU/server speed.

Results are saved to:

```text
results/longest6_results.json
```

If interrupted, re-run the same command; `RESUME=1` continues from the last completed route.

---

## Part I — Read evaluation metrics

### Step I1 — Print human-readable metrics

```bash
python scripts/print_metrics.py --file results/longest6_results.json
```

### Step I2 — Understand the main numbers

| Metric | Meaning |
|--------|---------|
| `score_route` | Route completion (0–100) |
| `score_penalty` | Multiplicative penalty from infractions (0–1) |
| `score_composed` | **Driving Score** = `score_route * score_penalty` |
| `collisions_vehicle` | Collisions with vehicles |
| `collisions_pedestrian` | Collisions with pedestrians |
| `collisions_layout` | Collisions with static objects |
| `red_light` | Red light infractions |
| `stop_infraction` | Stop sign infractions |
| `outside_route_lanes` | Lane departure events |
| `route_dev` | Route deviation failures |
| `vehicle_blocked` | Agent blocked and unable to proceed |

### Step I3 — Optional detailed CSV / maps

```bash
python tools/result_parser.py \
  --xml leaderboard/data/longest6/longest6.xml \
  --results results/ \
  --save_dir results/parsed/ \
  --town_maps leaderboard/data/town_maps_xodr
```

This creates `results/parsed/results.csv` and infraction map images.

---

## Part J — Record a demonstration video

Because CARLA runs headless, record the **terminal output** and/or use a desktop session.

### Method 1 — Record terminal with `asciinema` (simple)

```bash
apt install -y asciinema
asciinema rec longest6_demo.cast
# run your evaluation commands
exit
```

### Method 2 — Record desktop with ffmpeg

In one terminal, start recording:

```bash
./scripts/record_demo.sh demo_longest6.mp4
```

In another terminal, run the smoke test or full evaluation.

Stop ffmpeg with `Ctrl+C` when finished.

### What to show in the demo video

1. `nvidia-smi` showing RTX 5090
2. `python scripts/verify_environment.py` success
3. Evaluation command running
4. Final `print_metrics.py` output with Driving Score and infractions

---

## Part K — Switch between different `model.pth` files

The evaluation agent loads **every `.pth` file** in the model directory and ensembles them.

### Use a different single model

```bash
cp /path/to/new_model.pth model_ckpt/crossvit_fusion/model.pth
./scripts/run_longest6_eval.sh model_ckpt/crossvit_fusion
```

### Use a custom folder

```bash
mkdir -p model_ckpt/my_experiment
cp /path/to/model.pth model_ckpt/my_experiment/
cp model_ckpt/crossvit_fusion/args.txt model_ckpt/my_experiment/
./scripts/run_longest6_eval.sh model_ckpt/my_experiment
```

Edit `args.txt` if the backbone or architecture changed.

### Important note about DDP checkpoints

`submission_agent.py` strips a `module.` prefix from checkpoint keys (for multi-GPU training).

- If loading fails with missing/unexpected keys, comment out line 97 in `submission_agent.py`.
- If loading fails because keys are missing the prefix, keep line 97 enabled.

---

## Part L — Troubleshooting

### Problem: `No module named 'crossvit_fusion'`

**Fix:** Ensure `team_code_transfuser/crossvit_fusion.py` exists (copied from `Bi-Attenfusion.py`).

### Problem: mmcv / CUDA architecture error on RTX 5090

**Fix:** Use evaluation only. Do **not** import `model.py` for inference. This repo uses `model_eval.py` instead.

### Problem: `Agent couldn't be set up`

Check:

```bash
source .venv_eval/bin/activate
python scripts/verify_environment.py
ls model_ckpt/crossvit_fusion/
```

### Problem: CARLA connection timeout

1. Confirm CARLA is running in tmux
2. Check port 2000
3. Restart CARLA and retry

### Problem: CUDA out of memory

Start CARLA with lower quality:

```bash
CARLA_QUALITY=Epic ./scripts/start_carla.sh
# or edit start_carla.sh and use Low
```

Only one evaluation process should use the GPU at a time.

### Problem: Evaluation is very slow

Longest6 has **36 routes**. A full run is expected to take hours. Use the single-route smoke test first.

---

## Quick reference (copy/paste)

```bash
# 1. Server prep
apt update && apt upgrade -y
apt install -y git wget unzip curl tmux htop python3 python3-venv python3-pip ffmpeg

# 2. CARLA
cd ~/workspace
./transfuser-2022/scripts/setup_carla_0.9.16.sh ~/workspace/CARLA_0.9.16

# 3. Python env
cd ~/workspace/transfuser-2022
export CARLA_ROOT=~/workspace/CARLA_0.9.16
./scripts/setup_environment.sh
source .venv_eval/bin/activate

# 4. Start CARLA (tmux)
tmux new -s carla
export CARLA_ROOT=~/workspace/CARLA_0.9.16
./scripts/start_carla.sh
# Ctrl+B, D

# 5. Evaluate
source ~/workspace/transfuser-2022/.venv_eval/bin/activate
cd ~/workspace/transfuser-2022
./scripts/run_longest6_eval.sh model_ckpt/crossvit_fusion

# 6. Metrics
python scripts/print_metrics.py --file results/longest6_results.json
```

---

## File map

| Path | Purpose |
|------|---------|
| `team_code_transfuser/model_eval.py` | Inference model without mmcv |
| `team_code_transfuser/submission_agent.py` | CARLA leaderboard agent |
| `team_code_transfuser/crossvit_fusion.py` | CrossViT backbone |
| `scripts/setup_environment.sh` | RTX 5090 Python setup |
| `scripts/setup_carla_0.9.16.sh` | CARLA + maps installer |
| `scripts/start_carla.sh` | Start simulator |
| `scripts/run_longest6_eval.sh` | Full Longest6 evaluation |
| `scripts/run_single_route_eval.sh` | One-route smoke test |
| `scripts/print_metrics.py` | Print Driving Score + infractions |
| `model_ckpt/*/args.txt` | Model configuration |

---

## References

- TransFuser repo: https://github.com/autonomousvision/transfuser
- CARLA 0.9.16 release: https://github.com/carla-simulator/carla/releases/tag/0.9.16
- Longest6 routes: `leaderboard/data/longest6/longest6.xml`

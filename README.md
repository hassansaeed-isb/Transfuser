# CARLA TransFuser — Longest6 Evaluation (RTX 5090)

Evaluate autonomous driving on the **Longest6** benchmark with **CARLA 0.9.16** on **RTX 50-series GPUs**.

## Start here

Read the full step-by-step guide:

**[LONGEST6_EVALUATION_GUIDE.md](LONGEST6_EVALUATION_GUIDE.md)**

## Quick start (on Ubuntu server)

```bash
cd transfuser-2022
chmod +x scripts/*.sh
export CARLA_ROOT=~/workspace/CARLA_0.9.16

./scripts/setup_environment.sh
source .venv_eval/bin/activate

# Terminal 1 — CARLA
tmux new -s carla
./scripts/start_carla.sh

# Terminal 2 — evaluate
./scripts/run_longest6_eval.sh model_ckpt/crossvit_fusion
python scripts/print_metrics.py --file results/longest6_results.json
```

## Switch models

```bash
./scripts/run_longest6_eval.sh model_ckpt/official_transfuser
./scripts/run_longest6_eval.sh model_ckpt/crossvit_fusion
```

Place `model.pth` + `args.txt` in each `model_ckpt/<name>/` folder.

## Key fix for RTX 5090

Evaluation uses `team_code_transfuser/model_eval.py` (no mmcv/mmdet). The official `model.py` still depends on mmcv and is only needed for training.

## Deliverables in this repo

| Deliverable | Location |
|-------------|----------|
| User guide | `LONGEST6_EVALUATION_GUIDE.md` |
| Runnable code | `transfuser-2022/` |
| Evaluation scripts | `transfuser-2022/scripts/` |
| Demo recording helper | `transfuser-2022/scripts/record_demo.sh` |

Record the demo on your Vast.ai instance while evaluation runs; see Part J in the guide.

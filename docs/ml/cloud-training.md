# EarlyCare Cloud Training Notes

This note covers when to use cloud GPUs for EarlyCare speech ML. The live app must keep working without cloud compute, model weights, or GPU access.

## Decision

GMI Cloud can be useful for the heavy raw-audio research stage, but it is not needed for the core EarlyCare MVP.

Use cloud GPUs for:

- WavLM, wav2vec2, or MERaLiON embedding extraction over approved raw-audio datasets.
- Larger repeated experiments after a manifest has been reviewed locally.
- Faster offline evaluation when raw-audio datasets become too large for a laptop.

Do not use cloud GPUs for:

- The scheduled check-in workflow.
- UCI feature-table baselines.
- Concussion routing.
- Live FastAPI request handling.
- Any dataset whose access terms do not allow cloud processing.

## GMI Cloud Fit

GMI Cloud's public docs describe GPU Compute options for managed GPU clusters, container workloads, and bare-metal servers. That maps well to this repo because the heavy encoder modes are isolated in `research/speech_ml/extract_embeddings.py` and are not app runtime dependencies.

The pricing page listed dedicated NVIDIA GPU infrastructure, including H100, H200, B200, and GB200-class options when reviewed on 2026-07-05. Treat prices and availability as live values and verify them in the GMI console before running jobs.

Useful links:

- GMI GPU Compute docs: <https://docs.gmicloud.ai/cluster-engine>
- GMI pricing: <https://www.gmicloud.ai/en/pricing>
- GMI docs home: <https://docs.gmicloud.ai/>

## Safe Workflow

Run the local planning step first:

```bash
python3 research/speech_ml/run_training_cycle.py \
  --fetch-supported \
  --run-ready \
  --audit \
  --dry-run
```

Use GMI only after the local report shows which dataset is ready and what command should run.

On a GMI GPU host or container:

```bash
git clone https://github.com/dphyy/EarlyCare.git
cd EarlyCare
git checkout dev

python3 -m venv .venv-research
. .venv-research/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r backend/requirements.txt
python3 -m pip install torch transformers soundfile
```

Copy approved datasets into ignored folders only:

```bash
mkdir -p research/datasets research/artifacts
```

Prepare or copy a reviewed manifest, then run a heavy raw-audio experiment:

```bash
python3 research/speech_ml/run_experiment.py \
  --manifest research/datasets/neurovoz_manifest.csv \
  --audio-root research/datasets/NeuroVoz \
  --output-dir research/artifacts \
  --experiment-name neurovoz-wavlm \
  --model wavlm \
  --device cuda
```

For MERaLiON, use the repo flag only if the model loader requires remote code:

```bash
python3 research/speech_ml/run_experiment.py \
  --manifest research/datasets/neurovoz_manifest.csv \
  --audio-root research/datasets/NeuroVoz \
  --output-dir research/artifacts \
  --experiment-name neurovoz-meralion \
  --model meralion \
  --device cuda \
  --trust-remote-code
```

Audit before any app handoff:

```bash
python3 research/speech_ml/audit_model_artifacts.py \
  --artifacts-dir research/artifacts
```

Create an app payload only after audit:

```bash
python3 research/speech_ml/make_experiment_payload.py \
  --artifacts-dir research/artifacts \
  --experiment neurovoz-wavlm \
  --output research/artifacts/neurovoz-wavlm_payload.json \
  --speaker-id s-001
```

This payload defaults to `offline embedding`. Do not use `--runtime-mode "validated model"` unless the model-card gate and artifact audit are complete.

## Data Handling Rules

- Confirm dataset terms before uploading raw voice data to any cloud.
- Do not commit raw audio, embeddings, model artifacts, reports, or subject exports.
- Keep cloud outputs under `research/artifacts/`.
- Pull back only the artifacts needed for review.
- Delete cloud copies if the dataset terms or project policy require it.
- Never put API keys or subject data into commands, commit messages, docs, or screenshots.

## Cost Control

- Use `--dry-run` locally first.
- Run `--limit` smoke tests before full extraction.
- Start with one GPU and one dataset.
- Prefer a short-lived container or instance for a single extraction run.
- Stop the GPU immediately after artifacts are copied back and audited.

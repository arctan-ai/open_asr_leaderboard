# Open ASR Leaderboard

This repository contains the code for the Open ASR Leaderboard. The leaderboard is a Gradio Space that allows users to compare the accuracy of ASR models on a variety of datasets. The leaderboard is hosted at [hf-audio/open_asr_leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).

# Datasets

The Open ASR Leaderboard evaluates models on a diverse set of publicly available ASR benchmarks hosted on the Hugging Face Hub. These datasets cover a wide range of domains, languages, and recording conditions to provide a fair and comprehensive comparison across models.

* **Main Test Sets (English, short-form):**
  The main benchmark datasets used for evaluation (short-form English) are available [here](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard).

* **English, long-form:**
  The [**ASR Longform benchmark**](https://huggingface.co/datasets/hf-audio/asr-leaderboard-longform) dataset includes earnings21 and earnings22. We also evaluate on [CORAAL](https://huggingface.co/datasets/bezzam/coraal), but it is stored as a separate dataset since it has multiple splits.

* **Multilingual Benchmark:**
  The [**ASR Multilingual benchmark**](https://huggingface.co/datasets/nithinraok/asr-leaderboard-datasets) dataset includes fleurs, mcv and mls multilingual.


* **Private datasets:** 
  After submitting a model to the leaderboard, the maintainers will evaluate on private sets, as described [here](https://huggingface.co/blog/open-asr-leaderboard-private-data).


# Evaluate a model (as of 24 June 2026)

English short-form evaluations use [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs-overview) to guarantee reproducibility: every run executes a Docker image on the same hardware, to minimize environment and driver differences. Multilingual and long-form evaluations will migrate to HF Jobs in the future.

Jobs are launched on the following hardware ([flavor](https://huggingface.co/docs/hub/jobs-configuration#hardware-flavor) in HF Jobs terminology):
```
name             pretty name             cpu       ram      storage   accelerator               cost/min  cost/hour
h200             Nvidia H200             23 vCPU   256 GB   3000 GB   1x H200 (141 GB)          $0.0833   $5.00
```
Example costs for a full run over the main public datasets:
- $2.92 for `nvidia/parakeet-tdt-0.6b-v3`
- $4.75 for `openai/whisper-large-v3-turbo`
- $5.58 for `Qwen/Qwen3-ASR-1.7B`

Each model family has its own Docker image with the necessaru software requirements. The evalulation configurations are hosted as [HF Spaces](https://huggingface.co/collections/hf-audio/open-asr-leaderboard-eval-configurations).

**To launch an evaluation:**

1. **Hugging Face Hub setup**
   - Create an account at https://huggingface.co/ and add credits for HF Jobs: https://huggingface.co/settings/billing
   - Create a [WRITE token](https://huggingface.co/settings/tokens/new?tokenType=write) and copy it.
   - Create a Storage Bucket to store results: https://huggingface.co/new-bucket

2. **One-time local setup**

A local setup is needed to launch the evaluation and score with the repo's normalizer.
```bash
# Clone the repository
git clone git@github.com:huggingface/open_asr_leaderboard.git
cd open_asr_leaderboard

# Create a minimal conda environment (no GPU required locally)
conda create -n leaderboard_jobs python=3.10 -y
conda activate leaderboard_jobs
pip install -r requirements/requirements_jobs.txt
huggingface-cli login   # paste your WRITE token when prompted
```

3. **Launch an evaluation** 🚀
```bash
# Open the relevant submit_jobs script, uncomment the models/datasets you want, then run:
RESULTS_BUCKET="<your-bucket>" HF_TOKEN=hf_... bash qwen/submit_jobs.sh

# Jobs are submitted in parallel (one per dataset). The script waits for all
# jobs to finish, syncs results from the bucket, and prints a CSV summary.
# If SLACK_BOT_TOKEN and SLACK_CHANNEL_ID are set, the final scoring step also
# posts aggregate WER/RTFx and run metadata to Slack.

# Billing to org
ORG_NAME="<org-name>" RESULTS_BUCKET="<your-bucket>" HF_TOKEN=hf_... bash qwen/submit_jobs.sh
```

## Local evaluation

For contributors who want to test locally or evaluate multilingual/long-form models before HF Jobs support is added, the `requirements/` folder contains per-family dependency files. The Dockerfiles in the HF Spaces can also be used to build a local container.

Each model family has a `run_eval.py` entry point driven by a corresponding bash script (e.g. `run_whisper.sh`). The script outputs a JSONL file with predictions and prints WER and RTFx after completion. See the sub-folders of this repo for examples; the latest scripts are in the HF Spaces linked above.

When a script calls `normalizer.eval_utils.score_results`, Slack notification is automatic if `SLACK_BOT_TOKEN` and `SLACK_CHANNEL_ID` are present in the environment. Slack failures are reported as warnings and do not fail the evaluation.

### Optional audio preprocessing

English short-form runners that use the shared `normalizer.data_utils.prepare_data` path can run eval-time audio preprocessing before ASR:

```bash
python transformers/run_eval.py \
    --model_id=openai/whisper-large-v3-turbo \
    --dataset_path=hf-audio/open-asr-leaderboard \
    --dataset=librispeech \
    --split=test.clean \
    --device=0 \
    --batch_size=1 \
    --max_eval_samples=2 \
    --audio_preprocessor=arctan
```

Available preprocessors:

- `arctan`: requires the optional `arctan-vi` package and a valid `ARCTAN_SDK_KEY` in the eval environment or `.env` file.
- `ai_coustics_vfl_2_1`: runs the ai-coustics VFL 2.1 path through the `livekit-examples/noise-canceller` CLI used by [`ai-coustics/lk-noise-canceller-examples`](https://github.com/ai-coustics/lk-noise-canceller-examples). Clone that repo with submodules, set `AI_COUSTICS_NOISE_CANCELLER_DIR` to its `noise-canceller` directory, and set `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET`. The leaderboard wrapper uses filter `aic-quail-vfl`, enhancement level `1.0`, 16 kHz output, a 2-second leading-silence pad, and the CLI's ai-coustics `--direct` mode.

Optional Silero VAD is enabled with `--vad_position=pre` or `--vad_position=post`. `pre` applies VAD before the selected audio preprocessor; `post` applies it after preprocessing and before ASR. Without a preprocessor, both positions are equivalent. VAD preserves clip duration by replacing non-speech samples with zeros and uses a fixed threshold of `0.45`, minimum silence of `100 ms`, speech padding of `150 ms`, and Silero's default minimum speech duration of `250 ms`. VAD requires local 8 kHz or 16 kHz audio and cannot be combined with API `--use_url` mode.

The `.env` file is loaded with `python-dotenv` when audio preprocessing is enabled. The default is `--audio_preprocessor=none`, which preserves the existing evaluation path. API URL mode, multilingual scripts, Nemo scripts, and long-form scripts are not wired for this MVP path. Reported RTFx still measures ASR inference time only; preprocessing runs before the timed transcription section.

# Trade-off plots

For open-source models, you can plot tradeoff plots like below with `scripts/plot_all.sh`.

![EN Shortform RTFx vs WER](scripts/data/en_shortform_rtfx_wer.png)

You can highlight a particular model (see `scripts/data` for CSV results as of 26 March 2026):
```
./scripts/plot_all.sh --highlight "model_name"

# for example
./scripts/plot_all.sh --highlight "nvidia/parakeet-tdt-0.6b-v3"
```

![Highlight model](scripts/data/nvidia_parakeet-tdt-0.6b-v3_en_shortform_rtfx_wer.png)

You can also specify your own model and its performance as such:
```
./scripts/plot_all.sh --custom-model "MY MODEL" --model-size 2.0 --en-shortform-wer 5.5 --en-shortform-rtfx 1000
```

![Custom model](scripts/data/MY_MODEL_en_shortform_rtfx_wer.png)

# Contributing a model or dataset

Please follow the [pull request template](./.github/PULL_REQUEST_TEMPLATE.md); it contains a submission checklist and guidelines.

# Citation 


```bibtex
@misc{srivastav2025openasrleaderboardreproducible,
      title={Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual and Long-Form Speech Recognition Evaluation}, 
      author={Vaibhav Srivastav and Steven Zheng and Eric Bezzam and Eustache Le Bihan and Nithin Koluguri and Piotr Żelasko and Somshubra Majumdar and Adel Moumen and Sanchit Gandhi},
      year={2025},
      eprint={2510.06961},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.06961}, 
}
```

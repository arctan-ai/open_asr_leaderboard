#!/bin/bash

# Multilingual API ASR Evaluation Script
# Evaluates on FLEURS, MCV (Mozilla Common Voice), and MLS (Multilingual LibriSpeech)

export PYTHONPATH="..":$PYTHONPATH

export OPENAI_API_KEY="your_api_key"
export ASSEMBLYAI_API_KEY="your_api_key"
export ELEVENLABS_API_KEY="your_api_key"
export REVAI_API_KEY="your_api_key"
export AQUAVOICE_API_KEY="your_api_key"
export SPEECHMATICS_API_KEY="your_api_key"
export RESON8_API_KEY="your_api_key"
export AZURE_API_KEY="your_api_key"
export SONIOX_API_KEY="your_api_key"
export DEEPGRAM_API_KEY="your_api_key"

# Configuration
# Streaming examples, run directly when needed:
# python run_eval_ml.py --dataset_path="$DATASET_PATH" --config_name=fleurs_de --language=de --split=test --model_name deepgram/nova-3 --max_workers=16 --streaming
# python run_eval_ml.py --dataset_path="$DATASET_PATH" --config_name=fleurs_de --language=de --split=test --model_name soniox/stt-async-v5 --max_workers=16 --streaming
# python run_eval_ml.py --dataset_path="$DATASET_PATH" --config_name=fleurs_de --language=de --split=test --model_name assembly/universal-3-pro --max_workers=4 --streaming
MODEL_IDs=(
    # "openai/gpt-4o-transcribe"
    # "openai/gpt-4o-mini-transcribe"
    # "openai/whisper-1"
    # "assembly/universal-3-pro"
    # "elevenlabs/scribe_v2"
    # "speechmatics/enhanced"
    # "soniox/stt-async-v5"
    # "deepgram/nova-3"
    "reson8/resonant-1"
    "reson8/resonant-1-flash"
    "microsoft/azure-speech-05-2026"
)

MAX_WORKERS=20
DATASET_PATH="nithinraok/asr-leaderboard-datasets"

# German, French, Italian, Spanish, Portuguese
DATASET_NAMES=("fleurs" "mls" "mcv")
DATASET_LANGS_fleurs="de fr it es pt"
DATASET_LANGS_mcv="de es fr it"
DATASET_LANGS_mls="es fr it pt"

# Datasets that require lexical format prompt (azure only)
LEXICAL_DATASETS="mls-it"

# Function to run evaluation
run_evaluation() {
    local model_id=$1
    local dataset=$2
    local language=$3
    local config_name="${dataset}_${language}"

    # Build prompt args for azure + lexical datasets
    local prompt_args=()
    if [[ "$model_id" == microsoft/* ]] && [[ " $LEXICAL_DATASETS " == *" ${dataset}-${language} "* ]]; then
        prompt_args=(--prompt "Output must be in lexical format.")
    fi

    echo ""
    echo "Running evaluation: $config_name"
    echo "   Model: $model_id"
    echo "   Dataset: $dataset"
    echo "   Language: $language"
    echo "   Time: $(date)"
    echo "----------------------------------------"

    if ! python run_eval_ml.py \
        --dataset_path="$DATASET_PATH" \
        --config_name="$config_name" \
        --language="$language" \
        --split="test" \
        --model_name="$model_id" \
        --max_workers="$MAX_WORKERS" \
        "${prompt_args[@]}"; then
        echo "Evaluation failed for $config_name; stopping batch."
        return 1
    fi

    echo "Evaluation completed successfully for $config_name"

    echo "----------------------------------------"
    return 0
}

# Main execution
echo "========================================================"
echo "Starting Multilingual API ASR Evaluation"
echo "========================================================"

# Run evaluations for all models
for MODEL_ID in "${MODEL_IDs[@]}"; do
    echo ""
    echo "Processing Model: $MODEL_ID"
    echo "========================================================"

    # Run evaluations for all datasets and languages
    for dataset in "${DATASET_NAMES[@]}"; do
        varname="DATASET_LANGS_${dataset}"
        languages="${!varname}"

        if [[ -n "$languages" ]]; then
            echo ""
            echo "Processing dataset: $dataset"
            echo "   Languages: $languages"
            echo ""

            for language in $languages; do
                if ! run_evaluation "$MODEL_ID" "$dataset" "$language"; then
                    echo "Stopping multilingual batch after evaluation failure."
                    exit 1
                fi
            done
        fi
    done

    echo ""
    echo "========================================================"
    echo "Evaluating results for $MODEL_ID"
    echo "========================================================"

    # Evaluate results
    RUNDIR=`pwd`
    cd ../normalizer
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}', multilingual=True)"
    cd "$RUNDIR"

    echo ""
done

echo ""
echo "========================================================"
echo "All evaluations completed!"
echo "========================================================"

#!/usr/bin/env bash
set -euo pipefail

source scripts/sealed_evaluation_defaults.sh

RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
CHECKPOINT_ARGS="${CHECKPOINT_ARGS:?CHECKPOINT_ARGS is required}"
CHECKPOINT="${CHECKPOINT:-$RUN_ROOT/E3_final_multimodal_top2/checkpoint_final.pt}"
OUTPUT="${OUTPUT:-outputs/review_repair/sealed_evaluation_protocol.json}"
SEALED_MANIFEST="${SEALED_MANIFEST:-data/sealed_eval_v1/sealed_eval_manifest.json}"
IMAGE_MANIFEST="${IMAGE_MANIFEST:-data/sealed_eval_v1/image_test.jsonl}"
SPEECH_MANIFEST="${SPEECH_MANIFEST:-data/sealed_eval_v1/speech_test.jsonl}"
PROJECT="${PROJECT:?PROJECT is required to bind the frozen Run:AI protocol}"

for required in \
  "$CHECKPOINT" \
  "$CHECKPOINT_ARGS" "$SEALED_MANIFEST" "$IMAGE_MANIFEST" "$SPEECH_MANIFEST"; do
  if [ ! -s "$required" ]; then
    echo "missing protocol-freeze input: $required" >&2
    exit 2
  fi
done

python scripts/freeze_evaluation_protocol.py \
  --selected-root "$RUN_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-args-json "@$CHECKPOINT_ARGS" \
  --sealed-manifest "$SEALED_MANIFEST" \
  --image-test "$IMAGE_MANIFEST" \
  --speech-test "$SPEECH_MANIFEST" \
  --evaluator-script scripts/eval_conditional_retrieval.py \
  --evaluator-script scripts/sealed_position_allocator.py \
  --evaluator-script scripts/eval_representation_funnel.py \
  --evaluator-script scripts/analyze_specialization_and_quality.py \
  --evaluator-script hf_sources.py \
  --evaluator-script model/fusion.py \
  --evaluator-script model/olmoe_adapter.py \
  --evaluator-script training/olmoe_required_runs.py \
  --evaluator-script training/olmoe_real_subset_runs.py \
  --evaluator-script run.sh \
  --evaluator-script scripts/submit_runai.sh \
  --evaluator-script scripts/submit_sealed_control_matrix.sh \
  --evaluator-script scripts/sealed_evaluation_defaults.sh \
  --evaluator-script scripts/submit_representation_funnel.sh \
  --paired-analysis-script scripts/analyze_paired_controls.py \
  --paired-analysis-script scripts/analyze_sealed_matrix.py \
  --paired-analysis-script scripts/freeze_evaluation_protocol.py \
  --paired-analysis-script scripts/protocol_v2.py \
  --output "$OUTPUT" \
  --runai-project "$PROJECT" \
  --candidate-seed 314159 \
  --control-seed 42 \
  --candidate-size 5 \
  --candidate-size 10 \
  --candidate-size 250 \
  --candidate-protocol sealed_evaluation_v1 \
  --hard-negative-protocol "checkpoint-independent lexical_jaccard_v1 over lowercased alphanumeric token sets within the sealed candidate pool" \
  --image-query-count 250 \
  --speech-query-count 250 \
  --max-length "${MAX_LENGTH:-512}" \
  --conditional-batch-size "$CONDITIONAL_BATCH_SIZE" \
  --evaluation-cell r5:5:random:secondary \
  --evaluation-cell r10:10:random:secondary \
  --evaluation-cell h10:10:hard_text:primary \
  --evaluation-cell f250:250:random:secondary

python scripts/freeze_evaluation_protocol.py --verify "$OUTPUT" --verify-git-state

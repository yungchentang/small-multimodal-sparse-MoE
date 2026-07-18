#!/usr/bin/env bash
set -euo pipefail

# Strict matched attribution campaign for image normalization, speech init, and KD.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
GIT=(git -c "safe.directory=$REPO_ROOT")

SOURCE_COMMIT_SHA="${SOURCE_COMMIT_SHA:?SOURCE_COMMIT_SHA is required}"
[[ "$SOURCE_COMMIT_SHA" =~ ^[0-9a-fA-F]{40}$ ]] || {
  echo "SOURCE_COMMIT_SHA must be an exact 40-character hex commit" >&2
  exit 2
}
[ "$("${GIT[@]}" rev-parse HEAD)" = "$SOURCE_COMMIT_SHA" ] || {
  echo "source commit mismatch" >&2
  exit 2
}
if ! "${GIT[@]}" diff --quiet || ! "${GIT[@]}" diff --cached --quiet ||
  [ -n "$("${GIT[@]}" ls-files --others --exclude-standard)" ]; then
  echo "MM NORM+KD campaign requires a clean source worktree" >&2
  exit 2
fi

DATA_DIR="${DATA_DIR:-data/real_subset_clean_260708b}"
DEVELOPMENT_SPLIT_MANIFEST="${DEVELOPMENT_SPLIT_MANIFEST:?DEVELOPMENT_SPLIT_MANIFEST is required}"
DEVELOPMENT_SPEECH_SOURCE_SHA256="${DEVELOPMENT_SPEECH_SOURCE_SHA256:?DEVELOPMENT_SPEECH_SOURCE_SHA256 is required}"
export DEVELOPMENT_SPEECH_SOURCE_SHA256
BASE_OUT="${BASE_OUT:-outputs/development_mm_norm_kd}"
STAMP="${STAMP:-$(date +%y%m%d%H%M)}"
SEED="${SEED:-42}"
GPU="${GPU:-1}"
ONLY_RAW="${ONLY:-C_IMAGE_NORM_ONLY,C_SPEECH_INIT_ONLY,C_DUAL,C_DUAL_KD025}"
DRY_RUN="${DRY_RUN:-0}"
FINAL_STEPS="${FINAL_STEPS:-500}"
ALIGNMENT_PRETRAIN_STEPS="${ALIGNMENT_PRETRAIN_STEPS:-400}"
MODALITY_CYCLE="${MODALITY_CYCLE:-text,image,speech}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
SPEECH_TEACHER_BANK_BATCH_SIZE="${SPEECH_TEACHER_BANK_BATCH_SIZE:-64}"
CONDITIONAL_RANKING_NEGATIVE_MODE="${CONDITIONAL_RANKING_NEGATIVE_MODE:-stride}"
IMAGE_CONDITIONAL_RANKING_COEF="${IMAGE_CONDITIONAL_RANKING_COEF:-0.5}"
IMAGE_CONTRASTIVE_COEF="${IMAGE_CONTRASTIVE_COEF:-0.0}"
[[ "$FINAL_STEPS" =~ ^[0-9]+$ ]] && [ "$FINAL_STEPS" -ge 500 ] || {
  echo "FINAL_STEPS must be an integer >= 500" >&2
  exit 2
}
[[ "$ALIGNMENT_PRETRAIN_STEPS" =~ ^[0-9]+$ ]] && [ "$ALIGNMENT_PRETRAIN_STEPS" -ge 400 ] || {
  echo "ALIGNMENT_PRETRAIN_STEPS must be an integer >= 400" >&2
  exit 2
}
[[ "$TRAIN_BATCH_SIZE" =~ ^[0-9]+$ ]] && [ "$TRAIN_BATCH_SIZE" -ge 1 ] || {
  echo "TRAIN_BATCH_SIZE must be a positive integer" >&2
  exit 2
}
[[ "$SPEECH_TEACHER_BANK_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "SPEECH_TEACHER_BANK_BATCH_SIZE must be a positive integer" >&2; exit 2;
}
case ",$MODALITY_CYCLE," in
  *,text,* ) ;;
  * ) echo "MODALITY_CYCLE must include text" >&2; exit 2 ;;
esac
case ",$MODALITY_CYCLE," in
  *,speech,* ) ;;
  * ) echo "MODALITY_CYCLE must include speech" >&2; exit 2 ;;
esac
IFS=',' read -r -a modality_items <<<"$MODALITY_CYCLE"
for modality in "${modality_items[@]}"; do
  case "$modality" in
    text|image|speech) ;;
    *) echo "invalid MODALITY_CYCLE entry: $modality" >&2; exit 2 ;;
  esac
done
case "$CONDITIONAL_RANKING_NEGATIVE_MODE" in
  stride|hard_text|random) ;;
  *) echo "invalid CONDITIONAL_RANKING_NEGATIVE_MODE" >&2; exit 2 ;;
esac
for value in "$IMAGE_CONDITIONAL_RANKING_COEF" "$IMAGE_CONTRASTIVE_COEF"; do
  [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "image objective coefficients must be non-negative decimals" >&2
    exit 2
  }
done
image_objective_enabled="$(python3 -c 'import sys; print(int(any(float(value) > 0 for value in sys.argv[1:])))' "$IMAGE_CONDITIONAL_RANKING_COEF" "$IMAGE_CONTRASTIVE_COEF")"
if [ "$image_objective_enabled" = "1" ]; then
  case ",$MODALITY_CYCLE," in
    *,image,* ) ;;
    * ) echo "image objective requires image in MODALITY_CYCLE" >&2; exit 2 ;;
  esac
fi

STAGE_B_CHECKPOINT="${STAGE_B_CHECKPOINT:?STAGE_B_CHECKPOINT is required}"
STAGE_B_CHECKPOINT_SHA256="${STAGE_B_CHECKPOINT_SHA256:?STAGE_B_CHECKPOINT_SHA256 is required}"
BASELINE_LINEAR_INITIAL_CHECKPOINT="${BASELINE_LINEAR_INITIAL_CHECKPOINT:-}"
BASELINE_LINEAR_INITIAL_CHECKPOINT_SHA256="${BASELINE_LINEAR_INITIAL_CHECKPOINT_SHA256:-}"
BASELINE_LINEAR_INITIAL_MANIFEST="${BASELINE_LINEAR_INITIAL_MANIFEST:-}"
BASELINE_LINEAR_INITIAL_MANIFEST_SHA256="${BASELINE_LINEAR_INITIAL_MANIFEST_SHA256:-}"
NORM_IMAGE_INITIAL_CHECKPOINT="${NORM_IMAGE_INITIAL_CHECKPOINT:-}"
NORM_IMAGE_INITIAL_CHECKPOINT_SHA256="${NORM_IMAGE_INITIAL_CHECKPOINT_SHA256:-}"
NORM_IMAGE_INITIAL_MANIFEST="${NORM_IMAGE_INITIAL_MANIFEST:-}"
NORM_IMAGE_INITIAL_MANIFEST_SHA256="${NORM_IMAGE_INITIAL_MANIFEST_SHA256:-}"
SPEECH_INITIAL_CHECKPOINT="${SPEECH_INITIAL_CHECKPOINT:-}"
SPEECH_INITIAL_CHECKPOINT_SHA256="${SPEECH_INITIAL_CHECKPOINT_SHA256:-}"
SPEECH_INITIAL_MANIFEST="${SPEECH_INITIAL_MANIFEST:-}"
SPEECH_INITIAL_MANIFEST_SHA256="${SPEECH_INITIAL_MANIFEST_SHA256:-}"

NEED_BASELINE=0
NEED_NORM=0
NEED_SPEECH=0
IFS=',' read -r -a requested_arms <<<"$ONLY_RAW"
[ "${#requested_arms[@]}" -gt 0 ] || { echo "ONLY must select an arm" >&2; exit 2; }
for arm in "${requested_arms[@]}"; do
  case "$arm" in
    C0) NEED_BASELINE=1 ;;
    C_IMAGE_NORM_ONLY) NEED_NORM=1 ;;
    C_SPEECH_INIT_ONLY) NEED_BASELINE=1; NEED_SPEECH=1 ;;
    C_DUAL|C_DUAL_KD025) NEED_NORM=1; NEED_SPEECH=1 ;;
    *) echo "unsupported MM attribution arm: $arm" >&2; exit 2 ;;
  esac
done

if [ "$NEED_BASELINE" = "1" ]; then
  : "${BASELINE_LINEAR_INITIAL_CHECKPOINT:?baseline-linear image checkpoint is required}"
  : "${BASELINE_LINEAR_INITIAL_CHECKPOINT_SHA256:?baseline-linear image checkpoint SHA256 is required}"
  : "${BASELINE_LINEAR_INITIAL_MANIFEST:?baseline-linear image manifest is required}"
  : "${BASELINE_LINEAR_INITIAL_MANIFEST_SHA256:?baseline-linear image manifest SHA256 is required}"
fi
if [ "$NEED_NORM" = "1" ]; then
  : "${NORM_IMAGE_INITIAL_CHECKPOINT:?NORM50 image checkpoint is required}"
  : "${NORM_IMAGE_INITIAL_CHECKPOINT_SHA256:?NORM50 image checkpoint SHA256 is required}"
  : "${NORM_IMAGE_INITIAL_MANIFEST:?NORM50 image manifest is required}"
  : "${NORM_IMAGE_INITIAL_MANIFEST_SHA256:?NORM50 image manifest SHA256 is required}"
fi
if [ "$NEED_SPEECH" = "1" ]; then
  : "${SPEECH_INITIAL_CHECKPOINT:?speech checkpoint is required}"
  : "${SPEECH_INITIAL_CHECKPOINT_SHA256:?speech checkpoint SHA256 is required}"
  : "${SPEECH_INITIAL_MANIFEST:?speech manifest is required}"
  : "${SPEECH_INITIAL_MANIFEST_SHA256:?speech manifest SHA256 is required}"
fi

case "$DATA_DIR:$DEVELOPMENT_SPLIT_MANIFEST:$BASE_OUT:$STAGE_B_CHECKPOINT:$BASELINE_LINEAR_INITIAL_CHECKPOINT:$BASELINE_LINEAR_INITIAL_MANIFEST:$NORM_IMAGE_INITIAL_CHECKPOINT:$NORM_IMAGE_INITIAL_MANIFEST:$SPEECH_INITIAL_CHECKPOINT:$SPEECH_INITIAL_MANIFEST" in
  *sealed*|*synthetic*) echo "refusing sealed/synthetic path" >&2; exit 2 ;;
esac
[ "$GPU" = "1" ] || {
  echo "MM NORM+KD arms require exactly one GPU per job" >&2
  exit 2
}
reject_symlink_path_components() {
  local value="$1" absolute current component
  local -a components
  if [[ "$value" = /* ]]; then
    absolute="$value"
  else
    absolute="$PWD/$value"
  fi
  current="/"
  IFS=/ read -r -a components <<<"$absolute"
  for component in "${components[@]}"; do
    case "$component" in
      ""|.) continue ;;
      ..) current="${current%/*}"; [ -n "$current" ] || current="/" ;;
      *)
        if [ "$current" = "/" ]; then
          current="/$component"
        else
          current="$current/$component"
        fi
        [ ! -L "$current" ] || {
          echo "refusing unsafe DATA_DIR symlink component: $current" >&2
          exit 2
        }
        ;;
    esac
  done
}

reject_symlink_path_components "$DATA_DIR"
[ -d "$DATA_DIR" ] && [ -f "$DATA_DIR/manifest.json" ] || {
  echo "missing or unsafe real development data: $DATA_DIR" >&2
  exit 2
}
DATA_DIR="$(realpath -- "$DATA_DIR")"
[[ "$DATA_DIR" != *sealed* && "$DATA_DIR" != *synthetic* ]] || {
  echo "refusing sealed/synthetic canonical data path: $DATA_DIR" >&2
  exit 2
}

require_exact_file() {
  local label="$1" path="$2"
  [ -f "$path" ] && [ ! -L "$path" ] || {
    echo "missing or unsafe $label: $path" >&2
    exit 2
  }
  [[ "$path" = /* ]] && [ "$(realpath -- "$path")" = "$path" ] || {
    echo "$label must be an exact canonical absolute path: $path" >&2
    exit 2
  }
}

require_sha256() {
  local label="$1" path="$2" expected="$3"
  [[ "$expected" =~ ^[0-9a-fA-F]{64}$ ]] || {
    echo "$label SHA256 must be exact 64-character hex" >&2
    exit 2
  }
  [ "$(sha256sum "$path" | awk '{print $1}')" = "${expected,,}" ] || {
    echo "$label SHA-256 mismatch" >&2
    exit 2
  }
}

require_exact_file "development split manifest" "$DEVELOPMENT_SPLIT_MANIFEST"
python3 -c 'import sys; from scripts.audit_requirements import validate_development_multimodal_runtime_manifest as validate; result = validate(sys.argv[1], expected_source_commit_sha=sys.argv[2]); assert result["passed"], result["errors"]' "$DEVELOPMENT_SPLIT_MANIFEST" "$SOURCE_COMMIT_SHA"
require_exact_file "Stage B checkpoint" "$STAGE_B_CHECKPOINT"
require_sha256 "Stage B checkpoint" "$STAGE_B_CHECKPOINT" "$STAGE_B_CHECKPOINT_SHA256"
if [ "$NEED_BASELINE" = "1" ]; then
  require_exact_file "baseline-linear image checkpoint" "$BASELINE_LINEAR_INITIAL_CHECKPOINT"
  require_exact_file "baseline-linear image manifest" "$BASELINE_LINEAR_INITIAL_MANIFEST"
  require_sha256 "baseline-linear image checkpoint" "$BASELINE_LINEAR_INITIAL_CHECKPOINT" "$BASELINE_LINEAR_INITIAL_CHECKPOINT_SHA256"
  require_sha256 "baseline-linear image manifest" "$BASELINE_LINEAR_INITIAL_MANIFEST" "$BASELINE_LINEAR_INITIAL_MANIFEST_SHA256"
fi
if [ "$NEED_NORM" = "1" ]; then
  require_exact_file "NORM50 image checkpoint" "$NORM_IMAGE_INITIAL_CHECKPOINT"
  require_exact_file "NORM50 image manifest" "$NORM_IMAGE_INITIAL_MANIFEST"
  require_sha256 "NORM50 image checkpoint" "$NORM_IMAGE_INITIAL_CHECKPOINT" "$NORM_IMAGE_INITIAL_CHECKPOINT_SHA256"
  require_sha256 "NORM50 image manifest" "$NORM_IMAGE_INITIAL_MANIFEST" "$NORM_IMAGE_INITIAL_MANIFEST_SHA256"
fi
if [ "$NEED_SPEECH" = "1" ]; then
  require_exact_file "speech checkpoint" "$SPEECH_INITIAL_CHECKPOINT"
  require_exact_file "speech manifest" "$SPEECH_INITIAL_MANIFEST"
  require_sha256 "speech checkpoint" "$SPEECH_INITIAL_CHECKPOINT" "$SPEECH_INITIAL_CHECKPOINT_SHA256"
  require_sha256 "speech manifest" "$SPEECH_INITIAL_MANIFEST" "$SPEECH_INITIAL_MANIFEST_SHA256"
fi

verify_stage_a_manifest() {
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import json
import pathlib
import sys

checkpoint = pathlib.Path(sys.argv[1])
checkpoint_sha = sys.argv[2].lower()
manifest_path = pathlib.Path(sys.argv[3])
profile = sys.argv[4]
scope = "speech" if profile == "speech" else "image"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
completion = manifest.get("completion")
provenance = manifest.get("run_provenance")
args = manifest.get("args")
if not isinstance(completion, dict) or completion.get("status") != "completed":
    raise SystemExit("Stage A manifest completion is not completed")
if pathlib.Path(str(completion.get("e3_checkpoint_path", ""))).resolve() != checkpoint:
    raise SystemExit("Stage A manifest checkpoint path mismatch")
if completion.get("e3_checkpoint_sha256") != checkpoint_sha:
    raise SystemExit("Stage A manifest checkpoint SHA256 mismatch")
if not isinstance(provenance, dict) or not isinstance(args, dict):
    raise SystemExit("Stage A manifest provenance/args are missing")
for field in ("source_commit_sha", "runai_job_name", "runai_project"):
    if not provenance.get(field) or manifest.get(field) != provenance[field]:
        raise SystemExit(f"Stage A manifest {field} mismatch")
if provenance.get("policy") != "development_only_stage_a_multimodal_initialization":
    raise SystemExit("Stage A manifest provenance policy mismatch")
if provenance.get("sealed_evidence_used") is not False:
    raise SystemExit("Stage A manifest used sealed evidence")
if provenance.get("synthetic_evidence_used") is not False:
    raise SystemExit("Stage A manifest used synthetic evidence")
for field in ("resolved_data_root", "resolved_output_root"):
    value = pathlib.Path(str(provenance.get(field, "")))
    if not value.is_absolute() or any(term in str(value).lower() for term in ("sealed", "synthetic")):
        raise SystemExit(f"Stage A manifest invalid {field}")
requirements = {
    "final_steps": 500,
    "alignment_pretrain_steps": 400,
    "alignment_pretrain_modalities": scope,
}
if profile == "image_linear":
    requirements.update(
        image_bridge_type="linear_projector",
        image_prefix_tokens=50,
    )
elif profile == "image_norm":
    requirements.update(
        image_bridge_type="linear_projector_norm",
        image_prefix_tokens=50,
    )
elif profile == "speech":
    requirements.update(
        speech_unfreeze_last_blocks=1,
        speech_unfreeze_layer_norm=True,
        audio_bridge_type="attention_pool",
        audio_prefix_tokens=64,
    )
else:
    raise SystemExit(f"unsupported Stage A profile: {profile}")
for field, expected in requirements.items():
    if args.get(field) != expected:
        raise SystemExit(f"Stage A {profile} manifest mismatch for {field}")
PY
}

if [ "$NEED_BASELINE" = "1" ]; then
  verify_stage_a_manifest \
    "$BASELINE_LINEAR_INITIAL_CHECKPOINT" "$BASELINE_LINEAR_INITIAL_CHECKPOINT_SHA256" \
    "$BASELINE_LINEAR_INITIAL_MANIFEST" image_linear
fi
if [ "$NEED_NORM" = "1" ]; then
  verify_stage_a_manifest \
    "$NORM_IMAGE_INITIAL_CHECKPOINT" "$NORM_IMAGE_INITIAL_CHECKPOINT_SHA256" \
    "$NORM_IMAGE_INITIAL_MANIFEST" image_norm
fi
if [ "$NEED_SPEECH" = "1" ]; then
  verify_stage_a_manifest \
    "$SPEECH_INITIAL_CHECKPOINT" "$SPEECH_INITIAL_CHECKPOINT_SHA256" \
    "$SPEECH_INITIAL_MANIFEST" speech
fi

ONLY_SET=",${ONLY_RAW},"
SPECS=(
  "C0|baseline|0|0"
  "C_IMAGE_NORM_ONLY|norm|0|0"
  "C_SPEECH_INIT_ONLY|baseline|1|0"
  "C_DUAL|norm|1|0"
  "C_DUAL_KD025|norm|1|0.25"
)

submit_arm() {
  local arm="$1" image_kind="$2" use_speech="$3" kd_coef="$4"
  local lower out job image_bridge image_checkpoint image_sha image_manifest
  local speech_checkpoint="" speech_sha="" speech_manifest="" speech_initializer="none"
  [[ "$ONLY_SET" == *",${arm},"* ]] || return 0
  case "$image_kind" in
    baseline)
      image_bridge=linear_projector
      image_checkpoint="$BASELINE_LINEAR_INITIAL_CHECKPOINT"
      image_sha="$BASELINE_LINEAR_INITIAL_CHECKPOINT_SHA256"
      image_manifest="$BASELINE_LINEAR_INITIAL_MANIFEST"
      ;;
    norm)
      image_bridge=linear_projector_norm
      image_checkpoint="$NORM_IMAGE_INITIAL_CHECKPOINT"
      image_sha="$NORM_IMAGE_INITIAL_CHECKPOINT_SHA256"
      image_manifest="$NORM_IMAGE_INITIAL_MANIFEST"
      ;;
    *) echo "internal unsupported image initializer: $image_kind" >&2; return 2 ;;
  esac
  if [ "$use_speech" = "1" ]; then
    speech_checkpoint="$SPEECH_INITIAL_CHECKPOINT"
    speech_sha="$SPEECH_INITIAL_CHECKPOINT_SHA256"
    speech_manifest="$SPEECH_INITIAL_MANIFEST"
    speech_initializer=verified_last1_ln
  fi
  lower="${arm,,}"
  out="$BASE_OUT/${lower}_seed${SEED}"
  job="sme-${lower//_/-}-s${SEED}-${STAMP}"
  [ ! -e "$out" ] || { echo "refusing overwrite: $out" >&2; return 1; }
  [ "${#job}" -le 55 ] || { echo "job name too long: $job" >&2; return 1; }

  if [ "$DRY_RUN" = "1" ]; then
    printf 'arm=%s job=%s gpu=1 top_k=2 alignment=speech:%s main=%s image_initializer=%s image_bridge=%s image_checkpoint=%s image_sha256=%s image_manifest=%s speech_initializer=%s speech_checkpoint=%s speech_sha256=%s speech_manifest=%s development_split_manifest=%s modality_cycle=%s negative_mode=%s image_rank_coef=%s image_contrastive_coef=%s router=0 experts=0 lm_head=0 kd_coef=%s kd_temperature=1 stage_b_sha256=%s source=%s\n' \
      "$arm" "$job" "$ALIGNMENT_PRETRAIN_STEPS" "$FINAL_STEPS" "$image_kind" "$image_bridge" "$image_checkpoint" "$image_sha" \
      "$image_manifest" "$speech_initializer" "${speech_checkpoint:-none}" \
      "${speech_sha:-none}" "${speech_manifest:-none}" "$DEVELOPMENT_SPLIT_MANIFEST" "$MODALITY_CYCLE" "$CONDITIONAL_RANKING_NEGATIVE_MODE" "$IMAGE_CONDITIONAL_RANKING_COEF" "$IMAGE_CONTRASTIVE_COEF" "$kd_coef" "$STAGE_B_CHECKPOINT_SHA256" \
      "$SOURCE_COMMIT_SHA"
    return 0
  fi

  env JOB_NAME="$job" OUT="$out" DATA_DIR="$DATA_DIR" DEVELOPMENT_SPLIT_MANIFEST="$DEVELOPMENT_SPLIT_MANIFEST" SUBMIT_REPO_DIR="$REPO_ROOT" \
    GPU=1 CPU="${CPU:-8}" MEMORY="${MEMORY:-120G}" SEED="$SEED" \
    SOURCE_COMMIT_SHA="$SOURCE_COMMIT_SHA" TOP_K=2 FINAL_STEPS="$FINAL_STEPS" \
    ALIGNMENT_PRETRAIN_STEPS="$ALIGNMENT_PRETRAIN_STEPS" ALIGNMENT_PRETRAIN_LOG_EVERY=25 \
    ALIGNMENT_PRETRAIN_MODALITIES=speech MODALITY_CYCLE="$MODALITY_CYCLE" \
    ABLATION_STEPS=0 CAPACITY_ABLATION_STEPS=0 EXPERT_ABLATION_STEPS=0 \
    POSTPROCESS_REQUIRED_RUNS=0 CAPACITY_FACTOR=8.0 AUX_COEF=0.02 \
    TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" EVAL_BATCH_SIZE=1 \
    SPEECH_TEACHER_BANK_BATCH_SIZE="$SPEECH_TEACHER_BANK_BATCH_SIZE" TEXT_EVAL_BLOCKS=160 \
    IMAGE_EVAL_SAMPLES=137 SPEECH_EVAL_SAMPLES=137 RETRIEVAL_EVAL_SAMPLES=137 \
    CONDITIONAL_EVAL_SAMPLES=137 CONDITIONAL_NEGATIVES=9 CONDITIONAL_BATCH_SIZE=1 \
    CONDITIONAL_RANKING_NEGATIVES=9 CONDITIONAL_RANKING_NEGATIVE_MODE="$CONDITIONAL_RANKING_NEGATIVE_MODE" \
    CONDITIONAL_RANKING_HARD_POOL_SIZE=512 CONDITIONAL_RANKING_TEMPERATURE=0.7 \
    IMAGE_CONDITIONAL_RANKING_COEF="$IMAGE_CONDITIONAL_RANKING_COEF"  \
    IMAGE_PREFIX_TOKENS=50 AUDIO_PREFIX_TOKENS=64 ENCODER_FEATURE_TOKENS=100 \
    IMAGE_ALIGNMENT_TARGET=olmoe_caption_hidden IMAGE_BRIDGE_TYPE="$image_bridge" \
    AUDIO_BRIDGE_TYPE=attention_pool AUDIO_MAX_SECONDS=6.0 \
    TRAIN_ROUTER_GATES=0 TRAIN_EXPERTS=0 TRAIN_LM_HEAD=0 DYNAMIC_EXPERT_BIAS_LR=0.0 \
    EXPERT_SELECTION_JSON= EXPERT_UPDATE_MODE=full ALLOW_SELECTED_EXPERT_ROUTER_TUNING=0 \
    ROUTER_LEARNING_RATE=0.000002 EXPERT_LEARNING_RATE=0.000001  \
    EXPERT_ANCHOR_COEFFICIENT=0.01 LM_HEAD_LEARNING_RATE=0.00001  \
    STAGE_B_CHECKPOINT="$STAGE_B_CHECKPOINT" \
    STAGE_B_CHECKPOINT_SHA256="$STAGE_B_CHECKPOINT_SHA256" \
    MULTIMODAL_INITIAL_CHECKPOINT="$image_checkpoint" \
    MULTIMODAL_INITIAL_CHECKPOINT_SHA256="$image_sha" \
    MULTIMODAL_INITIAL_MANIFEST="$image_manifest" \
    MULTIMODAL_INITIALIZATION_SCOPE=image \
    SPEECH_INITIAL_CHECKPOINT="$speech_checkpoint" \
    SPEECH_INITIAL_CHECKPOINT_SHA256="$speech_sha" \
    SPEECH_INITIAL_MANIFEST="$speech_manifest" \
    SPEECH_UNFREEZE_LAST_BLOCKS=1 SPEECH_UNFREEZE_LAYER_NORM=1 \
    SPEECH_ENCODER_LEARNING_RATE=0.000005 LEARNING_RATE=0.0005 \
    RETRIEVAL_HEAD_LEARNING_RATE=0.0 CONTRASTIVE_COEF=0.2 \
    CENTER_POSITIVE_WEIGHT=1.0 RAW_POSITIVE_WEIGHT=0.0 \
    IMAGE_CONTRASTIVE_COEF="$IMAGE_CONTRASTIVE_COEF" SPEECH_CONTRASTIVE_COEF=0.2 \
    CONTRASTIVE_NEGATIVES=128 IMAGE_CONTRASTIVE_NEGATIVES=-1 \
    SPEECH_CONTRASTIVE_NEGATIVES=-1 CONTRASTIVE_TEMPERATURE=0.07 \
    IMAGE_CONTRASTIVE_TEMPERATURE=0.07 SPEECH_CONTRASTIVE_TEMPERATURE=0.04 \
    IMAGE_CENTER_POSITIVE_WEIGHT=1.5 IMAGE_RAW_POSITIVE_WEIGHT=0.05 \
    SPEECH_CENTER_POSITIVE_WEIGHT=5.0 SPEECH_RAW_POSITIVE_WEIGHT=0.0 \
    SPEECH_CONDITIONAL_RANKING_COEF=3.0 SPEECH_BEHAVIOR_KL_COEF="$kd_coef" \
    SPEECH_BEHAVIOR_KL_TEMPERATURE=1 WEIGHT_DECAY=0.0 GRAD_CLIP=5.0 \
    LOG_EVERY_STEPS=25 SAVE_EVERY_STEPS="$FINAL_STEPS" \
    bash scripts/submit_e3_candidate_runai.sh
}

for spec in "${SPECS[@]}"; do
  IFS='|' read -r arm image_kind use_speech kd_coef <<<"$spec"
  submit_arm "$arm" "$image_kind" "$use_speech" "$kd_coef"
done

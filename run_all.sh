set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT:-test_code_final.py}"        
RUNS_DIR="${RUNS_DIR:-runs_new}"               
SEED="${SEED:-42}"
N_TRIALS="${N_TRIALS:-120}"                  
N_FOLDS="${N_FOLDS:-5}"
N_REPEATS="${N_REPEATS:-2}"
MAX_JOBS="${MAX_JOBS:-2}"                   
  
BASES=(
  "base/44gg"
  "base/33gg"
  "base/base_menor_2_grupos_s_o"
  "base/spam_a"
  "base/3clusters_3outliers"
  "base/base_menor"
  "base/base_menor_2_grupos"
  "base/55gg"
  "base/landmine"
  "base/spam_b"
)

ESTIMATORS=(logreg)
DEPTHS=(4) 

ROOT="$(pwd)"

declare -a JOB_PIDS=()
declare -A JOB_DESC=()

launch_job() {
  local base="$1"
  local est="$2"
  local depth="$3"
  local bname
  bname="$(basename "$base")"
  local outdir="${RUNS_DIR}/${bname}/${est}/depth_${depth}"
  mkdir -p "$outdir"
  echo ">> Base=${base}  Estimator=${est}  Depth=${depth}"

  (
    cd "$outdir"
    "${PYTHON_BIN}" "${ROOT}/${SCRIPT}" \
      --path "${ROOT}/${base}" \
      --estimator "${est}" \
      --depth "${depth}" \
      --n_trials "${N_TRIALS}" \
      --seed "${SEED}" \
      --scoring auto \
      --test_size 0.50 \
      --val_size 0.25 \
      --improve_delta 0.01 \
      --compare_mtl \
      --mtl_folds "${N_FOLDS}" \
      --mtl_repeats "${N_REPEATS}" \
      2>&1 | tee run.log
  ) &
  local pid=$!
  JOB_PIDS+=("$pid")
  JOB_DESC["$pid"]="Base=${base} Estimator=${est} Depth=${depth}"
}

wait_for_slot() {
  while (( ${#JOB_PIDS[@]} >= MAX_JOBS )); do
    local pid="${JOB_PIDS[0]}"
    wait "$pid"
    local status=$?
    if (( status != 0 )); then
      echo "[ERRO] Execução falhou (${JOB_DESC[$pid]})" >&2
      exit "$status"
    fi
    unset 'JOB_PIDS[0]'
    JOB_PIDS=("${JOB_PIDS[@]}")
  done
}

for BASE in "${BASES[@]}"; do
  for EST in "${ESTIMATORS[@]}"; do
    for D in "${DEPTHS[@]}"; do
      wait_for_slot
      launch_job "$BASE" "$EST" "$D"
    done
  done
done

for pid in "${JOB_PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "[ERRO] Execução falhou (${JOB_DESC[$pid]})" >&2
    exit 1
  fi
done

echo "Concluído."

#!/usr/bin/env bash
set -euo pipefail

# Submit unconstrained ClaSSE MLE runs as a Slurm job array.
# One array task per (dataset × num_hidden) combination.
#
# Normal usage — submits all discovered datasets:
#   bash scripts/subsampled_leaves_experiment/cluster_run_classe_unconstrained.sh
#
# Common overrides:
#   TREE_COUNTS="64" TRIALS="0" SAMPLE_PROBS="0.2" \
#     bash scripts/subsampled_leaves_experiment/cluster_run_classe_unconstrained.sh
#
#   DRY_RUN=1 \
#     bash scripts/subsampled_leaves_experiment/cluster_run_classe_unconstrained.sh
#
#   BACKEND=vector_transport SBATCH_TIME=1-00:00:00 SBATCH_MEM=32G \
#     bash scripts/subsampled_leaves_experiment/cluster_run_classe_unconstrained.sh
#
#   NUM_RESTARTS=3 SBATCH_TIME=12:00:00 \
#     bash scripts/subsampled_leaves_experiment/cluster_run_classe_unconstrained.sh
#
# Dataset layout (same as TROUPE experiment):
#   <EXPERIMENT_ROOT>/trees_<n>/time_<t>/sample_<p>/trial_<k>/trees.pkl
# Results written to:
#   <RESULTS_ROOT>/trees_<n>/time_<t>/sample_<p>/trial_<k>/hidden=<N>/

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

BACKEND="${BACKEND:-fundamental}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${WORKDIR}/experiments/subsampled_leaves_4_terminals}"
RESULTS_ROOT="${RESULTS_ROOT:-${WORKDIR}/results/subsampled_leaves_4_terminals/classe_unconstrained/${BACKEND}}"

PYTHON_BIN="${PYTHON_BIN:-${WORKDIR}/../.venv/bin/python}"
VENV_ACTIVATE="${VENV_ACTIVATE:-}"

# Space-delimited list of hidden-state counts to evaluate per dataset.
NUM_HIDDEN_VALUES="${NUM_HIDDEN_VALUES:-4}"
# L1 regularisation on off-diagonal birth-kernel entries (default: none).
L1="${L1:-0.0}"
# Number of random restarts per job (restart 0 is deterministic).
NUM_RESTARTS="${NUM_RESTARTS:-1}"
DEVICE="${DEVICE:-cpu}"
UNCONSTRAINED_EXTRA_ARGS="${UNCONSTRAINED_EXTRA_ARGS:-}"

# Filters: space-delimited lists; leave empty to include all.
TREE_COUNTS="${TREE_COUNTS:-}"
PROCESS_TIMES="${PROCESS_TIMES:-}"
SAMPLE_PROBS="${SAMPLE_PROBS:-0.05 0.1 0.2}"
TRIALS="${TRIALS:-}"

SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-uncon}"

SBATCH_TIME="${SBATCH_TIME:-6:00:00}"
SBATCH_CPUS="${SBATCH_CPUS:-2}"
SBATCH_MEM="${SBATCH_MEM:-16G}"
SBATCH_PARTITION="${SBATCH_PARTITION:-}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"
SBATCH_QOS="${SBATCH_QOS:-}"
SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-}"
# Maximum simultaneously running array tasks (0 = unlimited).
MAX_ARRAY_CONCURRENT="${MAX_ARRAY_CONCURRENT:-0}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

list_contains() {
    local needle="$1"
    shift || true
    local candidate
    for candidate in "$@"; do
        [[ "${candidate}" == "${needle}" ]] && return 0
    done
    return 1
}


should_include_dataset() {
    local tree_count="$1" process_time="$2" sample_prob="$3" trial="$4"
    local values=()

    if [[ -n "${TREE_COUNTS}" ]]; then
        read -r -a values <<< "${TREE_COUNTS}"
        list_contains "${tree_count}" "${values[@]}" || return 1
    fi
    if [[ -n "${PROCESS_TIMES}" ]]; then
        read -r -a values <<< "${PROCESS_TIMES}"
        list_contains "${process_time}" "${values[@]}" || return 1
    fi
    if [[ -n "${SAMPLE_PROBS}" ]]; then
        read -r -a values <<< "${SAMPLE_PROBS}"
        list_contains "${sample_prob}" "${values[@]}" || return 1
    fi
    if [[ -n "${TRIALS}" ]]; then
        read -r -a values <<< "${TRIALS}"
        list_contains "${trial}" "${values[@]}" || return 1
    fi
    return 0
}


# ---------------------------------------------------------------------------
# Job execution (called inside each array task)
# ---------------------------------------------------------------------------

run_one_job() {
    local trees_file="$1"
    local outdir="$2"
    local sampling_prob="$3"
    local num_hidden="$4"

    local marker="${outdir}/classe_unconstrained_summary.txt"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${marker}" ]]; then
        printf 'Result already exists, skipping: %s\n' "${outdir}"
        return 0
    fi

    mkdir -p "${outdir}"
    cd "${WORKDIR}"

    if [[ -n "${VENV_ACTIVATE}" ]]; then
        # shellcheck disable=SC1090
        source "${VENV_ACTIVATE}"
    fi

    if [[ ! -x "${PYTHON_BIN}" ]]; then
        printf 'Python executable not found or not executable: %s\n' "${PYTHON_BIN}" >&2
        exit 1
    fi

    local -a cmd=(
        "${PYTHON_BIN}"
        "scripts/run_classe_unconstrained.py"
        "-i"  "${trees_file}"
        "-o"  "${outdir}"
        "--sampling_probability" "${sampling_prob}"
        "--num_hidden"           "${num_hidden}"
        "--num_restarts"         "${NUM_RESTARTS}"
        "--l1"                   "${L1}"
        "--device"               "${DEVICE}"
        "--backend"              "${BACKEND}"
    )

    if [[ -n "${UNCONSTRAINED_EXTRA_ARGS}" ]]; then
        local -a extra_args=()
        read -r -a extra_args <<< "${UNCONSTRAINED_EXTRA_ARGS}"
        cmd+=("${extra_args[@]}")
    fi

    printf 'Working directory: %s\n' "${WORKDIR}"
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'

    "${cmd[@]}"
}


# ---------------------------------------------------------------------------
# Array task entry point (called by Slurm via run-array-task)
# ---------------------------------------------------------------------------

run_array_task() {
    local manifest="$1"
    local task_id="${SLURM_ARRAY_TASK_ID:-0}"

    local line
    line=$(sed -n "$((task_id + 1))p" "${manifest}")

    if [[ -z "${line}" ]]; then
        printf 'No entry for task id %d in manifest %s\n' "${task_id}" "${manifest}" >&2
        exit 1
    fi

    local trees_file outdir sampling_prob num_hidden
    IFS=$'\t' read -r trees_file outdir sampling_prob num_hidden <<< "${line}"

    run_one_job "${trees_file}" "${outdir}" "${sampling_prob}" "${num_hidden}"
}


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

submit_jobs() {
    if [[ "${DRY_RUN}" != "1" ]] && ! command -v sbatch >/dev/null 2>&1; then
        printf 'sbatch is not available in PATH\n' >&2
        exit 1
    fi

    mkdir -p "${RESULTS_ROOT}"

    local -a all_trees_files=() all_outdirs=() all_sampling_probs=() all_num_hidden=()
    local found_any=0 skipped=0
    local -a hidden_values=()
    read -r -a hidden_values <<< "${NUM_HIDDEN_VALUES}"

    while IFS= read -r trees_file; do
        found_any=1
        local rel_path="${trees_file#${EXPERIMENT_ROOT}/}"
        IFS='/' read -r trees_dir time_dir sample_dir trial_dir _ <<< "${rel_path}"

        local tree_count="${trees_dir#trees_}"
        local process_time="${time_dir#time_}"
        local sample_prob="${sample_dir#sample_}"
        local trial="${trial_dir#trial_}"

        if ! should_include_dataset "${tree_count}" "${process_time}" "${sample_prob}" "${trial}"; then
            continue
        fi

        for num_hidden in "${hidden_values[@]}"; do
            local outdir="${RESULTS_ROOT}/${trees_dir}/${time_dir}/${sample_dir}/${trial_dir}/hidden=${num_hidden}"
            local marker="${outdir}/classe_unconstrained_summary.txt"
            if [[ "${SKIP_EXISTING}" == "1" && -f "${marker}" ]]; then
                printf 'Skipping existing result: %s\n' "${outdir}"
                skipped=$((skipped + 1))
                continue
            fi

            all_trees_files+=("${trees_file}")
            all_outdirs+=("${outdir}")
            all_sampling_probs+=("${sample_prob}")
            all_num_hidden+=("${num_hidden}")
        done
    done < <(find "${EXPERIMENT_ROOT}" -type f -name 'trees.pkl' | sort)

    if [[ "${found_any}" == "0" ]]; then
        printf 'No trees.pkl files found under %s\n' "${EXPERIMENT_ROOT}" >&2
        exit 1
    fi

    local n_jobs="${#all_trees_files[@]}"
    if [[ "${n_jobs}" == "0" ]]; then
        printf 'No new jobs to submit (skipped=%d).\n' "${skipped}"
        return 0
    fi

    # Write manifest: one tab-separated line per task.
    local manifest_dir="${RESULTS_ROOT}/manifests"
    mkdir -p "${manifest_dir}"
    local manifest="${manifest_dir}/$(date +%Y%m%d_%H%M%S).tsv"
    for i in "${!all_trees_files[@]}"; do
        printf '%s\t%s\t%s\t%s\n' \
            "${all_trees_files[$i]}" "${all_outdirs[$i]}" \
            "${all_sampling_probs[$i]}" "${all_num_hidden[$i]}"
    done > "${manifest}"

    # Build Slurm array spec, optionally rate-limiting concurrent tasks.
    local array_spec="0-$((n_jobs - 1))"
    if [[ "${MAX_ARRAY_CONCURRENT}" -gt 0 ]]; then
        array_spec="${array_spec}%${MAX_ARRAY_CONCURRENT}"
    fi

    local log_dir="${RESULTS_ROOT}/slurm_logs"
    mkdir -p "${log_dir}"

    local -a sbatch_args=(
        --parsable
        --job-name="${JOB_NAME_PREFIX}"
        --output="${log_dir}/%x.%A_%a.out"
        --error="${log_dir}/%x.%A_%a.err"
        --array="${array_spec}"
        --time="${SBATCH_TIME}"
        --cpus-per-task="${SBATCH_CPUS}"
        --mem="${SBATCH_MEM}"
    )

    if [[ "${DEVICE}" == cuda* ]]; then
        sbatch_args+=(--gres=gpu:1)
    fi

    [[ -n "${SBATCH_PARTITION}" ]]  && sbatch_args+=(--partition="${SBATCH_PARTITION}")
    [[ -n "${SBATCH_ACCOUNT}" ]]    && sbatch_args+=(--account="${SBATCH_ACCOUNT}")
    [[ -n "${SBATCH_QOS}" ]]        && sbatch_args+=(--qos="${SBATCH_QOS}")
    [[ -n "${SBATCH_CONSTRAINT}" ]] && sbatch_args+=(--constraint="${SBATCH_CONSTRAINT}")

    # Export all runtime config so array tasks inherit it.
    export WORKDIR PYTHON_BIN VENV_ACTIVATE BACKEND
    export NUM_HIDDEN_VALUES L1 NUM_RESTARTS DEVICE UNCONSTRAINED_EXTRA_ARGS SKIP_EXISTING

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY RUN: would submit array of %d tasks\n' "${n_jobs}"
        printf 'Manifest: %s\n' "${manifest}"
        printf 'sbatch args: %s\n' "${sbatch_args[*]}"
        printf 'Submission summary: to_submit=%d skipped=%d results_root=%s\n' \
            "${n_jobs}" "${skipped}" "${RESULTS_ROOT}"
    else
        local job_id
        job_id=$(sbatch "${sbatch_args[@]}" "${0}" run-array-task "${manifest}")
        printf 'Submitted array job %s (%d tasks)\n' "${job_id}" "${n_jobs}"
        printf 'Manifest: %s\n' "${manifest}"
        printf 'Logs:     %s\n' "${log_dir}"
        printf 'Submission summary: to_submit=%d skipped=%d results_root=%s\n' \
            "${n_jobs}" "${skipped}" "${RESULTS_ROOT}"
    fi
}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

main() {
    local mode="${1:-submit}"

    case "${mode}" in
        run-array-task)
            if [[ "$#" -ne 2 ]]; then
                printf 'Usage: %s run-array-task <manifest>\n' "$0" >&2
                exit 1
            fi
            run_array_task "$2"
            ;;
        run-one)
            # Direct single-job execution (useful for local testing).
            if [[ "$#" -ne 5 ]]; then
                printf 'Usage: %s run-one <trees_file> <outdir> <sampling_prob> <num_hidden>\n' "$0" >&2
                exit 1
            fi
            run_one_job "$2" "$3" "$4" "$5"
            ;;
        submit)
            submit_jobs
            ;;
        *)
            printf 'Unknown mode: %s\n' "${mode}" >&2
            printf 'Usage: %s [submit|run-one|run-array-task]\n' "$0" >&2
            exit 1
            ;;
    esac
}


main "$@"

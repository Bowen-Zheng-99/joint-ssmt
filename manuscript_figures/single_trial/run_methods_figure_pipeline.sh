#!/bin/bash
# Run simulation and inference for methods figure
#
# Computes BOTH parametric and permutation-based p-values for PLV/SFC
#
# Usage:
#   bash run_methods_figure_pipeline.sh [--duration 60] [--b0_mu -3.0] [--plot_duration 3]

# Defaults
DURATION=300           # Simulation duration (seconds)
B0_MU=-3.5
OUTPUT_DIR="./final_figure_data"
PLOT_DURATION=6        # How much raw data to plot (seconds)
N_PERMUTATIONS=500      # Number of permutations for permutation test
SKIP_SIMULATION=false
SKIP_TRADITIONAL=false
SKIP_INFERENCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --duration) DURATION="$2"; shift 2 ;;
        --b0_mu) B0_MU="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --plot_duration) PLOT_DURATION="$2"; shift 2 ;;
        --n_permutations) N_PERMUTATIONS="$2"; shift 2 ;;
        --skip_simulation) SKIP_SIMULATION=true; shift ;;
        --skip_traditional) SKIP_TRADITIONAL=true; shift ;;
        --skip_inference) SKIP_INFERENCE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

DATA_PATH="${OUTPUT_DIR}/sim_data.pkl"
RESULTS_DIR="${OUTPUT_DIR}/results"
TRADITIONAL_PATH="${RESULTS_DIR}/traditional_methods.pkl"
FIGURES_DIR="${OUTPUT_DIR}/figures"

echo "========================================"
echo "Methods Figure Pipeline"
echo "========================================"
echo "  Simulation duration: ${DURATION}s"
echo "  Plot duration: ${PLOT_DURATION}s"
echo "  b0_mu: ${B0_MU}"
echo "  Permutations: ${N_PERMUTATIONS}"
echo "  Output: ${OUTPUT_DIR}"
echo "========================================"

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${FIGURES_DIR}"

# 1. Simulate data
if [ "$SKIP_SIMULATION" = false ]; then
    echo ""
    echo "[1/5] Simulating data..."
    python -m ct_ssmt.simulate_single_trial \
        --output "${DATA_PATH}" \
        --duration "${DURATION}" \
        --b0_mu "${B0_MU}" \
        --seed 1000000
else
    echo ""
    echo "[1/5] Skipping simulation (using existing data)"
fi



# 2. Compute traditional methods (parametric + permutation)
if [ "$SKIP_TRADITIONAL" = false ]; then
    echo ""
    echo "[2/5] Computing traditional methods (parametric + permutation)..."
    python compute_traditional_methods_single.py \
        --data "${DATA_PATH}" \
        --output "${TRADITIONAL_PATH}" \
        --method both \
        --test_type both \
        --n_permutations "${N_PERMUTATIONS}"
else
    echo ""
    echo "[2/5] Skipping traditional methods (using existing results)"
fi

# 3. Run joint inference
if [ "$SKIP_INFERENCE" = false ]; then
    echo ""
    echo "[3/5] Running joint inference..."
    python run_analysis.py \
        --data "${DATA_PATH}" \
        --output "${OUTPUT_DIR}" \
        --traditional "${TRADITIONAL_PATH}"
else
    echo ""
    echo "[3/5] Skipping inference (using existing results)"
fi

# 4. Generate methods figure panels
echo ""
echo "[4/5] Generating figure panels..."
python plot_methods_figure.py \
    --data "${DATA_PATH}" \
    --results "${RESULTS_DIR}" \
    --output "${FIGURES_DIR}" \
    --freqs_to_show 11 27 \
    --plot_duration "${PLOT_DURATION}" \
    --snapshot_sec 30.0 \
    --n_snapshots 2 \
    --corr_window 10.0


# plotting heatmap 
python plot_heatmap.py

# plot the spike probability
python plot_posterior_spike_probability.py \
    --sim "${DATA_PATH}" \
    --joint "${RESULTS_DIR}" \
    --output "${FIGURES_DIR}/posterior_spike_prob.pdf" \
    --t-start 100 \
    --t-duration 2

python plot_sampling.py
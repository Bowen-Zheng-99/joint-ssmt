#!/bin/bash
# Run trial-structured spike-field coupling simulation and analysis pipeline.
#
# Usage:
#   bash run_sim_figures.sh                          # Full pipeline (sim + trad + inference + figures)
#   bash run_sim_figures.sh --data ./data/sim.pkl    # Skip simulation, use existing data
#   bash run_sim_figures.sh --skip_inference          # Skip inference, use existing results
#   bash run_sim_figures.sh --quick                   # Fewer iterations for quick testing
set -e

# Defaults
DATA_DIR="./data"
OUTPUT_DIR="./results/trial_test"
SEED=42
R=100  S=5  FREQ_STEP=2.0  WINDOW_SEC=0.4  NW=1
FIXED_ITER=500  N_REFRESHES=10  N_PERMUTATIONS=500  DOWNSAMPLE=10

SKIP_SIM=false
SKIP_TRADITIONAL=false
SKIP_INFERENCE=false
SKIP_FIGURES=false
DATA_PATH=""
SAMPLE_TRIALS="0 25 50 99"

while [[ $# -gt 0 ]]; do
    case $1 in
        --data)             DATA_PATH="$2"; SKIP_SIM=true; shift 2 ;;
        --output)           OUTPUT_DIR="$2"; shift 2 ;;
        --skip_simulation)  SKIP_SIM=true; shift ;;
        --skip_traditional) SKIP_TRADITIONAL=true; shift ;;
        --skip_inference)   SKIP_INFERENCE=true; shift ;;
        --skip_figures)     SKIP_FIGURES=true; shift ;;
        --quick)            FIXED_ITER=100; N_REFRESHES=3; N_PERMUTATIONS=50; shift ;;
        --seed)             SEED="$2"; shift 2 ;;
        --R)                R="$2"; shift 2 ;;
        --S)                S="$2"; shift 2 ;;
        --sample_trials)    SAMPLE_TRIALS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data PATH          Use existing simulation data (skips simulation)"
            echo "  --output DIR         Output directory (default: $OUTPUT_DIR)"
            echo "  --skip_simulation    Skip data simulation"
            echo "  --skip_traditional   Skip PLV/SFC computation"
            echo "  --skip_inference     Skip CT-SSMT inference (use existing results)"
            echo "  --skip_figures       Skip figure generation"
            echo "  --quick              Fewer iterations for quick testing"
            echo "  --seed N             Random seed (default: $SEED)"
            echo "  --R N                Number of trials (default: $R)"
            echo "  --S N                Number of units (default: $S)"
            echo "  --sample_trials STR  Space-separated trial indices for dynamics plot (default: $SAMPLE_TRIALS)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Default data path if not set via --data
if [ -z "$DATA_PATH" ]; then
    DATA_PATH="$DATA_DIR/sim_with_trials.pkl"
fi

RESULTS_DIR="$OUTPUT_DIR/results"
TRADITIONAL_PATH="$OUTPUT_DIR/traditional_methods.pkl"
FIGURES_DIR="$OUTPUT_DIR/figures"

echo "========================================"
echo "Trial-Structured Pipeline"
echo "========================================"
echo "  Data:        $DATA_PATH"
echo "  Output:      $OUTPUT_DIR"
echo "  Seed:        $SEED"
echo "  R=$R  S=$S"
echo "  MCMC:        fixed_iter=$FIXED_ITER  n_refreshes=$N_REFRESHES"
echo "  Permutations: $N_PERMUTATIONS"
echo "  Skip sim:    $SKIP_SIM"
echo "  Skip trad:   $SKIP_TRADITIONAL"
echo "  Skip infer:  $SKIP_INFERENCE"
echo "  Skip figs:   $SKIP_FIGURES"
echo "========================================"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$RESULTS_DIR" "$FIGURES_DIR"

# =========================================================================
# 1. Simulate data
# =========================================================================
if [ "$SKIP_SIM" = false ]; then
    echo ""
    echo "[1/4] Simulating trial data..."
    python simulate_trial_data.py \
        --out "$DATA_PATH" \
        --R $R --S $S \
        --seed $SEED
else
    echo ""
    echo "[1/4] Skipping simulation (using existing data: $DATA_PATH)"
    if [ ! -f "$DATA_PATH" ]; then
        echo "ERROR: Data file not found: $DATA_PATH"
        exit 1
    fi
fi

# =========================================================================
# 2. Traditional methods (PLV, SFC)
# =========================================================================
if [ "$SKIP_TRADITIONAL" = false ]; then
    echo ""
    echo "[2/4] Computing traditional methods (PLV, SFC)..."
    python compute_traditional_methods.py \
        --data "$DATA_PATH" \
        --output "$TRADITIONAL_PATH" \
        --n_permutations $N_PERMUTATIONS \
        --seed $SEED \
        --skip_permutation
else
    echo ""
    echo "[2/4] Skipping traditional methods"
fi

# =========================================================================
# 3. Run CT-SSMT analysis pipeline
# =========================================================================
echo ""
echo "[3/4] Running analysis pipeline..."

TRAD_ARG=""
if [ -f "$TRADITIONAL_PATH" ]; then
    TRAD_ARG="--traditional $TRADITIONAL_PATH"
fi

INFERENCE_ARG=""
if [ "$SKIP_INFERENCE" = true ]; then
    INFERENCE_ARG="--skip_inference"
fi

python run_analysis_trials.py \
    --data "$DATA_PATH" \
    --output "$OUTPUT_DIR" \
    --freq_step $FREQ_STEP \
    --window_sec $WINDOW_SEC \
    --NW $NW \
    --fixed_iter $FIXED_ITER \
    --n_refreshes $N_REFRESHES \
    --downsample_factor $DOWNSAMPLE \


# =========================================================================
# 4. Generate manuscript figures
# =========================================================================
if [ "$SKIP_FIGURES" = false ]; then
    echo ""
    echo "[4/4] Generating manuscript figures..."

    JOINT_ARG=""
    if [ -d "$RESULTS_DIR" ]; then
        JOINT_ARG="--joint $RESULTS_DIR"
    fi

    TRAD_FIG_ARG=""
    if [ -f "$TRADITIONAL_PATH" ]; then
        TRAD_FIG_ARG="--traditional $TRADITIONAL_PATH"
    fi

    python plot_trial_manuscript_figures.py \
        --data "$DATA_PATH" \
        $JOINT_ARG \
        $TRAD_FIG_ARG \
        --output "$FIGURES_DIR" \
        --sample_trials $SAMPLE_TRIALS
else
    echo ""
    echo "[4/4] Skipping figure generation"
fi

# =========================================================================
# Summary
# =========================================================================
echo ""
echo "========================================"
echo "Done. Output in $OUTPUT_DIR"
echo "========================================"
echo "  Results:  $RESULTS_DIR/"
echo "  Figures:  $FIGURES_DIR/"
if [ -d "$FIGURES_DIR" ]; then
    for f in "$FIGURES_DIR"/*.pdf "$FIGURES_DIR"/*.png; do
        [ -f "$f" ] && echo "    - $(basename "$f")"
    done
fi

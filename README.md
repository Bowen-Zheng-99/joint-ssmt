# Joint SSMT

Bayesian state-space model for joint inference of oscillatory dynamics and point-process coupling.

Joint SSMT treats narrowband LFP activity as a latent process evolving in continuous time, with spike trains linked to the complex spectral state through a Bernoulli-logistic model. It jointly infers LFP spectrograms and spike-field coupling strength using Polya-Gamma augmentation and Kalman smoothing.

## Installation

We recommend running Joint SSMT with GPU acceleration. The default install below includes CUDA 12 support.

### Using conda

```bash
conda create -n ssmt python=3.11
conda activate ssmt
pip install "joint-ssmt[gpu] @ git+https://github.com/Bowen-Zheng-99/joint-ssmt.git"
```

### Using uv

```bash
uv venv ssmt --python 3.11
source ssmt/bin/activate
uv pip install "joint-ssmt[gpu] @ git+https://github.com/Bowen-Zheng-99/joint-ssmt.git"
```

### CPU-only install

If you do not have a CUDA-compatible GPU, omit the `[gpu]` extra:

```bash
pip install git+https://github.com/Bowen-Zheng-99/joint-ssmt.git
```

### Development install

```bash
git clone https://github.com/Bowen-Zheng-99/joint-ssmt.git
cd joint-ssmt
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from joint_ssmt import simulate_single_trial, SingleTrialSimConfig
from joint_ssmt import run_inference, load_results

# Simulate data
cfg = SingleTrialSimConfig(duration_sec=10.0, S=3, k_active=2)
data = simulate_single_trial(cfg, seed=42)

# Run inference (generates visualizations automatically)
saved = run_inference(
    lfp=data['LFP'],
    spikes=data['spikes'],
    spectral_config={'freq_min': 1, 'freq_max': 61},
    inference_config={'warmup_iterations': 200, 'n_refresh_cycles': 2},
    output_config={'output_dir': './results'},
    fs=cfg.fs,
    plot=True,
)

# Load and inspect results
results = load_results('./results')
coupling = results['coupling']
print(coupling['beta_mag'].shape)   # (S, J) — effect sizes
print(coupling['wald_pval'].shape)  # (S, J) — p-values
```

## Notebooks

Detailed tutorials and examples are available in the `examples/` directory:

- **quickstart_single_trial.ipynb** — Basic inference workflow for a single trial
- **quickstart_trials.ipynb** — Hierarchical inference across trial-structured data

## Citation

If you use Joint SSMT, please cite:

Zheng, Brincat, Donoghue, Miller, Brown (2026). Bayesian State-Space Model for Joint Inference of Oscillatory Dynamics and Point-Process Coupling. *arXiv preprint*.

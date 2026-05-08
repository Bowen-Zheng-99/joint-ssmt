# Joint SSMT

Bayesian state-space model for joint inference of oscillatory dynamics and point-process coupling.

Joint SSMT treats narrowband LFP activity as a latent process evolving in continuous time, with spike trains linked to the complex spectral state through a Bernoulli-logistic model. It jointly infers the LFP spectrogram and spike-field coupling strength using Polya-Gamma augmentation and Kalman smoothing, and returns posterior credible intervals on both.

## Installation

Joint SSMT is built around JAX. On Linux, `pip` pulls in the CUDA 12 build automatically (JAX falls back to CPU at runtime if no GPU is present). On macOS and Windows, plain CPU JAX is installed. The same install command works on all three.

### conda

```bash
conda create -n ssmt python=3.11
conda activate ssmt
pip install "joint-ssmt @ git+https://github.com/Bowen-Zheng-99/joint-ssmt.git"
```

### uv

```bash
uv venv ssmt --python 3.11
source ssmt/bin/activate
uv pip install "joint-ssmt @ git+https://github.com/Bowen-Zheng-99/joint-ssmt.git"
```

### From a clone

```bash
git clone https://github.com/Bowen-Zheng-99/joint-ssmt.git
cd joint-ssmt
pip install -e .
```

## Verify the install

`joint_ssmt.test()` runs a short end-to-end pipeline (10 s synthetic data, brief inference) and checks the outputs. Takes about a minute on CPU.

```python
from joint_ssmt import test
test()
```

A small bundled dataset (180 s LFP, 2 spike units at 1 kHz) is also shipped with the package, so you can try the full pipeline without supplying your own data:

```python
from joint_ssmt import load_demo_data, run_auto_inference

data = load_demo_data()
run_auto_inference(data['lfp'], data['spikes'], data['fs'], output_dir='./results')
```

## Get the example notebooks

`pip install` only installs the importable Python package. To run the example notebooks, clone the repo:

```bash
git clone https://github.com/Bowen-Zheng-99/joint-ssmt.git
cd joint-ssmt/examples
jupyter notebook
```

Or download just the tutorial notebook:

```bash
curl -O https://raw.githubusercontent.com/Bowen-Zheng-99/joint-ssmt/main/examples/tutorial_notebook.ipynb
```

## Quick start

We highly recommend going through the tutorial notebook to build an intuitive understanding of the algorithm and the main functionality. The two quickstart notebooks walk through the lower-level API for single-trial and trial-structured data, where each config field is exposed and editable.

If you are looking for a "data in, figures out" approach, the auto-inference notebook shows how to use `run_auto_inference()`, which takes in your data, checks a few basic things about its shape and sampling rate, and returns the results and the default figures.

```python
import numpy as np
from joint_ssmt import run_auto_inference

data = np.load('demo_dataset.npz')         # your LFP + spikes + fs
saved = run_auto_inference(
    lfp=data['lfp'],                        # shape (T,)
    spikes=data['spikes'],                  # shape (S, T) of 0/1
    fs=float(data['fs']),
    output_dir='./results',
)
```

For trial-structured data, pass `trial_structure=True` and shape the arrays as `(R, T)` and `(R, S, T)`.

If you want fine-grained control over each config, use the lower-level `run_inference()` and `run_inference_trials()` directly.

## Notebooks

Examples live in the `examples/` directory.

- `auto_inference_demo.ipynb`: shortest runnable example. Loads a small bundled dataset and runs `run_auto_inference()` for both the single-trial and trial-structured cases.
- `quickstart_single_trial.ipynb`: short walkthrough of single-trial inference using the lower-level `run_inference()`.
- `quickstart_trials.ipynb`: same for trial-structured data using `run_inference_trials()`.
- `tutorial_notebook.ipynb`: longer walkthrough that explains the model, the configs, the spectral posterior with credible bands, the coupling-detection heatmap, and a comparison against PLV and SFC. Recommended for first-time users.

## Citation

Zheng, Brincat, Donoghue, Miller, Brown (2026). Bayesian State-Space Model for Joint Inference of Oscillatory Dynamics and Point-Process Coupling. *arXiv preprint*.

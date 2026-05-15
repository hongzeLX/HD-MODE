# Learning high-dimensional PDEs from sparse and noisy observations via HD-MODE

## Overview

This repository implements an experimental framework for high-dimensional partial differential equation (PDE) modeling and neural inverse problem solving. The core idea is to fit the state variable `u` with a neural network while simultaneously learning a physics generator `phi`, enabling synergistic training from both data and physics.

Supported PDE types:
- `Fokker_Planck`
- `Allen_Cahn`
- `Zeldovich`
- `Fisher_KPP`
- `KPZ`

Supported dimensionalities:
- `20D`
- `50D`
- `100D`

The system also supports sensor data simulation at multiple noise levels.

## Repository Structure

- `data.py`
  - Defines the PDE dataset class `RealWorldPDEDataset`
  - Provides analytical solutions `u_exact` and physics generator `phi_exact`
  - Generates random training/test samples with optional noise

- `shotgun.py`
  - Defines the neural networks `Net_U` and `GeneratorNN`
  - Implements the `ShotgunInverseModel`
  - Computes data loss, physics extraction loss, and synergistic loss

- `main_experiment.py`
  - Main pipeline script
  - Runs experiments across PDE types, dimensions, and noise levels
  - Saves experiment metrics to `results/experiment_metrics_log.csv`

- `Numerical_Experiment_1.py`
  - Produces error analysis visualizations from logged results
  - Outputs `results/Error_Analysis.pdf`

- `Numerical_Experiment_2.py`
  - Builds standardized correlation heatmaps from `.npz` inference data
  - Outputs `results/Standardized_Heatmaps.pdf`

- `Numerical_Experiment_3_and_4.py`
  - Creates large dashboard figures for physics response and manifold collapse
  - Outputs `results/Physics_Response.pdf` and `results/Manifold_Collapse.pdf`

- `results/`
  - `experiment_metrics_log.csv`: logged experiment metrics
  - `plot_data/`: stored inference results and visualization data in `.npz` format

## Dependencies

Recommended Python version: `3.8+`

Required packages:
- `torch`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `psutil`
- `einops`

Install with pip:

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn psutil einops
```

Using Conda:

```bash
conda create -n hd_mode python=3.11
conda activate hd_mode
pip install torch numpy pandas matplotlib seaborn scikit-learn psutil einops
```

## Quick Start

### 1. Run the main experiment

```bash
cd ./HD-MODE
python main_experiment.py
```

To override the default GPU ID (`0`):

```bash
python main_experiment.py 1
```

### 2. Inspect experiment outputs

After the main experiment completes, it will generate:

- `results/experiment_metrics_log.csv`
- `results/plot_data/data_<Equation>_<D>D_noise_<N>.npz`

### 3. Generate error analysis figures

```bash
python Numerical_Experiment_1.py
```

Output:

- `results/Error_Analysis.pdf`

### 4. Generate standardized heatmaps

```bash
python Numerical_Experiment_2.py
```

Output:

- `results/Standardized_Heatmaps.pdf`

### 5. Generate physics response and manifold collapse charts

```bash
python Numerical_Experiment_3_and_4.py
```

Output:

- `results/Physics_Response.pdf`
- `results/Manifold_Collapse.pdf`

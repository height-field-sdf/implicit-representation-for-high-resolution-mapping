# HF-SDF: Height-Field Signed Distance Function

[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41612--025--01044--6-blue)](https://www.nature.com/articles/s41612-025-01044-6)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://height-field-sdf.github.io/)

Official implementation of the paper **"Adaptive high-resolution mapping of air pollution with a novel implicit 3D representation approach"**, published in *npj Climate and Atmospheric Science* (Volume 8, Article 180, 2025).

HF-SDF introduces a novel 3D implicit representation that models air pollution concentration as a continuous height-field surface in an abstract 3D space defined by (longitude, latitude, pollutant density). An 8-layer MLP with skip connections and geometric initialization learns to represent these surfaces as signed distance functions, and a latent autodecoder compresses multiple pollution maps into compact latent codes.

## Overview

The height-field formulation treats pollution concentration as the Z-axis height, forming a surface in (lon, lat, concentration) space. The signed distance to this surface is learned by an implicit neural network, enabling:
- **Self-supervised reconstruction** from sparse or coarse observations without handcrafted geographic predictors
- **Strong transferability** across unseen regions and different pollutant species
- **Flexible resolution** — reconstruct fine-scale maps (up to 1 km) from coarse inputs (up to 40 km)

### Key Features

- **Single-surface reconstruction**: Fit an implicit neural representation to a single pollution map
- **Multi-surface shape space**: Learn a shared decoder with per-timestep latent codes 

## Project Structure

```
brief_hfSDF/
├── model/                    
│   ├── network.py            
│   └── sample.py             
├── utils/                    
│   ├── general.py            
│   └── plots.py              
├── datasets/                 
│   └── dfaustdataset.py      
├── reconstruction/           
│   ├── run.py                
│   └── setup.conf            
├── shapespace/                
│   ├── train.py              
│   ├── eval.py               
│   ├── latent_optimizer.py   
│   └── dfaust_setup.conf     
├── scripts/                  
├── splits/                   
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/height-field-sdf/implicit-representation-for-high-resolution-mapping.git

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

## Usage

### 1. Data Preprocessing

The input data consists of air pollution concentration maps (e.g., PM2.5, NO₂) in CSV or NetCDF format with columns for longitude, latitude, and pollutant concentration. Preprocessing converts these into normalized NumPy arrays (`.npy`) by:

- Filtering data to a geographical domain of interest
- Applying min-max normalization to the [-1, 1] range
- Splitting into training and test sets
- Optionally saving min/max statistics for later denormalization

The resulting `.npy` files serve as point clouds where X = longitude, Y = latitude, Z = normalized concentration. These are fed into the HF-SDF reconstruction or shape space training pipelines.

### 2. Evaluation with Pretrained Checkpoint

A pretrained model checkpoint (epoch 10000, trained on PM2.5 reanalysis data over Chinese cities) is provided at `checkpoints/ModelParameters/latest.pth`. The model uses `shapespace/dfaust_setup.conf` configuration.

```bash
python shapespace/eval.py \
    --exp-name my_eval \
    --conf shapespace/dfaust_setup.conf \
    --split splits/your_test_split.json \
    --checkpoint 10000 \
    --gpu 0
```

### 3. Single-Map Reconstruction

Train an implicit neural representation for a single pollution map:

```bash
python reconstruction/run.py \
    --conf reconstruction/setup.conf \
    --expname my_experiment \
    --points_batch 16384 \
    --nepoch 10000 \
    --gpu 0
```

### 4. Multi-Maps Reconstruction (Latent Space)

Train a shared decoder with latent codes across multiple pollution maps:

```bash
python shapespace/train.py \
    --conf shapespace/dfaust_setup.conf \
    --expname shapespace_experiment \
    --batch_size 16 \
    --points_batch 4096 \
    --nepoch 10000 \
    --gpu 0 \
    --split splits/your_split.json
```

Evaluate a trained model on new data with latent optimization:

```bash
python shapespace/eval.py \
    --exp-name shapespace_experiment \
    --conf shapespace/dfaust_setup.conf \
    --split splits/your_test_split.json \
    --checkpoint latest \
    --gpu 0
```

## Model Architecture

### Loss Functions

- **Manifold loss**: |f(x)| for surface points → encourages zero-crossing at the surface
- **Eikonal loss (partial Z)**: |∂f/∂z - 1|² → constrains gradient magnitude along Z-axis
- **Bias loss**: |f(x_bias) - gt_bias| → predicts SDF at offset positions
- **Latent regularization**: ||z||₂ → prevents latent codes from growing unbounded
- **Normal loss (optional)**: matches predicted gradient to ground-truth normals

## Configuration

All training parameters are controlled via HOCON (.conf) files. Key sections:

- `train` — dataset path, latent size, learning rate schedule, network class
- `network.inputs` — MLP architecture (dims, skip connections, initialization)
- `network.sampler` — point sampling strategy and parameters
- `network.loss` — loss weights (lambda, normals_lambda, latent_lambda)
- `plot` — visualization settings (resolution, output formats)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{zhang2025adaptive,
  title={Adaptive high-resolution mapping of air pollution with a novel implicit 3D representation approach},
  author={Zhang, Ting and Zheng, Bo and Huang, Ruqi},
  journal={npj Climate and Atmospheric Science},
  volume={8},
  number={1},
  pages={180},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

Paper: [https://www.nature.com/articles/s41612-025-01044-6](https://www.nature.com/articles/s41612-025-01044-6)

## License

This project is for research purposes. Please contact the author for usage permissions.

## Related Papers

- [IGR (Implicit Geometric Regularization)](https://arxiv.org/abs/2002.10099) — Gropp et al., CVPR 2020. Learning signed distance functions from raw point clouds with eikonal constraint.
- [DeepSDF](https://arxiv.org/abs/1901.05103) — Park et al., CVPR 2019. Learning continuous signed distance functions for shape representation with latent autodecoder.

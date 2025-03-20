# Power Flow Calculation undr Incomplete Topology

![Power Grid Network](https://i.imgur.com/placeholder-image.png)

## Overview

This project implements a graph neural network framework for temporal power grid analysis. It uses a temporal graph transformer architecture to predict hidden nodes in power grid networks while respecting both electrical physics constraints and temporal consistency.

The system can help identify missing or hidden nodes in power grids, predict their electrical properties, and analyze how these properties evolve over time.

## Key Features

- **Physics-Constrained Graph Neural Network**: Incorporates power flow constraints into the learning process
- **Temporal Consistency**: Ensures predictions maintain consistency across time steps
- **Node Existence Prediction**: Identifies potential hidden nodes in the power grid
- **Electrical Property Prediction**: Estimates voltage and power flow parameters

## Architecture

The system consists of several key components:

1. **Dataset Processing**: Time-series power grid data with strategic node hiding
2. **Temporal Graphormer**: Graph transformer with multi-head attention for both spatial and temporal relationships
3. **Virtual Node Predictor**: Predicts existence and properties of hidden nodes
4. **Physics Constraint Module**: Enforces Kirchhoff's laws and power flow equations
5. **Visualization Tools**: Network visualizations and temporal prediction analysis

## Model Details

The core of the system is the `TemporalVirtualNodePredictor` which uses a graph transformer encoder with temporal awareness:

```
                      ┌─────────────────┐
                      │ Input Features  │
                      └────────┬────────┘
                               │
                      ┌────────▼────────┐
           ┌──────────┤ Temporal Graph  │◄─────┐
Time Steps │          │   Transformer   │      │ Edge Features
           └──────────►                 │◄─────┘
                      └────────┬────────┘
                               │
                   ┌───────────┴───────────┐
                   │                       │
          ┌────────▼────────┐    ┌─────────▼─────────┐
          │  Node Existence │    │ Node Feature Pred │
          │    Prediction   │    │ (Voltage, Power)  │
          └─────────────────┘    └───────────────────┘
```

## Results

The model achieves:

- High accuracy in predicting hidden node existence
- Physically plausible electrical parameter estimations 
- Temporally consistent predictions across multiple time steps

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/temporal-power-grid-analysis.git
cd temporal-power-grid-analysis

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- NumPy
- Matplotlib
- NetworkX
- pandas

## Usage

### Dataset Preparation

Place your power grid measurement data in the following format:
- `Vmeas_data_*.csv` - Voltage measurements
- `Imeas_data_*.csv` - Current measurements

### Training

```bash
python main.py --train_dir ./train_data/Node_666 --test_dir ./test_data/Node_666 --mode train --epochs 3000
```

### Testing

```bash
python main.py --train_dir ./train_data/Node_666 --test_dir ./test_data/Node_666 --mode test
```

### Parameters

- `--train_dir`: Directory containing training data
- `--test_dir`: Directory containing test data
- `--mode`: Operation mode (train, test, or both)
- `--epochs`: Number of training epochs
- `--learning_rate`: Learning rate for optimization
- `--hide_ratio`: Ratio of nodes to hide (default: 0.2)
- `--sequence_length`: Number of consecutive time steps (default: 3)
- `--max_time_steps`: Maximum time steps in dataset (default: 24)

## Visualization Examples

The system generates visualizations of predicted nodes and their properties:

![Node Predictions](https://i.imgur.com/placeholder-image-2.png)

## Performance Optimization

For faster training, consider:

1. Adjusting batch size based on available memory
2. Using mixed precision training
3. Setting `shuffle=False` for more stable convergence with time-series data
4. Implementing gradient accumulation for effective larger batch sizes
5. Using multiple workers for data loading

## Citation

If you use this code for your research, please cite:

```
awaiting...
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors and researchers in the field of graph neural networks and power systems
- Special thanks to the PyTorch Geometric team for their excellent library

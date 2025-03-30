# LilResNets 🚀
> ResNets that are light on resources, heavy on performance!

## Project Overview
LilResNets is a comprehensive study of lightweight ResNet variants, all under 5M parameters, designed for resource-constrained environments. Through extensive ablation studies, I explore the impact of various network components while maintaining high accuracy and efficiency.

### Key Features
- **Lightweight Models**: All variants under 5M parameters
- **Extensive Experimentation**: 9 different model architectures
- **Comprehensive Training Techniques**: Multiple data augmentations, optimizers, and schedulers
- **End-to-end Monitoring**: Detailed logging, visualization, and metrics tracking
- **Resource Efficient**: Optimized for environments with limited computational resources

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LilResNets.git
cd LilResNets
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Downloaded CIFAR-10 dataset is present in data folder:
```bash
python data/cifar-10-python/*
```

## 🚀 Usage

Run any model configuration using:
```bash
python main.py --config_name <Model_Config_Name>
```

Available model configurations:
- DeepSlim
- MyResNet
- PreNet
- PreNet2
- PreNet3
- ResMix
- ResNetX
- ResReg
- ResWide

## 🎯 Model Architecture

### Base Architecture
All models are built on the ResNet architecture with the following key components:
- Residual blocks with varying configurations
- Squeeze-and-Excitation blocks (optional)
- Dropout layers (optional)
- Average pooling
- Fully connected classification layer

### Model Variants
I experimented with 9 different architectures, each with unique characteristics:

| Model | Parameters | Key Features |
|-------|------------|--------------|
| DeepSlim | 4.85M | Balanced block distribution [6,4,3] |
| MyResNet | 4.77M | Optimized for CIFAR-10 [5,4,3] |
| PreNet | 4.40M | Progressive kernel sizes [5,3,3] |
| PreNet2 | 4.40M | Enhanced dropout (0.1) |
| PreNet3 | 3.70M | Four-block architecture |
| ResMix | 4.62M | Mixed kernel configurations |
| ResNetX | 4.69M | Balanced architecture |
| ResReg | 3.76M | Regularized design |
| ResWide | 3.29M | Wide channels (90) |

## 🔧 Training Components

### Data Augmentation
I implemented and tested multiple augmentation strategies:

1. **CutMix**
   - Replaces a patch from one image with another
   - α = 1.0 for uniform mixing ratio distribution
   - Improves model robustness and generalization

2. **MixUp**
   - Linear interpolation between images and labels
   - α = 0.5 for balanced mixing
   - Encourages smoother decision boundaries

3. **AutoAugment**
   - Policy-based augmentation
   - Automatically learned augmentation strategies
   - Particularly effective for CIFAR-10

4. **RandAugment**
   - Random selection of augmentation operations
   - Simpler than AutoAugment but equally effective
   - More computationally efficient

### Optimizers
I experimented with various optimization strategies:

1. **AdamW**
   - Adaptive moment estimation with weight decay
   - Particularly effective for deep networks
   - Default learning rate: 0.09

2. **SGD with Momentum**
   - Classical stochastic gradient descent
   - Momentum: 0.9
   - Weight decay: 0.0005-0.001

3. **RAdam**
   - Rectified Adam optimizer
   - Better convergence properties
   - Learning rate: 0.1

### Learning Rate Schedulers
Multiple scheduling strategies were tested:

1. **CosineAnnealingWarmRestarts**
   - Cyclical learning rate with warm restarts
   - Helps escape local minima
   - T_0: 10, T_mult: 2

2. **CosineAnnealingLR**
   - Smooth cosine decay
   - Better final convergence
   - T_max: 200

3. **OneCycleLR**
   - One cycle policy
   - Faster convergence
   - max_lr: 0.1

4. **ReduceLROnPlateau**
   - Adaptive learning rate reduction
   - Patience: 5
   - Factor: 0.5

## 📈 Results

### Training Metrics
- Training accuracy: Up to 99.98%
- Test accuracy: Up to 95.47%
- Early stopping patience: 12-40 epochs
- Minimum delta: 0.0001-0.0009

### Key Findings
1. **Architecture Impact**
   - Block distribution significantly affects performance
   - Wider channels (90) can be more effective than deeper networks
   - Squeeze-and-Excitation blocks improve efficiency

2. **Training Strategy**
   - CutMix + MixUp combination provides best regularization
   - Label smoothing (0.05-0.1) improves generalization
   - Gradient clipping (0.05-0.1) stabilizes training

3. **Optimization**
   - AdamW with CosineAnnealingWarmRestarts works best for most models
   - SGD with momentum provides better final accuracy
   - OneCycleLR achieves faster convergence

## 📁 Project Structure
```
LilResNets/
├── configs/                 # Model configuration files
├── core/                    # Core implementation
│   ├── models/             # Model architectures
│   ├── dataset.py          # Data loading and augmentation
│   ├── train.py            # Training logic
│   └── utils.py            # Utility functions
├── experiments/            # Training results
│   └── <model_name>/       # Per-model results
│       ├── metrics.csv     # Training metrics
│       ├── learning_curve.png
│       └── lr_schedule.png
├── data/                   # Dataset directory
└── main.py                 # Main entry point
```

## 📊 Results Visualization
Each experiment generates:
- Training/validation accuracy curves
- Learning rate schedule visualization
- Detailed metrics CSV file
- Training logs

Results are saved in `experiments/<model_name>/` directory.


## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

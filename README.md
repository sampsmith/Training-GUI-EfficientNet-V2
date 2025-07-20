# ISPM 15 Defect Classifier - GUI Applications

This folder contains user-friendly GUI applications for training and using ISPM 15 defect classification models.

## 📁 Folder Structure

```
GUI_Applications/
├── launcher.py              # Main launcher application
├── gui_app.py               # Training GUI application
├── gui_inference.py         # Inference GUI application
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd GUI_Applications
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
python launcher.py
```

This will open the main launcher window where you can choose between:
- **Training GUI**: Train new models with your dataset
- **Inference GUI**: Use trained models for predictions
- **Command Line Tools**: Access the underlying scripts directly

## 🎯 Applications Overview

### 📊 Training GUI (`gui_app.py`)
**Purpose**: Train ISPM 15 defect classification models with a user-friendly interface.

**Features**:
- **Dataset Management**: Load and configure your training dataset
- **Training Configuration**: Adjust epochs, batch size, learning rate, dropout
- **Data Augmentation**: Toggle advanced augmentation techniques
- **Real-time Monitoring**: Live training progress and metrics
- **Model Evaluation**: Comprehensive evaluation with confusion matrix
- **Logging**: Detailed training logs and history

**Usage**:
1. Launch from launcher or run `python gui_app.py`
2. Select your dataset directory (must have train/val/test subfolders)
3. Configure training parameters
4. Choose model save location
5. Start training and monitor progress

### 🔍 Inference GUI (`gui_inference.py`)
**Purpose**: Use trained models to classify new ISPM 15 images.

**Features**:
- **Model Loading**: Load trained models with metadata
- **Single Image Classification**: Classify individual images with confidence scores
- **Batch Processing**: Process entire directories of images
- **Results Visualization**: View predictions with confidence thresholds
- **Statistics**: Generate comprehensive analysis of results
- **Export Results**: Save results in JSON format

**Usage**:
1. Launch from launcher or run `python gui_inference.py`
2. Load a trained model (.pth file)
3. Choose single image or batch processing
4. Set confidence thresholds
5. View results and export as needed

### 🎮 Launcher (`launcher.py`)
**Purpose**: Central hub to access all applications and tools.

**Features**:
- **Application Selection**: Choose between training and inference
- **Script Access**: Open command-line scripts in text editor
- **Status Monitoring**: Real-time status updates
- **Cross-platform**: Works on Windows, macOS, and Linux

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 2GB free space for models and dependencies
- **GPU**: Optional but recommended (MPS for Apple Silicon, CUDA for NVIDIA)

### Python Dependencies
- `torch>=2.0.0` - PyTorch deep learning framework
- `torchvision>=0.15.0` - Computer vision utilities
- `numpy>=1.21.0` - Numerical computing
- `scikit-learn>=1.0.0` - Machine learning utilities
- `matplotlib>=3.5.0` - Plotting and visualization
- `seaborn>=0.11.0` - Statistical visualization
- `tqdm>=4.62.0` - Progress bars
- `Pillow>=8.3.0` - Image processing

## 🛠️ Installation Guide

### Step 1: Clone or Download
Make sure you have the complete project folder with the `GUI_Applications` directory.

### Step 2: Install Python Dependencies
```bash
cd GUI_Applications
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python launcher.py
```

If the launcher opens successfully, you're ready to go!

## 📖 Detailed Usage Guide

### Training a New Model

1. **Launch Training GUI**
   - Run `python launcher.py` and click "Launch Training GUI"
   - Or run `python gui_app.py` directly

2. **Load Dataset**
   - Click "Browse" to select your dataset directory
   - Your dataset should have this structure:
     ```
     dataset/
     ├── train/
     │   ├── good/
     │   └── bad/
     ├── val/
     │   ├── good/
     │   └── bad/
     └── test/
         ├── good/
         └── bad/
     ```
   - Click "Load Dataset" to verify and analyze your data

3. **Configure Training**
   - Set model save path
   - Adjust training parameters:
     - **Epochs**: Number of training cycles (default: 50)
     - **Batch Size**: Images per batch (default: 32)
     - **Learning Rate**: Training speed (default: 0.0001)
     - **Dropout Rate**: Regularization (default: 0.4)
   - Enable/disable data augmentation

4. **Start Training**
   - Click "Start Training"
   - Monitor progress in the Logs tab
   - Training will automatically save the best model

### Using a Trained Model

1. **Launch Inference GUI**
   - Run `python launcher.py` and click "Launch Inference GUI"
   - Or run `python gui_inference.py` directly

2. **Load Model**
   - Click "Browse" to select your trained model (.pth file)
   - Click "Load Model" to initialize the classifier

3. **Single Image Classification**
   - Go to "Single Image" tab
   - Select an image file
   - Set confidence threshold
   - Click "Classify Image"
   - View results with confidence scores

4. **Batch Processing**
   - Go to "Batch Processing" tab
   - Select directory containing images
   - Set output file path
   - Click "Process Batch"
   - View summary and save results

## 🔧 Troubleshooting

### Common Issues

**"Module not found" errors**
```bash
pip install -r requirements.txt
```

**CUDA/MPS not available**
- The application will automatically fall back to CPU
- Training will be slower but still functional

**Memory errors during training**
- Reduce batch size (try 16 or 8)
- Close other applications
- Use smaller image sizes

**Dataset loading errors**
- Ensure your dataset follows the required structure
- Check file permissions
- Verify image formats (jpg, png, etc.)

### Performance Tips

1. **For Training**:
   - Use GPU acceleration when available
   - Adjust batch size based on available memory
   - Enable data augmentation for better generalization

2. **For Inference**:
   - Use batch processing for multiple images
   - Set appropriate confidence thresholds
   - Save results for later analysis

## 📊 Model Architecture

The GUI applications use the same improved model architecture:

- **Backbone**: EfficientNetV2-M (pretrained on ImageNet)
- **Classifier**: 3-layer MLP with batch normalization
- **Loss Function**: Focal Loss for class imbalance
- **Optimizer**: AdamW with weight decay
- **Scheduler**: OneCycleLR for optimal convergence

## 🔄 Integration with Command Line

The GUI applications are built on top of the command-line scripts:
- `scripts/train_improved.py` - Training script
- `scripts/inference.py` - Inference script

You can access these directly from the launcher or use them independently.

## 📝 License and Support

This GUI application is part of the ISPM 15 Defect Classifier project. For support or questions, please refer to the main project documentation.

## 🆕 Updates and Improvements

The GUI applications include several improvements over the command-line versions:
- Real-time progress monitoring
- Interactive parameter adjustment
- Visual result analysis
- Batch processing capabilities
- Comprehensive logging and error handling 

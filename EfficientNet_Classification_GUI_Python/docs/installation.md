# Installation Guide 📦

This guide will help you install and set up the ISPM 15 Defect Classifier project on your system.

## 📋 Prerequisites

### System Requirements

- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 2GB free space for models and dependencies
- **GPU**: Optional but recommended
  - **NVIDIA**: CUDA 11.0+ with compatible drivers
  - **Apple Silicon**: macOS 12.0+ (MPS support)
  - **AMD**: ROCm support (experimental)

### Python Environment

We recommend using a virtual environment to avoid conflicts with other projects:

```bash
# Create virtual environment
python -m venv ispm_env

# Activate virtual environment
# On Windows:
ispm_env\Scripts\activate
# On macOS/Linux:
source ispm_env/bin/activate
```

## 🚀 Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/ISPM_Defect_Classifier.git
cd ISPM_Defect_Classifier
```

### Step 2: Install Core Dependencies

```bash
# Install main project dependencies
pip install -r requirements.txt
```

### Step 3: Install GUI Dependencies

```bash
# Navigate to GUI applications
cd GUI_Applications

# Install GUI-specific dependencies
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Test core functionality
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision version: {torchvision.__version__}')"

# Test GUI launcher
python launcher.py
```

## 🔧 GPU Setup (Optional)

### NVIDIA GPU (CUDA)

1. **Install NVIDIA Drivers**
   - Download from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
   - Install appropriate drivers for your GPU

2. **Install CUDA Toolkit**
   - Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Install CUDA 11.0 or higher

3. **Install PyTorch with CUDA**
   ```bash
   # Uninstall CPU version first
   pip uninstall torch torchvision
   
   # Install CUDA version (adjust CUDA version as needed)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Verify CUDA Installation**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')"
   ```

### Apple Silicon (MPS)

1. **Update macOS**
   - Ensure you're running macOS 12.0 or higher

2. **Install PyTorch with MPS Support**
   ```bash
   pip install torch torchvision
   ```

3. **Verify MPS Installation**
   ```bash
   python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
   ```

## 📦 Package Managers

### Conda Installation

If you prefer using Conda:

```bash
# Create conda environment
conda create -n ispm_env python=3.9
conda activate ispm_env

# Install PyTorch with conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Docker Installation

For containerized deployment:

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port (if web interface is added)
EXPOSE 8000

# Run the application
CMD ["python", "GUI_Applications/launcher.py"]
```

## 🧪 Development Setup

For developers who want to contribute:

```bash
# Clone repository
git clone https://github.com/yourusername/ISPM_Defect_Classifier.git
cd ISPM_Defect_Classifier

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/
```

## 🔍 Troubleshooting Installation

### Common Issues

**"Module not found" errors**
```bash
# Ensure virtual environment is activated
source ispm_env/bin/activate  # or ispm_env\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**CUDA not available**
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Memory errors during installation**
```bash
# Use pip with memory optimization
pip install -r requirements.txt --no-cache-dir

# Or install packages one by one
pip install torch
pip install torchvision
pip install numpy scikit-learn matplotlib seaborn
```

**Permission errors on Linux/macOS**
```bash
# Use user installation
pip install --user -r requirements.txt

# Or fix permissions
sudo chown -R $USER:$USER ~/.local/lib/python3.*/site-packages/
```

### Platform-Specific Issues

**Windows**
- Ensure Visual C++ Redistributable is installed
- Use Windows Subsystem for Linux (WSL) for better compatibility

**macOS**
- Install Xcode Command Line Tools: `xcode-select --install`
- Use Homebrew for system dependencies: `brew install libpng`

**Linux (Ubuntu/Debian)**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip python3-venv
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
```

## ✅ Verification Checklist

After installation, verify everything works:

- [ ] Python 3.8+ installed and accessible
- [ ] Virtual environment created and activated
- [ ] All dependencies installed without errors
- [ ] PyTorch installed and CUDA/MPS available (if applicable)
- [ ] GUI launcher opens successfully
- [ ] Training GUI loads without errors
- [ ] Inference GUI loads without errors
- [ ] Sample dataset can be loaded
- [ ] Model training starts without errors

## 📞 Getting Help

If you encounter issues during installation:

1. **Check the troubleshooting section above**
2. **Search existing issues** on GitHub
3. **Create a new issue** with:
   - Operating system and version
   - Python version
   - Error messages
   - Steps to reproduce
   - System specifications

## 🔄 Updating

To update to the latest version:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# For GUI applications
cd GUI_Applications
pip install -r requirements.txt --upgrade
``` 
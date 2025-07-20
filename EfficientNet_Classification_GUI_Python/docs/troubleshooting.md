# Troubleshooting Guide 🔧

This guide helps you resolve common issues when using the ISPM 15 Defect Classifier.

## 🚨 Common Issues

### Installation Problems

#### "Module not found" Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'torch'
ModuleNotFoundError: No module named 'torchvision'
```

**Solutions:**
```bash
# 1. Ensure virtual environment is activated
source ispm_env/bin/activate  # Linux/macOS
ispm_env\Scripts\activate     # Windows

# 2. Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# 3. Install PyTorch separately if needed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### CUDA/MPS Not Available

**Symptoms:**
```
CUDA not available, using CPU
MPS not available, using CPU
```

**Solutions:**
```bash
# Check CUDA installation
nvidia-smi

# Check PyTorch CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For Apple Silicon
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

#### Memory Errors During Installation

**Symptoms:**
```
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
```

**Solutions:**
```bash
# 1. Clear pip cache
pip cache purge

# 2. Install with no cache
pip install -r requirements.txt --no-cache-dir

# 3. Install packages one by one
pip install torch
pip install torchvision
pip install numpy scikit-learn matplotlib seaborn
```

### Training Issues

#### CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**
```python
# 1. Reduce batch size
batch_size = 16  # or 8 for limited memory

# 2. Reduce image size
transforms.Resize((192, 192))  # instead of 224x224

# 3. Use gradient accumulation
accumulation_steps = 4
```

**GUI Solution:**
- Reduce "Batch Size" in Training tab
- Try 16 or 8 instead of 32

#### Training Loss Not Decreasing

**Symptoms:**
- Loss stays constant or increases
- Accuracy doesn't improve
- Model predictions are random

**Solutions:**
```python
# 1. Check learning rate
learning_rate = 0.0001  # Try 0.001 or 0.00001

# 2. Verify data loading
print(f"Dataset size: {len(dataset)}")
print(f"Class distribution: {Counter(dataset.targets)}")

# 3. Check data augmentation
# Disable augmentation temporarily
use_augmentation = False
```

**GUI Solution:**
- Reduce "Learning Rate" to 0.00001
- Disable "Data Augmentation" temporarily
- Check dataset information in Dataset tab

#### Overfitting

**Symptoms:**
- Training accuracy much higher than validation accuracy
- Validation loss increases while training loss decreases
- Poor performance on test set

**Solutions:**
```python
# 1. Increase dropout
dropout_rate = 0.5  # or 0.6

# 2. Add more regularization
weight_decay = 1e-3  # increase from 1e-4

# 3. Reduce model complexity
# Use smaller backbone or fewer classifier layers
```

**GUI Solution:**
- Increase "Dropout Rate" to 0.5-0.6
- Enable "Data Augmentation"
- Reduce number of epochs

#### Underfitting

**Symptoms:**
- Both training and validation accuracy are low
- Model doesn't learn patterns
- Poor performance overall

**Solutions:**
```python
# 1. Increase model capacity
# Use larger backbone or more classifier layers

# 2. Reduce regularization
dropout_rate = 0.2  # or 0.3

# 3. Increase training time
epochs = 100  # or more

# 4. Check data quality
# Ensure images are clear and properly labeled
```

**GUI Solution:**
- Increase "Epochs" to 100+
- Decrease "Dropout Rate" to 0.2-0.3
- Check dataset quality and size

### Dataset Issues

#### Dataset Loading Errors

**Symptoms:**
```
RuntimeError: Found 0 files in subfolders of
FileNotFoundError: [Errno 2] No such file or directory
```

**Solutions:**
```bash
# 1. Verify dataset structure
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

# 2. Check file permissions
ls -la /path/to/dataset

# 3. Verify image formats
# Supported: .jpg, .jpeg, .png, .bmp, .tiff
```

#### Class Imbalance Issues

**Symptoms:**
- Model always predicts the same class
- Poor recall for minority class
- High accuracy but low F1-score

**Solutions:**
```python
# 1. Check class distribution
class_counts = Counter(dataset.targets)
print(f"Class distribution: {class_counts}")

# 2. Use weighted sampling (already implemented)
# 3. Adjust focal loss parameters
focal_loss = FocalLoss(alpha=0.25, gamma=3)

# 4. Collect more data for minority class
```

### Inference Issues

#### Model Loading Errors

**Symptoms:**
```
RuntimeError: Error(s) in loading state_dict
KeyError: 'model_state_dict'
```

**Solutions:**
```python
# 1. Check model file format
# Ensure it's a .pth file saved by this application

# 2. Verify model loading code
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# 3. Check PyTorch version compatibility
# Models saved with different PyTorch versions may not be compatible
```

#### Prediction Errors

**Symptoms:**
- All predictions have same confidence
- Predictions don't make sense
- Model outputs errors

**Solutions:**
```python
# 1. Check input preprocessing
# Ensure images are properly normalized
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# 2. Verify model is in eval mode
model.eval()

# 3. Check input image format
# Should be RGB, 224x224, normalized
```

### GUI Issues

#### GUI Not Opening

**Symptoms:**
- Launcher doesn't start
- GUI windows don't appear
- Tkinter errors

**Solutions:**
```bash
# 1. Check Python installation
python --version

# 2. Install tkinter (Linux)
sudo apt-get install python3-tk

# 3. Run with verbose output
python -v launcher.py

# 4. Check display settings (Linux)
export DISPLAY=:0
```

#### GUI Freezing During Training

**Symptoms:**
- GUI becomes unresponsive
- Progress bars don't update
- Can't stop training

**Solutions:**
```python
# 1. Ensure training runs in separate thread
training_thread = threading.Thread(target=self.train_model)
training_thread.daemon = True

# 2. Update GUI from main thread
self.root.after(0, lambda: self.progress_var.set(value))

# 3. Add proper error handling
try:
    # training code
except Exception as e:
    self.log_message(f"Training error: {str(e)}")
```

## 🔍 Debugging Techniques

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Add to your code
logging.debug(f"Dataset loaded: {len(dataset)} samples")
logging.debug(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
```

### Check System Resources

```python
# Check GPU memory
import torch
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# Check CPU memory
import psutil
print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
print(f"RAM used: {psutil.virtual_memory().used / 1e9:.1f} GB")
```

### Validate Data Pipeline

```python
# Test data loading
for batch_idx, (inputs, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}: inputs shape {inputs.shape}, labels {labels}")
    if batch_idx >= 2:  # Check first few batches
        break

# Test model forward pass
with torch.no_grad():
    outputs = model(inputs)
    print(f"Output shape: {outputs.shape}")
    print(f"Output range: {outputs.min():.3f} to {outputs.max():.3f}")
```

## 📊 Performance Optimization

### GPU Memory Optimization

```python
# 1. Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 2. Clear cache periodically
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 3. Use gradient checkpointing
model.gradient_checkpointing_enable()
```

### Training Speed Optimization

```python
# 1. Increase num_workers for data loading
DataLoader(dataset, num_workers=4, pin_memory=True)

# 2. Use prefetch factor
DataLoader(dataset, num_workers=4, prefetch_factor=2)

# 3. Enable cudnn benchmarking
torch.backends.cudnn.benchmark = True
```

## 🆘 Getting Help

### Before Asking for Help

1. **Check this troubleshooting guide**
2. **Search existing issues** on GitHub
3. **Try minimal reproduction** with simple dataset
4. **Check system requirements** are met

### When Creating an Issue

Include the following information:

**System Information:**
- Operating system and version
- Python version
- PyTorch version
- GPU model and drivers (if applicable)

**Error Details:**
- Complete error message
- Stack trace
- Steps to reproduce
- Expected vs actual behavior

**Code Context:**
- Relevant code snippets
- Dataset structure
- Model configuration

**Example Issue Template:**
```
**System:**
- OS: Windows 10
- Python: 3.9.7
- PyTorch: 2.0.1
- GPU: NVIDIA RTX 3080

**Error:**
```
RuntimeError: CUDA out of memory
```

**Steps to Reproduce:**
1. Load dataset with 1000 images
2. Set batch size to 32
3. Start training
4. Error occurs after 5 epochs

**Expected Behavior:**
Training should complete without memory errors

**Additional Context:**
- Dataset size: 1000 images
- Image resolution: 224x224
- Batch size: 32
```

### Community Resources

- **GitHub Issues**: [Project Issues](https://github.com/yourusername/ISPM_Defect_Classifier/issues)
- **GitHub Discussions**: [Community Forum](https://github.com/yourusername/ISPM_Defect_Classifier/discussions)
- **PyTorch Forums**: [PyTorch Community](https://discuss.pytorch.org/)
- **Stack Overflow**: Tag with `pytorch`, `deep-learning`

## 🔄 Recovery Procedures

### Training Recovery

```python
# Save checkpoints during training
if epoch % 10 == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'checkpoint_epoch_{epoch}.pth')

# Load checkpoint to resume training
checkpoint = torch.load('checkpoint_epoch_50.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

### Data Recovery

```python
# Backup dataset before modifications
import shutil
shutil.copytree('dataset', 'dataset_backup')

# Validate dataset integrity
for split in ['train', 'val', 'test']:
    for class_name in ['good', 'bad']:
        path = f'dataset/{split}/{class_name}'
        if not os.path.exists(path):
            print(f"Missing: {path}")
```

### Model Recovery

```python
# Save multiple model versions
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f'model_{timestamp}.pth'

# Keep best models
if f1_score > best_f1:
    best_f1 = f1_score
    shutil.copy(model_path, 'best_model.pth')
``` 
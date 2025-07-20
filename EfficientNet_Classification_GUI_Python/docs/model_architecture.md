# Model Architecture Documentation 🧠

This document provides a detailed technical overview of the ISPM 15 Defect Classifier model architecture, implementation details, and design decisions.

## 🏗️ Architecture Overview

The ISPM Defect Classifier uses a transfer learning approach based on EfficientNetV2-M, a state-of-the-art convolutional neural network architecture optimized for both accuracy and computational efficiency.

### High-Level Architecture

```
Input Image (224×224×3)
        ↓
EfficientNetV2-M Backbone (pretrained)
        ↓
Global Average Pooling
        ↓
Flatten
        ↓
Dropout (0.4)
        ↓
Linear(1280 → 512) + ReLU + BatchNorm
        ↓
Dropout (0.4)
        ↓
Linear(512 → 128) + ReLU + BatchNorm
        ↓
Dropout (0.2)
        ↓
Linear(128 → 2)
        ↓
Output: [good_probability, bad_probability]
```

## 🔧 Backbone: EfficientNetV2-M

### Why EfficientNetV2-M?

**Advantages:**
- **Efficiency**: Optimized for both training and inference speed
- **Accuracy**: State-of-the-art performance on ImageNet
- **Scalability**: Good balance between model size and performance
- **Transfer Learning**: Excellent feature extraction capabilities

**Specifications:**
- **Parameters**: ~54 million
- **Input Resolution**: 224×224 pixels
- **Feature Dimensions**: 1280 channels after global pooling
- **Pretrained Weights**: ImageNet-1K

### Backbone Modifications

```python
class ISPMClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.4):
        super(ISPMClassifier, self).__init__()
        
        # Load pretrained EfficientNetV2-M
        self.backbone = efficientnet_v2_m(
            weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1
        )
        
        # Freeze early layers for transfer learning
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False
            
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
```

**Transfer Learning Strategy:**
- **Frozen Layers**: First ~90% of backbone parameters
- **Trainable Layers**: Last 30 layers + custom classifier
- **Rationale**: Preserve general image features, adapt to specific domain

## 🎯 Custom Classifier Head

### Design Philosophy

The custom classifier is designed to:
1. **Reduce Overfitting**: Multiple dropout layers
2. **Stabilize Training**: Batch normalization
3. **Feature Learning**: Gradual dimension reduction
4. **Domain Adaptation**: Learn ISPM-specific features

### Implementation Details

```python
num_features = self.backbone.classifier[1].in_features  # 1280

self.classifier = nn.Sequential(
    # Global pooling and flattening
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    
    # First dense layer
    nn.Dropout(p=dropout_rate),
    nn.Linear(num_features, 512),
    nn.ReLU(inplace=True),
    nn.BatchNorm1d(512),
    
    # Second dense layer
    nn.Dropout(p=dropout_rate),
    nn.Linear(512, 128),
    nn.ReLU(inplace=True),
    nn.BatchNorm1d(128),
    
    # Final classification layer
    nn.Dropout(p=dropout_rate/2),
    nn.Linear(128, num_classes)
)
```

### Layer-by-Layer Analysis

**1. AdaptiveAvgPool2d((1, 1))**
- **Purpose**: Global average pooling
- **Input**: Feature maps from backbone
- **Output**: 1280×1×1 feature maps
- **Benefits**: Reduces spatial dimensions, maintains channel information

**2. Flatten**
- **Purpose**: Convert 3D tensor to 1D
- **Input**: 1280×1×1
- **Output**: 1280-dimensional vector

**3. First Dense Block (1280 → 512)**
```python
nn.Dropout(p=0.4),           # Regularization
nn.Linear(1280, 512),        # Feature reduction
nn.ReLU(inplace=True),       # Non-linearity
nn.BatchNorm1d(512),         # Training stability
```

**4. Second Dense Block (512 → 128)**
```python
nn.Dropout(p=0.4),           # Regularization
nn.Linear(512, 128),         # Further feature reduction
nn.ReLU(inplace=True),       # Non-linearity
nn.BatchNorm1d(128),         # Training stability
```

**5. Output Layer (128 → 2)**
```python
nn.Dropout(p=0.2),           # Reduced dropout for final layer
nn.Linear(128, 2),           # Binary classification
```

## 🎯 Loss Function: Focal Loss

### Why Focal Loss?

**Problem**: Class imbalance in defect detection datasets
- **Good samples**: Often more abundant
- **Bad samples**: Rare but critical to detect

**Solution**: Focal Loss addresses class imbalance and hard example mining

### Implementation

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
```

### Parameters

- **α (alpha)**: Class weighting factor (default: 1.0)
- **γ (gamma)**: Focusing parameter (default: 2.0)
  - γ = 0: Standard cross-entropy
  - γ = 2: Reduces loss for easy examples, focuses on hard ones

### Mathematical Formulation

```
FL(pt) = -αt(1-pt)^γ log(pt)

Where:
- pt = probability of correct class
- αt = class weight for class t
- γ = focusing parameter
```

## ⚡ Optimizer: AdamW

### Configuration

```python
optimizer = optim.AdamW(
    self.model.parameters(),
    lr=self.lr_var.get(),        # Default: 0.0001
    weight_decay=1e-4            # L2 regularization
)
```

### Why AdamW?

**Advantages:**
- **Adaptive Learning Rates**: Different learning rates for different parameters
- **Weight Decay**: Proper L2 regularization implementation
- **Momentum**: Helps escape local minima
- **Robustness**: Works well with various hyperparameters

## 📈 Learning Rate Scheduler: OneCycleLR

### Configuration

```python
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=self.lr_var.get() * 10,  # Peak learning rate
    epochs=self.epochs_var.get(),   # Total epochs
    steps_per_epoch=len(self.dataloaders['train'])
)
```

### OneCycleLR Strategy

**Phase 1: Warm-up (10% of training)**
- Learning rate increases from `lr/10` to `max_lr`
- Helps stabilize early training

**Phase 2: Annealing (90% of training)**
- Learning rate decreases from `max_lr` to `lr/100`
- Fine-tunes model parameters

**Benefits:**
- **Faster Convergence**: Higher learning rates early
- **Better Generalization**: Lower learning rates late
- **Reduced Overfitting**: Cyclical nature prevents overfitting

## 🔄 Data Augmentation Pipeline

### Training Augmentation

```python
train_transform = transforms.Compose([
    # Resize to larger size for random cropping
    transforms.Resize((256, 256)),
    
    # Random crop to final size
    transforms.RandomCrop((224, 224)),
    
    # Geometric augmentations
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    
    # Color augmentations
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, 
            saturation=0.2, hue=0.1
        )
    ], p=0.7),
    
    # Blur augmentation
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    ], p=0.4),
    
    # Affine transformations
    transforms.RandomApply([
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
        )
    ], p=0.5),
    
    # Normalization
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],  # ImageNet means
        [0.229, 0.224, 0.225]   # ImageNet stds
    )
])
```

### Validation/Test Augmentation

```python
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])
```

## ⚖️ Class Imbalance Handling

### Weighted Sampling

```python
# Calculate class weights
class_counts = Counter(image_datasets['train'].targets)
total_samples = len(image_datasets['train'])
class_weights = {
    cls: total_samples / (len(class_counts) * count) 
    for cls, count in class_counts.items()
}

# Create weighted sampler
weights = [class_weights[label] for label in image_datasets['train'].targets]
weighted_sampler = WeightedRandomSampler(
    weights, len(image_datasets['train']), replacement=True
)
```

### Benefits

- **Balanced Training**: Ensures equal representation of classes
- **Better Convergence**: Prevents bias towards majority class
- **Improved Metrics**: Better precision, recall, and F1-score

## 🎯 Training Strategy

### Early Stopping

```python
patience = 10
patience_counter = 0
best_f1 = 0.0

for epoch in range(total_epochs):
    # Training and validation
    if f1 > best_f1:
        best_f1 = f1
        best_model_wts = self.model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        break  # Early stopping
```

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

**Purpose**: Prevents exploding gradients, stabilizes training

## 📊 Model Performance Characteristics

### Computational Requirements

**Training:**
- **GPU Memory**: ~4GB for batch size 32
- **Training Time**: ~2-4 hours for 50 epochs (GPU)
- **CPU Training**: 10-20x slower

**Inference:**
- **GPU**: <100ms per image
- **CPU**: ~500ms per image
- **Memory**: ~2GB model size

### Accuracy Metrics

**Target Performance:**
- **Accuracy**: >95%
- **F1-Score**: >0.94
- **ROC AUC**: >0.96
- **Precision**: >0.93
- **Recall**: >0.95

## 🔧 Customization Options

### Architecture Modifications

**Change Backbone Size:**
```python
# Smaller model (faster, less accurate)
self.backbone = efficientnet_v2_s()

# Larger model (slower, more accurate)
self.backbone = efficientnet_v2_l()
```

**Modify Classifier:**
```python
# Add more layers
self.classifier = nn.Sequential(
    # ... existing layers ...
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.BatchNorm1d(64),
    nn.Linear(64, num_classes)
)
```

**Change Loss Function:**
```python
# Standard cross-entropy
criterion = nn.CrossEntropyLoss()

# Weighted cross-entropy
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Custom focal loss parameters
criterion = FocalLoss(alpha=0.25, gamma=3)
```

### Hyperparameter Tuning

**Learning Rate:**
- **Range**: 1e-5 to 1e-3
- **Default**: 1e-4
- **Tuning**: Reduce by 10x if unstable

**Dropout Rate:**
- **Range**: 0.2 to 0.6
- **Default**: 0.4
- **Tuning**: Increase if overfitting, decrease if underfitting

**Batch Size:**
- **Range**: 8 to 64
- **Default**: 32
- **Tuning**: Limited by GPU memory

## 🚀 Deployment Considerations

### Model Optimization

**Quantization:**
```python
# Post-training quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

**ONNX Export:**
```python
# Export for production deployment
torch.onnx.export(
    model, dummy_input, "ispm_model.onnx",
    export_params=True, opset_version=11
)
```

### Production Requirements

**Hardware:**
- **Minimum**: 4GB RAM, CPU inference
- **Recommended**: 8GB RAM, GPU acceleration
- **Optimal**: 16GB RAM, NVIDIA GPU

**Software:**
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **Dependencies**: See requirements.txt

## 📈 Future Improvements

### Potential Enhancements

1. **Multi-class Classification**: Support for different defect types
2. **Attention Mechanisms**: Visual attention for defect localization
3. **Ensemble Methods**: Combine multiple models for better accuracy
4. **Real-time Processing**: Optimize for video stream processing
5. **Edge Deployment**: Quantization and pruning for mobile devices

### Research Directions

1. **Self-supervised Learning**: Pre-training on unlabeled data
2. **Few-shot Learning**: Adapt to new defect types with minimal data
3. **Domain Adaptation**: Transfer learning between different wood types
4. **Explainable AI**: Generate explanations for predictions 
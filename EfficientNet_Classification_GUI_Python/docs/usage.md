# Usage Guide 📖

This guide will walk you through using the ISPM 15 Defect Classifier for both training and inference.

## 🎮 GUI Applications

### Getting Started with the Launcher

1. **Launch the Application**
   ```bash
   cd GUI_Applications
   python launcher.py
   ```

2. **Choose Your Application**
   - **Training GUI**: Train new models with your dataset
   - **Inference GUI**: Use trained models for predictions
   - **Script Access**: Open command-line tools

### 🏋️ Training GUI Usage

#### Step 1: Dataset Preparation

Your dataset should be organized as follows:

```
your_dataset/
├── train/
│   ├── good/          # Non-defective samples
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── bad/           # Defective samples
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── val/
│   ├── good/
│   └── bad/
└── test/
    ├── good/
    └── bad/
```

**Dataset Requirements:**
- **Image Formats**: JPG, PNG, BMP, TIFF
- **Image Size**: Any size (will be resized automatically)
- **Minimum Images**: 50 per class for training
- **Recommended**: 500+ images per class for good performance

#### Step 2: Launch Training GUI

1. **Open Training GUI**
   - From launcher: Click "Launch Training GUI"
   - Direct: `python gui_app.py`

2. **Load Dataset**
   - Click "Browse" next to "Dataset Path"
   - Select your dataset directory
   - Click "Load Dataset"
   - Review dataset information in the Dataset tab

3. **Configure Training Parameters**

   **Model Save Path**
   - Click "Browse" next to "Model Save Path"
   - Choose where to save your trained model
   - Use `.pth` extension

   **Training Parameters**
   - **Epochs**: Number of training cycles (default: 50)
     - Start with 50 for initial training
     - Increase to 100+ for fine-tuning
   - **Batch Size**: Images per batch (default: 32)
     - Reduce to 16 or 8 if you get memory errors
     - Increase to 64 if you have powerful GPU
   - **Learning Rate**: Training speed (default: 0.0001)
     - Start with default value
     - Reduce by 10x if training is unstable
   - **Dropout Rate**: Regularization (default: 0.4)
     - Increase to 0.5-0.6 if overfitting
     - Decrease to 0.2-0.3 if underfitting

   **Data Augmentation**
   - ✅ **Enable**: Recommended for better generalization
   - ❌ **Disable**: Only if you have very large dataset

#### Step 3: Start Training

1. **Review Configuration**
   - Check all parameters in the Training tab
   - Verify dataset information in Dataset tab

2. **Start Training**
   - Click "Start Training"
   - Monitor progress in the Logs tab
   - Watch real-time metrics in Training tab

3. **Monitor Training**

   **Progress Bars**
   - **Overall Progress**: Complete training progress
   - **Epoch Progress**: Current epoch progress
   - **Batch Progress**: Current batch progress

   **Status Information**
   - **Status Label**: Current training phase
   - **Detailed Status**: Real-time loss and accuracy
   - **Logs Tab**: Detailed training history

4. **Training Completion**
   - Model automatically saves when training completes
   - Best model is saved based on validation F1-score
   - Evaluation results appear in Model tab

#### Step 4: Model Evaluation

1. **Review Results**
   - Check Model tab for evaluation metrics
   - Review confusion matrix
   - Note best F1-score achieved

2. **Interpret Results**
   - **Accuracy**: Overall correct predictions
   - **Precision**: Correct positive predictions
   - **Recall**: Correctly identified defects
   - **F1-Score**: Balanced measure (target metric)
   - **ROC AUC**: Classification confidence

### 🔍 Inference GUI Usage

#### Step 1: Load Trained Model

1. **Launch Inference GUI**
   - From launcher: Click "Launch Inference GUI"
   - Direct: `python gui_inference.py`

2. **Load Model**
   - Click "Browse" next to "Model Path"
   - Select your trained model (.pth file)
   - Click "Load Model"
   - Verify model information appears

#### Step 2: Single Image Classification

1. **Select Image**
   - Go to "Single Image" tab
   - Click "Browse" to select image file
   - Supported formats: JPG, PNG, BMP, TIFF

2. **Configure Settings**
   - **Confidence Threshold**: Minimum confidence for prediction (default: 0.5)
     - Higher threshold = more confident predictions
     - Lower threshold = more predictions but less confident

3. **Classify Image**
   - Click "Classify Image"
   - View results with confidence scores
   - Image preview shows with prediction overlay

#### Step 3: Batch Processing

1. **Select Directory**
   - Go to "Batch Processing" tab
   - Click "Browse" to select image directory
   - All images in directory will be processed

2. **Configure Output**
   - **Output File**: Choose where to save results
   - **Confidence Threshold**: Set minimum confidence
   - **File Format**: JSON or CSV output

3. **Process Images**
   - Click "Process Batch"
   - Monitor progress bar
   - View summary statistics when complete

4. **Review Results**
   - **Summary Tab**: Overall statistics
   - **Results Tab**: Individual image results
   - **Export**: Save results to file

## 💻 Command-Line Usage

### Training Script

```bash
# Basic training
python scripts/train_improved.py \
    --dataset_path /path/to/dataset \
    --model_save_path /path/to/save/model.pth \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.0001

# Advanced training with custom parameters
python scripts/train_improved.py \
    --dataset_path /path/to/dataset \
    --model_save_path /path/to/save/model.pth \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.00005 \
    --dropout_rate 0.5 \
    --use_augmentation \
    --early_stopping_patience 15
```

### Inference Script

```bash
# Single image classification
python scripts/inference.py \
    --model_path /path/to/model.pth \
    --image_path /path/to/image.jpg \
    --confidence_threshold 0.7

# Batch processing
python scripts/inference.py \
    --model_path /path/to/model.pth \
    --image_dir /path/to/images/ \
    --output_file /path/to/results.json \
    --confidence_threshold 0.5
```

### Evaluation Script

```bash
# Evaluate model on test set
python scripts/evaluate.py \
    --model_path /path/to/model.pth \
    --test_dir /path/to/test/dataset \
    --output_file /path/to/evaluation_results.json
```

## 📊 Understanding Results

### Training Metrics

**Loss Values**
- **Training Loss**: Should decrease over time
- **Validation Loss**: Should follow training loss
- **Overfitting**: Validation loss increases while training loss decreases

**Accuracy Metrics**
- **Training Accuracy**: Model performance on training data
- **Validation Accuracy**: Model performance on unseen data
- **Target**: Validation accuracy should be close to training accuracy

**F1-Score**
- **Primary Metric**: Balanced measure of precision and recall
- **Target**: >0.90 for good performance
- **Interpretation**: Higher is better, maximum is 1.0

### Inference Results

**Confidence Scores**
- **Range**: 0.0 to 1.0
- **Interpretation**: Higher = more confident prediction
- **Threshold**: Adjust based on your needs
  - High threshold (0.8+): Fewer predictions, higher confidence
  - Low threshold (0.3-): More predictions, lower confidence

**Prediction Classes**
- **Good**: Non-defective wood packaging
- **Bad**: Defective wood packaging (holes, cracks, etc.)

## 🔧 Advanced Usage

### Hyperparameter Tuning

**Learning Rate Schedule**
```python
# The model uses OneCycleLR scheduler
# Peak learning rate is 10x the base learning rate
# Automatically adjusts throughout training
```

**Data Augmentation Techniques**
- **Random Crop**: 256→224 pixels
- **Random Flip**: Horizontal (50%) and vertical (30%)
- **Random Rotation**: ±15 degrees
- **Color Jittering**: Brightness, contrast, saturation, hue
- **Gaussian Blur**: Random blur effects
- **Random Affine**: Translation and scaling

### Model Customization

**Architecture Modifications**
```python
# In gui_app.py, modify ISPMClassifier class
class ISPMClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.4):
        # Change backbone
        self.backbone = efficientnet_v2_s()  # Smaller model
        # Or
        self.backbone = efficientnet_v2_l()  # Larger model
        
        # Modify classifier layers
        self.classifier = nn.Sequential(
            # Add more layers or change dimensions
        )
```

**Loss Function Customization**
```python
# Modify Focal Loss parameters
criterion = FocalLoss(alpha=0.25, gamma=2)  # Different alpha/gamma
# Or use different loss functions
criterion = nn.CrossEntropyLoss()
```

### Performance Optimization

**GPU Memory Management**
```python
# Reduce batch size if you get CUDA out of memory
batch_size = 16  # or 8 for limited memory

# Use gradient accumulation for larger effective batch size
accumulation_steps = 4  # Effective batch size = batch_size * accumulation_steps
```

**Multi-GPU Training**
```python
# The model automatically uses available GPUs
# For explicit multi-GPU usage:
model = nn.DataParallel(model)
```

## 📈 Best Practices

### Dataset Preparation
1. **Balance Classes**: Ensure similar number of good/bad samples
2. **Quality Images**: Use clear, well-lit images
3. **Variety**: Include different lighting, angles, backgrounds
4. **Validation**: Set aside 20% for validation, 10% for testing

### Training Strategy
1. **Start Simple**: Use default parameters first
2. **Monitor Metrics**: Watch for overfitting
3. **Early Stopping**: Let patience handle stopping
4. **Save Best**: Model saves best validation F1-score

### Model Deployment
1. **Test Thoroughly**: Use diverse test images
2. **Set Thresholds**: Adjust confidence thresholds for your use case
3. **Monitor Performance**: Track real-world performance
4. **Retrain Periodically**: Update model with new data

## 🚨 Common Mistakes

### Training Issues
- **Too Few Epochs**: Model may not converge
- **Too High Learning Rate**: Training may be unstable
- **Insufficient Data**: Model may overfit
- **No Validation**: Can't monitor overfitting

### Inference Issues
- **Wrong Model Path**: Ensure correct model file
- **Incorrect Image Format**: Use supported formats
- **Memory Issues**: Reduce batch size or image size
- **Threshold Too High**: May miss valid predictions

## 📞 Getting Help

If you encounter issues:

1. **Check Logs**: Review detailed error messages
2. **Verify Setup**: Ensure all dependencies installed
3. **Test Simple**: Try with minimal dataset first
4. **Search Issues**: Check GitHub issues for similar problems
5. **Create Issue**: Provide detailed error information 
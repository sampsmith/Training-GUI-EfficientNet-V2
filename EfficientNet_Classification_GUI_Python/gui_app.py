import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
from PIL import Image, ImageTk
import time
from datetime import datetime
warnings.filterwarnings('ignore')

class ISPMTrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ISPM 15 Defect Classifier - Training GUI")
        self.root.geometry("1200x800")
        
        self.dataset_path = tk.StringVar()
        self.model_save_path = tk.StringVar()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        self.training_thread = None
        self.is_training = False
        self.training_log = []
        
        self.model = None
        self.dataloaders = {}
        self.class_names = []
        
        self.setup_ui()
        
    def setup_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.create_dataset_tab(notebook)
        
        self.create_training_tab(notebook)
        
        self.create_model_tab(notebook)
        
        self.create_logs_tab(notebook)
        
    def create_dataset_tab(self, notebook):
        dataset_frame = ttk.Frame(notebook)
        notebook.add(dataset_frame, text="Dataset")
        
        path_frame = ttk.LabelFrame(dataset_frame, text="Dataset Configuration", padding=10)
        path_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(path_frame, text="Dataset Path:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(path_frame, textvariable=self.dataset_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(path_frame, text="Browse", command=self.browse_dataset).grid(row=0, column=2, padx=5, pady=5)
        
        self.dataset_info_frame = ttk.LabelFrame(dataset_frame, text="Dataset Information", padding=10)
        self.dataset_info_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.dataset_info_text = scrolledtext.ScrolledText(self.dataset_info_frame, height=15)
        self.dataset_info_text.pack(fill='both', expand=True)
        
        ttk.Button(dataset_frame, text="Load Dataset", command=self.load_dataset).pack(pady=10)
        
    def create_training_tab(self, notebook):
        training_frame = ttk.Frame(notebook)
        notebook.add(training_frame, text="Training")
        
        config_frame = ttk.LabelFrame(training_frame, text="Training Configuration", padding=10)
        config_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(config_frame, text="Model Save Path:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(config_frame, textvariable=self.model_save_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(config_frame, text="Browse", command=self.browse_model_save).grid(row=0, column=2, padx=5, pady=5)
        
        params_frame = ttk.LabelFrame(training_frame, text="Training Parameters", padding=10)
        params_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(params_frame, text="Epochs:").grid(row=0, column=0, sticky='w', pady=5)
        self.epochs_var = tk.IntVar(value=50)
        ttk.Entry(params_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Batch Size:").grid(row=0, column=2, sticky='w', pady=5)
        self.batch_size_var = tk.IntVar(value=32)
        ttk.Entry(params_frame, textvariable=self.batch_size_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Learning Rate:").grid(row=1, column=0, sticky='w', pady=5)
        self.lr_var = tk.DoubleVar(value=0.0001)
        ttk.Entry(params_frame, textvariable=self.lr_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Dropout Rate:").grid(row=1, column=2, sticky='w', pady=5)
        self.dropout_var = tk.DoubleVar(value=0.4)
        ttk.Entry(params_frame, textvariable=self.dropout_var, width=10).grid(row=1, column=3, padx=5, pady=5)
        
        aug_frame = ttk.LabelFrame(training_frame, text="Data Augmentation", padding=10)
        aug_frame.pack(fill='x', padx=10, pady=5)
        
        self.use_augmentation = tk.BooleanVar(value=True)
        ttk.Checkbutton(aug_frame, text="Use Data Augmentation", variable=self.use_augmentation).pack(anchor='w')
        
        control_frame = ttk.Frame(training_frame)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        self.train_button = ttk.Button(control_frame, text="Start Training", command=self.start_training)
        self.train_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Training", command=self.stop_training, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        progress_frame = ttk.LabelFrame(training_frame, text="Training Progress", padding=10)
        progress_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(progress_frame, text="Overall Progress:").grid(row=0, column=0, sticky='w', pady=2)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        
        ttk.Label(progress_frame, text="Epoch Progress:").grid(row=1, column=0, sticky='w', pady=2)
        self.epoch_progress_var = tk.DoubleVar()
        self.epoch_progress_bar = ttk.Progressbar(progress_frame, variable=self.epoch_progress_var, maximum=100)
        self.epoch_progress_bar.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        
        ttk.Label(progress_frame, text="Batch Progress:").grid(row=2, column=0, sticky='w', pady=2)
        self.batch_progress_var = tk.DoubleVar()
        self.batch_progress_bar = ttk.Progressbar(progress_frame, variable=self.batch_progress_var, maximum=100)
        self.batch_progress_bar.grid(row=2, column=1, sticky='ew', padx=5, pady=2)
        
        progress_frame.columnconfigure(1, weight=1)
        
        self.status_label = ttk.Label(training_frame, text="Ready to train")
        self.status_label.pack(pady=5)
        
        self.detailed_status_label = ttk.Label(training_frame, text="", font=("Arial", 9))
        self.detailed_status_label.pack(pady=2)
        
    def create_model_tab(self, notebook):
        model_frame = ttk.Frame(notebook)
        notebook.add(model_frame, text="Model")
        
        arch_frame = ttk.LabelFrame(model_frame, text="Model Architecture", padding=10)
        arch_frame.pack(fill='x', padx=10, pady=5)
        
        self.arch_text = scrolledtext.ScrolledText(arch_frame, height=10)
        self.arch_text.pack(fill='both', expand=True)
        
        eval_frame = ttk.LabelFrame(model_frame, text="Model Evaluation", padding=10)
        eval_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.eval_text = scrolledtext.ScrolledText(eval_frame, height=15)
        self.eval_text.pack(fill='both', expand=True)
        
        ttk.Button(model_frame, text="Evaluate Model", command=self.evaluate_model).pack(pady=10)
        
    def create_logs_tab(self, notebook):
        logs_frame = ttk.Frame(notebook)
        notebook.add(logs_frame, text="Logs")
        
        self.log_text = scrolledtext.ScrolledText(logs_frame, height=30)
        self.log_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        ttk.Button(logs_frame, text="Clear Logs", command=self.clear_logs).pack(pady=5)
        
    def browse_dataset(self):
        path = filedialog.askdirectory(title="Select Dataset Directory")
        if path:
            self.dataset_path.set(path)
            self.log_message(f"Dataset path set to: {path}")
            
    def browse_model_save(self):
        path = filedialog.asksaveasfilename(
            title="Save Model As",
            defaultextension=".pth",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        if path:
            self.model_save_path.set(path)
            self.log_message(f"Model save path set to: {path}")
            
    def load_dataset(self):
        if not self.dataset_path.get():
            messagebox.showerror("Error", "Please select a dataset path")
            return
            
        try:
            self.log_message("Loading dataset...")
            
            if self.use_augmentation.get():
                train_transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.3),
                    transforms.RandomRotation(degrees=15),
                    transforms.RandomApply([
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
                    ], p=0.7),
                    transforms.RandomApply([
                        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                    ], p=0.4),
                    transforms.RandomApply([
                        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
                    ], p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                train_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
            val_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            image_datasets = {
                'train': datasets.ImageFolder(os.path.join(self.dataset_path.get(), 'train'), train_transform),
                'val': datasets.ImageFolder(os.path.join(self.dataset_path.get(), 'val'), val_transform),
                'test': datasets.ImageFolder(os.path.join(self.dataset_path.get(), 'test'), val_transform)
            }
            
            class_counts = Counter(image_datasets['train'].targets)
            total_samples = len(image_datasets['train'])
            class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
            
            weights = [class_weights[label] for label in image_datasets['train'].targets]
            weighted_sampler = WeightedRandomSampler(weights, len(image_datasets['train']), replacement=True)
            
            batch_size = self.batch_size_var.get()
            self.dataloaders = {
                'train': DataLoader(image_datasets['train'], batch_size=batch_size, sampler=weighted_sampler, num_workers=0),
                'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=0),
                'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=0)
            }
            
            self.class_names = image_datasets['train'].classes
            dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
            
            info_text = f"Dataset loaded successfully!\n\n"
            info_text += f"Classes: {self.class_names}\n"
            info_text += f"Number of classes: {len(self.class_names)}\n\n"
            info_text += f"Dataset sizes:\n"
            for split, size in dataset_sizes.items():
                info_text += f"  {split}: {size} images\n"
            info_text += f"\nClass distribution (train):\n"
            for cls, count in class_counts.items():
                info_text += f"  {self.class_names[cls]}: {count} images\n"
            info_text += f"\nClass weights: {class_weights}\n"
            
            self.dataset_info_text.delete(1.0, tk.END)
            self.dataset_info_text.insert(1.0, info_text)
            
            self.update_model_architecture()
            
            self.log_message("Dataset loaded successfully!")
            messagebox.showinfo("Success", "Dataset loaded successfully!")
            
        except Exception as e:
            error_msg = f"Error loading dataset: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("Error", error_msg)
            
    def update_model_architecture(self):
        if not self.class_names:
            return
            
        arch_text = f"Model Architecture:\n\n"
        arch_text += f"Backbone: EfficientNetV2-M (pretrained on ImageNet)\n"
        arch_text += f"Input size: 224x224 pixels\n"
        arch_text += f"Number of classes: {len(self.class_names)}\n"
        arch_text += f"Classes: {self.class_names}\n\n"
        arch_text += f"Classifier Head:\n"
        arch_text += f"  - AdaptiveAvgPool2d(1,1)\n"
        arch_text += f"  - Flatten\n"
        arch_text += f"  - Dropout(p={self.dropout_var.get()})\n"
        arch_text += f"  - Linear(1280, 512) + ReLU + BatchNorm1d\n"
        arch_text += f"  - Dropout(p={self.dropout_var.get()})\n"
        arch_text += f"  - Linear(512, 128) + ReLU + BatchNorm1d\n"
        arch_text += f"  - Dropout(p={self.dropout_var.get()/2})\n"
        arch_text += f"  - Linear(128, {len(self.class_names)})\n\n"
        arch_text += f"Loss Function: Focal Loss (α=1, γ=2)\n"
        arch_text += f"Optimizer: AdamW\n"
        arch_text += f"Learning Rate Scheduler: OneCycleLR\n"
        
        self.arch_text.delete(1.0, tk.END)
        self.arch_text.insert(1.0, arch_text)
        
    def start_training(self):
        if not self.dataloaders:
            messagebox.showerror("Error", "Please load a dataset first")
            return
            
        if not self.model_save_path.get():
            messagebox.showerror("Error", "Please select a model save path")
            return
            
        if self.is_training:
            messagebox.showwarning("Warning", "Training is already in progress")
            return
            
        self.training_thread = threading.Thread(target=self.train_model)
        self.training_thread.daemon = True
        self.training_thread.start()
        
    def stop_training(self):
        self.is_training = False
        self.log_message("Training stopped by user")
        
    def train_model(self):
        self.is_training = True
        self.train_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.status_label.config(text="Training...")
        
        try:
            self.model = ISPMClassifier(num_classes=len(self.class_names), dropout_rate=self.dropout_var.get())
            self.model = self.model.to(self.device)
            
            criterion = FocalLoss(alpha=1, gamma=2)
            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr_var.get(), weight_decay=1e-4)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=self.lr_var.get() * 10, 
                epochs=self.epochs_var.get(), 
                steps_per_epoch=len(self.dataloaders['train'])
            )
            
            best_f1 = 0.0
            best_model_wts = self.model.state_dict()
            patience = 10
            patience_counter = 0
            
            total_epochs = self.epochs_var.get()
            
            for epoch in range(total_epochs):
                if not self.is_training:
                    break
                    
                self.log_message(f"Epoch {epoch+1}/{total_epochs}")
                
                for phase in ['train', 'val']:
                    if not self.is_training:
                        break
                        
                    if phase == 'train':
                        self.model.train()
                        dataloader = self.dataloaders['train']
                        total_batches = len(dataloader)
                    else:
                        self.model.eval()
                        dataloader = self.dataloaders['val']
                        total_batches = len(dataloader)
                        
                    running_loss = 0.0
                    running_corrects = 0
                    all_preds = []
                    all_labels = []
                    
                    self.root.after(0, lambda p=phase: self.detailed_status_label.config(text=f"Current phase: {p}"))
                    
                    for batch_idx, (inputs, labels) in enumerate(dataloader):
                        if not self.is_training:
                            break
                            
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        
                        optimizer.zero_grad()
                        
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                            
                            if phase == 'train':
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                                optimizer.step()
                                scheduler.step()
                                
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        
                        batch_progress = ((batch_idx + 1) / total_batches) * 100
                        self.root.after(0, lambda p=batch_progress: self.batch_progress_var.set(p))
                        
                        if batch_idx % 10 == 0:
                            current_loss = running_loss / ((batch_idx + 1) * inputs.size(0))
                            current_acc = running_corrects.float() / ((batch_idx + 1) * inputs.size(0))
                            status_text = f"{phase.capitalize()}: Batch {batch_idx+1}/{total_batches} | Loss: {current_loss:.4f} | Acc: {current_acc:.4f}"
                            self.root.after(0, lambda s=status_text: self.detailed_status_label.config(text=s))
                        
                    epoch_loss = running_loss / len(dataloader.dataset)
                    epoch_acc = running_corrects.float() / len(dataloader.dataset)
                    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
                    
                    self.log_message(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {f1:.4f}")
                    
                    epoch_progress = ((epoch + 1) / total_epochs) * 100
                    self.root.after(0, lambda p=epoch_progress: self.progress_var.set(p))
                    
                    self.root.after(0, lambda: self.batch_progress_var.set(0))
                    
                    if phase == 'val' and f1 > best_f1:
                        best_f1 = f1
                        best_model_wts = self.model.state_dict()
                        patience_counter = 0
                        self.log_message(f"New best F1 score: {best_f1:.4f}")
                    elif phase == 'val':
                        patience_counter += 1
                        
                if patience_counter >= patience:
                    self.log_message(f"Early stopping at epoch {epoch+1}")
                    break
                    
            if self.is_training:
                self.model.load_state_dict(best_model_wts)
                
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'class_names': self.class_names,
                    'data_transforms': {
                        'train': str(self.dataloaders['train'].dataset.transform),
                        'val': str(self.dataloaders['val'].dataset.transform),
                        'test': str(self.dataloaders['test'].dataset.transform)
                    }
                }, self.model_save_path.get())
                
                self.log_message(f"Training completed! Best F1: {best_f1:.4f}")
                self.log_message(f"Model saved to: {self.model_save_path.get()}")
                
                self.evaluate_model()
                
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("Error", error_msg)
            
        finally:
            self.is_training = False
            self.train_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.status_label.config(text="Training completed")
            self.detailed_status_label.config(text="")
            self.progress_var.set(0)
            self.epoch_progress_var.set(0)
            self.batch_progress_var.set(0)
            
    def evaluate_model(self):
        if not self.model or not self.dataloaders:
            messagebox.showerror("Error", "No trained model available")
            return
            
        try:
            self.log_message("Evaluating model...")
            
            self.model.eval()
            all_preds = []
            all_labels = []
            all_probs = []
            
            with torch.no_grad():
                for inputs, labels in self.dataloaders['test']:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
            accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
            roc_auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
            cm = confusion_matrix(all_labels, all_preds)
            
            eval_text = f"Model Evaluation Results:\n\n"
            eval_text += f"Accuracy: {accuracy:.4f}\n"
            eval_text += f"Precision: {precision:.4f}\n"
            eval_text += f"Recall: {recall:.4f}\n"
            eval_text += f"F1 Score: {f1:.4f}\n"
            eval_text += f"ROC AUC: {roc_auc:.4f}\n\n"
            eval_text += f"Confusion Matrix:\n"
            eval_text += f"True\\Pred\t{self.class_names[0]}\t{self.class_names[1]}\n"
            eval_text += f"{self.class_names[0]}\t\t{cm[0,0]}\t{cm[0,1]}\n"
            eval_text += f"{self.class_names[1]}\t\t{cm[1,0]}\t{cm[1,1]}\n"
            
            self.eval_text.delete(1.0, tk.END)
            self.eval_text.insert(1.0, eval_text)
            
            self.log_message("Model evaluation completed!")
            
        except Exception as e:
            error_msg = f"Evaluation error: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("Error", error_msg)
            
    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.training_log.append(log_entry)
        
        self.root.after(0, lambda: self.log_text.insert(tk.END, log_entry))
        self.root.after(0, lambda: self.log_text.see(tk.END))
        
    def clear_logs(self):
        self.log_text.delete(1.0, tk.END)
        self.training_log.clear()

class ISPMClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.4):
        super(ISPMClassifier, self).__init__()
        self.backbone = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False
            
        num_features = self.backbone.classifier[1].in_features
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout_rate/2),
            nn.Linear(128, num_classes)
        )
        
        self.backbone.classifier = nn.Identity()
        
    def forward(self, x):
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        output = self.classifier(features)
        return output

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

def main():
    root = tk.Tk()
    app = ISPMTrainingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 
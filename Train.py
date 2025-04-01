import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import seaborn as sns
import json
from datetime import datetime

# 1. Custom Preprocessing Function (기존 코드 유지)
def custom_preprocess(image):
    image = image.resize((384, 384), Image.Resampling.LANCZOS)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(image_clahe, (5, 5), 0)
    edge_enhanced = cv2.addWeighted(image_clahe, 1.5, blurred, -0.5, 0)
    image_enhanced = cv2.cvtColor(edge_enhanced, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_enhanced)
    return image_pil

# 2. Custom Dataset Class (기존 코드 유지)
class DogEyesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Eyelid Tumor', 'Nuclear Sclerosis', 'Cataract', 'Ulcerative Keratitis', 'Epiphora']
        self.images = []
        self.labels = []
        self.label2id = {label: idx for idx, label in enumerate(self.classes)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: {class_path} does not exist")
                continue
            for img_name in os.listdir(class_path):
                self.images.append(os.path.join(class_path, img_name))
                self.labels.append(class_idx)
    def __len__(self): return len(self.images)
    def __getitem__(self, idx): 
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform: image = self.transform(image)
        return image, label

# 3. Data Transformations (기존 코드 유지)
data_transforms = {
    'train': transforms.Compose([
        transforms.Lambda(custom_preprocess),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Lambda(custom_preprocess),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Lambda(custom_preprocess),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 4. Training Function (기존 코드 유지)
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, patience=5):
    best_acc = 0.0
    patience_counter = 0
    best_model_path = 'best_model'
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        model.train()
        running_loss, running_corrects, num_batches = 0.0, 0, 0
        train_progress = tqdm(train_loader, desc='Training', leave=False)
        for inputs, labels in train_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).logits
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data).item()
            num_batches += 1
            avg_loss = running_loss / (num_batches * train_loader.batch_size)
            avg_acc = running_corrects / (num_batches * train_loader.batch_size)
            train_progress.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{avg_acc:.4f}'})
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        model.eval()
        running_loss, running_corrects, num_batches = 0.0, 0, 0
        val_progress = tqdm(val_loader, desc='Validation', leave=False)
        with torch.no_grad():
            for inputs, labels in val_progress:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).logits
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data).item()
                num_batches += 1
                avg_loss = running_loss / (num_batches * val_loader.batch_size)
                avg_acc = running_corrects / (num_batches * val_loader.batch_size)
                val_progress.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{avg_acc:.4f}'})
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects / len(val_loader.dataset)
        val_losses.append(epoch_loss)
        val_accs.append(epoch_acc)
        print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        scheduler.step()
        print(f'Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            patience_counter = 0
            if isinstance(model, nn.DataParallel):
                model.module.save_pretrained(best_model_path)
            else:
                model.save_pretrained(best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    save_training_plots(train_losses, train_accs, val_losses, val_accs)
    return model

# 5. Plot Saving Function (기존 코드 유지)
def save_training_plots(train_losses, train_accs, val_losses, val_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, label='Training Accuracy', marker='o')
    plt.plot(epochs, val_accs, label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

# 6. Main Function with F1-Score, ROC-AUC, and Additional Graphs
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
    
    data_dir = '/home/dongbeen/ML/dog_eye_classification/data/split_data/'
    
    # Create datasets with custom transforms
    image_datasets = {
        'train': DogEyesDataset(os.path.join(data_dir, 'train'), transform=data_transforms['train']),
        'val': DogEyesDataset(os.path.join(data_dir, 'validation'), transform=data_transforms['val']),
        'test': DogEyesDataset(os.path.join(data_dir, 'test'), transform=data_transforms['test'])
    }
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=8, pin_memory=True),
        'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=8, pin_memory=True),
        'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    }
    
    # Initialize model
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-384',
        num_labels=5,
        ignore_mismatched_sizes=True
    )
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Class weights for imbalance
    class_counts = [len(os.listdir(os.path.join(data_dir, 'train', cls))) for cls in image_datasets['train'].classes]
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    
    # Optimizer and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Train the model
    trained_model = train_model(model, dataloaders['train'], dataloaders['val'], 
                                criterion, optimizer, scheduler, num_epochs=25, device=device, patience=5)
    
    # Test phase with confusion matrix, F1-Score, ROC-AUC, and confidence
    trained_model.eval()
    running_corrects = 0
    all_preds, all_labels, all_confidences, all_probs = [], [], [], []  # all_probs 추가
    
    start_time = datetime.now()
    print(f"Testing started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['test'], desc='Testing'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = trained_model(inputs).logits
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            confidences = torch.gather(probs, 1, preds.unsqueeze(1)).squeeze(1)
            
            running_corrects += torch.sum(preds == labels.data).item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())  # 모든 클래스에 대한 확률 저장
    
    test_acc = running_corrects / len(dataloaders['test'].dataset)
    print(f'Test Accuracy: {test_acc:.4f}')
    
    # Separate confidences for correct and wrong predictions
    correct_conf = [conf for pred, label, conf in zip(all_preds, all_labels, all_confidences) if pred == label]
    wrong_conf = [conf for pred, label, conf in zip(all_preds, all_labels, all_confidences) if pred != label]
    
    # F1-Score 계산
    f1_scores = f1_score(all_labels, all_preds, average=None)
    class_f1_scores = {cls: float(score) for cls, score in zip(image_datasets['test'].classes, f1_scores)}
    avg_f1_score = float(np.mean(f1_scores))
    print(f"Average F1-Score: {avg_f1_score:.4f}")
    
    # ROC-AUC 계산 (One-vs-Rest 방식)
    y_true_bin = label_binarize(all_labels, classes=range(len(image_datasets['test'].classes)))
    roc_auc = roc_auc_score(y_true_bin, np.array(all_probs), multi_class='ovr')
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=image_datasets['test'].classes, yticklabels=image_datasets['test'].classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Normalized Confusion Matrix (백분율)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=image_datasets['test'].classes, yticklabels=image_datasets['test'].classes)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC Curve 그리기 (각 클래스별)
    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(image_datasets['test'].classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], np.array(all_probs)[:, i])
        plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc_score(y_true_bin[:, i], np.array(all_probs)[:, i]):.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confidence Distribution 그래프
    plt.figure(figsize=(10, 6))
    plt.hist(correct_conf, bins=20, alpha=0.7, label='Correct Predictions', color='green', density=True)
    plt.hist(wrong_conf, bins=20, alpha=0.7, label='Wrong Predictions', color='red', density=True)
    plt.axvline(np.mean(correct_conf), color='green', linestyle='--', 
                label=f'Mean Correct: {np.mean(correct_conf):.3f}')
    plt.axvline(np.mean(wrong_conf) if wrong_conf else 0, color='red', linestyle='--', 
                label=f'Mean Wrong: {np.mean(wrong_conf) if wrong_conf else 0:.3f}')
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title('Confidence Score Distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results to JSON
    end_time = datetime.now()
    final_results = {
        'overall_accuracy': float(test_acc),
        'class_accuracies': {cls: 0.0 for cls in image_datasets['test'].classes},
        'class_f1_scores': class_f1_scores,  # F1-Score 추가
        'avg_f1_score': avg_f1_score,        # 평균 F1-Score 추가
        'roc_auc': float(roc_auc),           # ROC-AUC 추가
        'confusion_matrix': cm.tolist(),
        'avg_confidence_correct': float(np.mean(correct_conf)) if correct_conf else 0.0,
        'avg_confidence_wrong': float(np.mean(wrong_conf)) if wrong_conf else 0.0,
        'execution_time_seconds': (end_time - start_time).total_seconds()
    }
    
    # Update class-wise accuracies
    class_correct = {cls: 0 for cls in image_datasets['test'].classes}
    class_total = {cls: 0 for cls in image_datasets['test'].classes}
    for pred, label in zip(all_preds, all_labels):
        cls = image_datasets['test'].classes[label]
        class_total[cls] += 1
        if pred == label:
            class_correct[cls] += 1
    for cls in image_datasets['test'].classes:
        if class_total[cls] > 0:
            final_results['class_accuracies'][cls] = class_correct[cls] / class_total[cls]
    
    with open('final_results_multiclass.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    
    print(f"Average confidence (correct): {final_results['avg_confidence_correct']:.4f}")
    print(f"Average confidence (wrong): {final_results['avg_confidence_wrong']:.4f}")

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel
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

# 1. Custom Preprocessing Functions
def custom_preprocess_original(image):
    image = image.resize((384, 384), Image.Resampling.LANCZOS)
    image_np = np.array(image)
    if len(image_np.shape) == 2:  # 그레이스케일인 경우 RGB로 변환
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    image_pil = Image.fromarray(image_np)
    return image_pil

def custom_preprocess_fourier(image):
    image = image.resize((384, 384), Image.Resampling.LANCZOS)
    image_np = np.array(image)
    if len(image_np.shape) == 2:  # 그레이스케일인 경우 RGB로 변환
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    image_pil = Image.fromarray(image_np)
    return image_pil

# 2. Custom Dataset Class (원본과 푸리에 변환 이미지를 함께 로드)
class DogEyesDataset(Dataset):
    def __init__(self, root_dir_original, root_dir_fourier, transform_original=None, transform_fourier=None):
        self.root_dir_original = root_dir_original
        self.root_dir_fourier = root_dir_fourier
        self.transform_original = transform_original
        self.transform_fourier = transform_fourier
        self.classes = ['Eyelid Tumor', 'Nuclear Sclerosis', 'Cataract', 'Ulcerative Keratitis', 'Epiphora']
        self.images_original = []
        self.images_fourier = []
        self.labels = []
        self.label2id = {label: idx for idx, label in enumerate(self.classes)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        for class_idx, class_name in enumerate(self.classes):
            class_path_original = os.path.join(root_dir_original, class_name)
            class_path_fourier = os.path.join(root_dir_fourier, class_name)
            if not (os.path.exists(class_path_original) and os.path.exists(class_path_fourier)):
                print(f"Warning: {class_path_original} or {class_path_fourier} does not exist")
                continue
            img_files_original = os.listdir(class_path_original)
            img_files_fourier = os.listdir(class_path_fourier)
            # 파일 이름 기준으로 페어링 (예: image1.jpg와 image1_fourier.png)
            for img_name_original in img_files_original:
                base_name = img_name_original.replace('.jpg', '')
                for img_name_fourier in img_files_fourier:
                    if base_name in img_name_fourier:
                        self.images_original.append(os.path.join(class_path_original, img_name_original))
                        self.images_fourier.append(os.path.join(class_path_fourier, img_name_fourier))
                        self.labels.append(class_idx)
                        break
    
    def __len__(self): return len(self.images_original)
    
    def __getitem__(self, idx):
        img_path_original = self.images_original[idx]
        img_path_fourier = self.images_fourier[idx]
        
        image_original = Image.open(img_path_original).convert('RGB')
        image_fourier = Image.open(img_path_fourier).convert('RGB')
        
        label = self.labels[idx]
        
        if self.transform_original: image_original = self.transform_original(image_original)
        if self.transform_fourier: image_fourier = self.transform_fourier(image_fourier)
        
        return image_original, image_fourier, label

# 3. Data Transformations
data_transforms = {
    'train': {
        'original': transforms.Compose([
            transforms.Lambda(custom_preprocess_original),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'fourier': transforms.Compose([
            transforms.Lambda(custom_preprocess_fourier),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    },
    'val': {
        'original': transforms.Compose([
            transforms.Lambda(custom_preprocess_original),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'fourier': transforms.Compose([
            transforms.Lambda(custom_preprocess_fourier),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    },
    'test': {
        'original': transforms.Compose([
            transforms.Lambda(custom_preprocess_original),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'fourier': transforms.Compose([
            transforms.Lambda(custom_preprocess_fourier),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
}

# 4. Custom Model (원본과 푸리에 이미지를 결합)
class MultiInputViT(nn.Module):
    def __init__(self, num_classes=5):
        super(MultiInputViT, self).__init__()
        self.vit_original = ViTModel.from_pretrained('google/vit-base-patch16-384')
        self.vit_fourier = ViTModel.from_pretrained('google/vit-base-patch16-384')
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 256),  # 두 ViT 출력 결합 (768은 ViT base 출력 차원)
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x_original, x_fourier):
        outputs_original = self.vit_original(pixel_values=x_original).last_hidden_state[:, 0]  # [CLS] 토큰
        outputs_fourier = self.vit_fourier(pixel_values=x_fourier).last_hidden_state[:, 0]
        combined = torch.cat((outputs_original, outputs_fourier), dim=1)
        logits = self.classifier(combined)
        return logits

# 5. Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, patience=5):
    best_acc = 0.0
    patience_counter = 0
    best_model_path = 'best_multi_model.pth'
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        model.train()
        running_loss, running_corrects, num_batches = 0.0, 0, 0
        train_progress = tqdm(train_loader, desc='Training', leave=False)
        for inputs_original, inputs_fourier, labels in train_progress:
            inputs_original, inputs_fourier, labels = inputs_original.to(device), inputs_fourier.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs_original, inputs_fourier)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_size = inputs_original.size(0)
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
            for inputs_original, inputs_fourier, labels in val_progress:
                inputs_original, inputs_fourier, labels = inputs_original.to(device), inputs_fourier.to(device), labels.to(device)
                outputs = model(inputs_original, inputs_fourier)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                batch_size = inputs_original.size(0)
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
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    save_training_plots(train_losses, train_accs, val_losses, val_accs)
    return model

# 6. Plot Saving Function
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
    plt.savefig('loss_plot_wavelet.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, label='Training Accuracy', marker='o')
    plt.plot(epochs, val_accs, label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_plot_wavelet.png', dpi=300, bbox_inches='tight')
    plt.close()

# 7. Test Function (image_datasets를 인자로 받음)
def test_model(model, test_loader, device, image_datasets):
    model.eval()
    running_corrects = 0
    all_preds, all_labels, all_confidences, all_probs = [], [], [], []
    
    start_time = datetime.now()
    print(f"Testing started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    with torch.no_grad():
        for inputs_original, inputs_fourier, labels in tqdm(test_loader, desc='Testing'):
            inputs_original, inputs_fourier, labels = inputs_original.to(device), inputs_fourier.to(device), labels.to(device)
            outputs = model(inputs_original, inputs_fourier)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            confidences = torch.gather(probs, 1, preds.unsqueeze(1)).squeeze(1)
            
            running_corrects += torch.sum(preds == labels.data).item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    test_acc = running_corrects / len(test_loader.dataset)
    print(f'Test Accuracy: {test_acc:.4f}')
    
    # F1-Score 계산
    f1_scores = f1_score(all_labels, all_preds, average=None)
    class_f1_scores = {cls: float(score) for cls, score in zip(image_datasets['test'].classes, f1_scores)}
    avg_f1_score = float(np.mean(f1_scores))
    print(f"Average F1-Score: {avg_f1_score:.4f}")
    
    # ROC-AUC 계산
    y_true_bin = label_binarize(all_labels, classes=range(len(image_datasets['test'].classes)))
    roc_auc = roc_auc_score(y_true_bin, np.array(all_probs), multi_class='ovr')
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=image_datasets['test'].classes, yticklabels=image_datasets['test'].classes)
    plt.title('Confusion Matrix (wavelet)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix_wavelet.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Normalized Confusion Matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=image_datasets['test'].classes, yticklabels=image_datasets['test'].classes)
    plt.title('Normalized Confusion Matrix (wavelet)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix_normalized_wavelet.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC Curve
    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(image_datasets['test'].classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], np.array(all_probs)[:, i])
        plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc_score(y_true_bin[:, i], np.array(all_probs)[:, i]):.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class (wavelet)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('roc_curves_wavelet.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confidence Distribution
    correct_conf = [conf for pred, label, conf in zip(all_preds, all_labels, all_confidences) if pred == label]
    wrong_conf = [conf for pred, label, conf in zip(all_preds, all_labels, all_confidences) if pred != label]
    plt.figure(figsize=(10, 6))
    plt.hist(correct_conf, bins=20, alpha=0.7, label='Correct Predictions', color='green', density=True)
    plt.hist(wrong_conf, bins=20, alpha=0.7, label='Wrong Predictions', color='red', density=True)
    plt.axvline(np.mean(correct_conf), color='green', linestyle='--', 
                label=f'Mean Correct: {np.mean(correct_conf):.3f}')
    plt.axvline(np.mean(wrong_conf) if wrong_conf else 0, color='red', linestyle='--', 
                label=f'Mean Wrong: {np.mean(wrong_conf) if wrong_conf else 0:.3f}')
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title('Confidence Score Distribution (wavelet)')
    plt.legend()
    plt.grid(True)
    plt.savefig('confidence_distribution_wavelet.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results to JSON
    end_time = datetime.now()
    final_results = {
        'overall_accuracy': float(test_acc),
        'class_accuracies': {cls: 0.0 for cls in image_datasets['test'].classes},
        'class_f1_scores': class_f1_scores,
        'avg_f1_score': avg_f1_score,
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'avg_confidence_correct': float(np.mean(correct_conf)) if correct_conf else 0.0,
        'avg_confidence_wrong': float(np.mean(wrong_conf)) if wrong_conf else 0.0,
        'execution_time_seconds': (end_time - start_time).total_seconds()
    }
    
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
    
    with open('final_results_wavelet.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    
    print(f"Average confidence (correct): {final_results['avg_confidence_correct']:.4f}")
    print(f"Average confidence (wrong): {final_results['avg_confidence_wrong']:.4f}")

# 8. Main Function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
    
    data_dir_original = '/home/dongbeen/ML/dog_eye_classification/data/split_data/'
    data_dir_wavelet = '/home/dongbeen/ML/dog_eye_classification/data/split_data_wavelet/'
    
    image_datasets = {
        'train': DogEyesDataset(
            os.path.join(data_dir_original, 'train'),
            os.path.join(data_dir_wavelet, 'train'),
            transform_original=data_transforms['train']['original'],
            transform_fourier=data_transforms['train']['fourier']
        ),
        'val': DogEyesDataset(
            os.path.join(data_dir_original, 'validation'),
            os.path.join(data_dir_wavelet, 'validation'),
            transform_original=data_transforms['val']['original'],
            transform_fourier=data_transforms['val']['fourier']
        ),
        'test': DogEyesDataset(
            os.path.join(data_dir_original, 'test'),
            os.path.join(data_dir_wavelet, 'test'),
            transform_original=data_transforms['test']['original'],
            transform_fourier=data_transforms['test']['fourier']
        )
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=8, pin_memory=True),
        'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=8, pin_memory=True),
        'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    }
    
    model = MultiInputViT(num_classes=5).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    class_counts = [len(os.listdir(os.path.join(data_dir_original, 'train', cls))) for cls in image_datasets['train'].classes]
    weights = 1. / torch.tensor(class_counts, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    trained_model = train_model(model, dataloaders['train'], dataloaders['val'], 
                                criterion, optimizer, scheduler, num_epochs=25, device=device, patience=5)
    
    test_model(trained_model, dataloaders['test'], device, image_datasets)

if __name__ == '__main__':
    main()

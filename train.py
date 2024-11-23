# Standard library imports
from __future__ import print_function
import argparse
from datetime import datetime
import os
import shutil
import time

# Third-party library imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import (
    Precision, Recall, F1Score, 
    ConfusionMatrix, PrecisionRecallCurve, 
    ROC, Accuracy, MeanMetric
)
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
from torchvision import transforms as T

# Local imports
from ConvNet import ConvNet



# -------------------------------------------------------------------------
# Classes
# -------------------------------------------------------------------------

class CustomDataset(Dataset):
    """
    Description:
        A custom dataset class that loads images and their corresponding labels in YOLO format.
        Inherits from torch.utils.data.Dataset.

    Args:
        images_dir (str): Directory path containing the image files
        labels_dir (str): Directory path containing the label files
        transform (callable, optional): Optional transform to be applied on a sample

    Methods:
        __len__(): Returns the total number of samples
        __getitem__(idx): Returns the image and its corresponding labels at given index
    """

    def __init__(self, images_dir, labels_dir, transform=None):
        super(CustomDataset, self).__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = os.listdir(images_dir)
        self.transform = transform

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.image_files[idx].replace('.jpg', '.txt'))
        
        image = Image.open(img_path).convert("RGB")
        
        # Read YOLO format labels (class_id, x_center, y_center, width, height)
        with open(label_path, 'r') as f:
            labels = f.readlines()
        
        # Initialize targets
        class_target = torch.zeros(1, dtype=torch.long)
        bbox_target = torch.zeros(4, dtype=torch.float)
        
        if labels:
            parts = list(map(float, labels[0].strip().split()))
            class_target[0] = int(parts[0])
            bbox_target = torch.tensor(parts[1:], dtype=torch.float)
        
        if self.transform:
            image = self.transform(image)
            
        return image, (class_target, bbox_target)
    


# -------------------------------------------------------------------------
# Methods
# -------------------------------------------------------------------------

def train(model, device, train_loader, optimizer, criterion, epoch, batch_size, file, metric_logger):
    """
    Description:
        Trains the model for one epoch using the provided data loader and optimization parameters.
        Updates model weights and tracks various performance metrics.

    Args:
        model (nn.Module): The neural network model to train
        device (torch.device): The device to run the training on ('cuda' or 'cpu')
        train_loader (DataLoader): DataLoader containing the training data
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters
        criterion (nn.Module): The loss function to use for training
        epoch (int): Current epoch number
        batch_size (int): Size of each training batch
        file (file object): File handle for logging training progress
        metric_logger (dict): Dictionary containing various metric tracking objects

    Returns:
        tuple: Contains:
            - train_loss (float): Average training loss for the epoch
            - train_acc (float): Training accuracy for the epoch
            - precision (float): Training precision score
            - recall (float): Training recall score
            - f1 (float): Training F1 score
            - bbox_error (float): Average bounding box error
    """
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty lists to store losses 
    losses = []
    cls_correct = 0
    bbox_losses = []

    # Reset metrics at the start of each epoch
    metric_logger['precision'].reset()
    metric_logger['recall'].reset()
    metric_logger['f1'].reset()
    metric_logger['bbox_error'].reset()
    
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, (cls_target, bbox_target) = batch_sample
        
        # Push data/label to correct device
        data = data.to(device)
        cls_target = cls_target.to(device)
        bbox_target = bbox_target.to(device)

        # Ensure proper dimensions
        cls_target = cls_target.squeeze()
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        cls_output, bbox_output = model(data)
        
        # Compute loss based on criterion
        loss = criterion(cls_output, bbox_output, cls_target, bbox_target)
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()

        # Calculate classification accuracy
        cls_pred = cls_output.argmax(dim=1)
        cls_correct += cls_pred.eq(cls_target).sum().item()

        # Calculate bbox loss
        bbox_loss = F.smooth_l1_loss(bbox_output, bbox_target)
        bbox_losses.append(bbox_loss.item())

        # Update metrics
        metric_logger['precision'].update(cls_pred, cls_target)
        metric_logger['recall'].update(cls_pred, cls_target)
        metric_logger['f1'].update(cls_pred, cls_target)
        metric_logger['bbox_error'].update(bbox_loss)

        if batch_idx < len(train_loader) - 1:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                f'({100. * batch_idx / len(train_loader):.0f}%)]     '
                f'Loss: {loss.item():.6f}\t\t', end='\r', flush=True)
        else:
            print(f'Train Epoch: {epoch} [{len(train_loader.dataset)}/{len(train_loader.dataset)} '
                f'(100%)]     Loss: {loss.item():.6f}\t\t', end='\r', flush=True)

        #BBox Loss: {bbox_loss.item():.6f}

    print()

    # Calculate metrics
    train_loss = float(np.mean(losses))
    train_bbox_loss = float(np.mean(bbox_losses))
    train_acc = cls_correct / len(train_loader.dataset)
    precision = metric_logger['precision'].compute()
    recall = metric_logger['recall'].compute()
    f1 = metric_logger['f1'].compute()
    bbox_error = metric_logger['bbox_error'].compute()

    # Log results
    file.write(f'Epoch {epoch}\n')
    file.write(f'Train set: Total Loss: {train_loss:.4f}, BBox Loss: {train_bbox_loss:.4f}\n')
    file.write(f'Classification Accuracy: {cls_correct}/{len(train_loader.dataset)} ({train_acc*100:.2f}%)\n')
    file.write(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n')
    file.write(f'Average BBox Error: {bbox_error:.4f}\n')

    return train_loss, train_acc, precision, recall, f1, bbox_error
    
def test(model, device, test_loader, criterion, file, metric_logger):
    """
    Description:
        Evaluates the model's performance on test data, computing various metrics
        without updating model parameters.

    Args:
        model (nn.Module): The neural network model to evaluate
        device (torch.device): The device to run the evaluation on ('cuda' or 'cpu')
        test_loader (DataLoader): DataLoader containing the test data
        criterion (nn.Module): The loss function to use for evaluation
        file (file object): File handle for logging test results
        metric_logger (dict): Dictionary containing various metric tracking objects

    Returns:
        tuple: Contains:
            - test_loss (float): Average test loss
            - accuracy (float): Test accuracy
            - precision (float): Test precision score
            - recall (float): Test recall score
            - f1 (float): Test F1 score
            - bbox_error (float): Average bounding box error
    """
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    # Empty lists to store losses 
    losses = []
    cls_correct = 0
    bbox_losses = []

    # Reset metrics at the start of testing
    metric_logger['precision'].reset()
    metric_logger['recall'].reset()
    metric_logger['f1'].reset()
    metric_logger['bbox_error'].reset()
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, (cls_target, bbox_target) = sample
            data = data.to(device)
            cls_target = cls_target.to(device)
            bbox_target = bbox_target.to(device)

            # Ensure proper dimensions
            cls_target = cls_target.squeeze()
            
            # Forward pass
            cls_output, bbox_output = model(data)
            
            # Compute combined loss
            loss = criterion(cls_output, bbox_output, cls_target, bbox_target)
            losses.append(loss.item())
            
            # Calculate classification accuracy
            cls_pred = cls_output.argmax(dim=1)
            cls_correct += cls_pred.eq(cls_target).sum().item()
            
            # Calculate bbox loss
            bbox_loss = F.smooth_l1_loss(bbox_output, bbox_target.view(bbox_output.size()))
            bbox_losses.append(bbox_loss.item())
            
            # Update metrics
            cls_pred = cls_output.argmax(dim=1)
            cls_target = cls_target.view(-1)
            metric_logger['precision'].update(cls_pred, cls_target)
            metric_logger['recall'].update(cls_pred, cls_target)
            metric_logger['f1'].update(cls_pred, cls_target)
            metric_logger['bbox_error'].update(bbox_loss)

    # Calculate metrics
    test_loss = float(np.mean(losses))
    test_bbox_loss = float(np.mean(bbox_losses))
    accuracy = cls_correct / len(test_loader.dataset)
    precision = metric_logger['precision'].compute()
    recall = metric_logger['recall'].compute()
    f1 = metric_logger['f1'].compute()
    bbox_error = metric_logger['bbox_error'].compute()

    # Log results
    file.write(f'Test set: Total Loss: {test_loss:.4f}, BBox Loss: {test_bbox_loss:.4f}\n')
    file.write(f'Classification Accuracy: {cls_correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    file.write(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n')
    file.write(f'Average BBox Error: {bbox_error:.4f}\n\n')

    return test_loss, accuracy, precision, recall, f1, bbox_error

def calculate_confidence_curves(all_predictions, all_confidences, all_targets, num_classes, thresholds=None):
    """
    Description:
        Calculates precision, recall, and F1 scores at different confidence thresholds
        for each class to generate confidence curves.

    Args:
        all_predictions (torch.Tensor): Tensor containing model predictions
        all_confidences (torch.Tensor): Tensor containing confidence scores for predictions
        all_targets (torch.Tensor): Tensor containing ground truth labels
        num_classes (int): Number of classes in the dataset
        thresholds (list, optional): List of confidence thresholds to evaluate

    Returns:
        tuple: Contains:
            - metrics (dict): Dictionary containing F1, precision, and recall scores for each class at each threshold
            - thresholds (list): List of thresholds used for evaluation
    """

    if thresholds is None:
        thresholds = torch.linspace(0, 1, 100)
    
    # Initialize dictionaries for each metric
    metrics = {
        'f1': {i: [] for i in range(num_classes)},
        'precision': {i: [] for i in range(num_classes)},
        'recall': {i: [] for i in range(num_classes)}
    }
    # Add 'all' key for overall metrics
    for metric in metrics:
        metrics[metric]['all'] = []
    
    # Calculate metrics for each threshold
    for threshold in thresholds:
        # Initialize counters for each class
        tp = torch.zeros(num_classes)
        fp = torch.zeros(num_classes)
        fn = torch.zeros(num_classes)
        
        # Filter predictions by confidence threshold
        mask = all_confidences >= threshold
        filtered_preds = all_predictions[mask]
        filtered_targets = all_targets[mask]
        
        # Calculate metrics for each class
        for class_idx in range(num_classes):
            pred_class = filtered_preds == class_idx
            true_class = filtered_targets == class_idx
            
            tp[class_idx] = (pred_class & true_class).sum().float()
            fp[class_idx] = (pred_class & ~true_class).sum().float()
            fn[class_idx] = (~pred_class & true_class).sum().float()
        
        # Calculate metrics for each class
        for class_idx in range(num_classes):
            if tp[class_idx] + fp[class_idx] + fn[class_idx] == 0:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
            else:
                precision = tp[class_idx] / (tp[class_idx] + fp[class_idx] + 1e-10)
                recall = tp[class_idx] / (tp[class_idx] + fn[class_idx] + 1e-10)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            metrics['f1'][class_idx].append(f1)
            metrics['precision'][class_idx].append(precision)
            metrics['recall'][class_idx].append(recall)
        
        # Calculate overall metrics (macro average)
        for metric_name in ['f1', 'precision', 'recall']:
            overall = sum(metrics[metric_name][i][-1] for i in range(num_classes)) / num_classes
            metrics[metric_name]['all'].append(overall)
    
    return metrics, thresholds.tolist()

def plot_confidence_curve(thresholds, metrics, metric_name, class_names, log_dir):
    """
    Description:
        Creates and saves a plot showing how a specific metric (F1, precision, or recall)
        varies with confidence threshold for each class.

    Args:
        thresholds (list): List of confidence thresholds
        metrics (dict): Dictionary containing metric values for each class
        metric_name (str): Name of the metric to plot ('f1', 'precision', or 'recall')
        class_names (list): List of class names
        log_dir (str): Directory path to save the plot

    Returns:
        None: Saves the plot to the specified directory
    """

    plt.figure(figsize=(12, 8))
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    # Plot individual class curves
    for i, (class_idx, color) in enumerate(zip(range(len(class_names)), colors)):
        plt.plot(thresholds, metrics[metric_name][class_idx], 
                color=color, label=class_names[i], alpha=0.7)
    
    # Plot overall curve (thicker line)
    plt.plot(thresholds, metrics[metric_name]['all'], 
            color='black', label='all classes', linewidth=3, alpha=0.8)
    
    # Find and annotate maximum score point
    max_score_idx = np.argmax(metrics[metric_name]['all'])
    max_score = metrics[metric_name]['all'][max_score_idx]
    max_conf = thresholds[max_score_idx]
    plt.plot([max_conf], [max_score], 'ko')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Confidence')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'{metric_name.capitalize()}-Confidence Curve')
    
    # Add annotation for maximum point
    annotation = f'all classes {max_score:.2f} at {max_conf:.3f}'
    plt.text(0.98, 0.02, annotation, transform=plt.gca().transAxes,
             horizontalalignment='right', verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{log_dir}/curves/{metric_name}_confidence_curve.png", bbox_inches='tight')
    plt.close()

def log_final_results(model, device, test_loader, metric_logger, log_dir):
    """
    Description:
        Generates and saves final evaluation metrics and visualizations including
        confusion matrix, ROC curves, and precision-recall curves.

    Args:
        model (nn.Module): The trained model to evaluate
        device (torch.device): The device to run evaluation on
        test_loader (DataLoader): DataLoader containing test data
        metric_logger (dict): Dictionary containing metric tracking objects
        log_dir (str): Directory path to save results and visualizations

    Returns:
        None: Saves various plots and metrics to the specified directory
    """
        
    model.eval()
    all_preds = []
    all_targets = []
    all_confidences = []
    
    # First collect all predictions and targets
    with torch.no_grad():
        for data, (cls_target, _) in test_loader:
            data, cls_target = data.to(device), cls_target.to(device)
            cls_target = cls_target.view(-1)  # Ensure 1D for targets
            cls_output, _ = model(data)

            # Get class predictions and confidences
            cls_probs = F.softmax(cls_output, dim=1)
            confidences, cls_pred = torch.max(cls_probs, dim=1)

            all_preds.extend(cls_pred.cpu().numpy())
            all_targets.extend(cls_target.cpu().numpy())
            all_confidences.extend(confidences.cpu())
            
            # Update precision-recall and ROC metrics
            metric_logger['pr_curve'].update(cls_output.softmax(dim=1), cls_target)
            metric_logger['roc_curve'].update(cls_output.softmax(dim=1), cls_target)

    # Convert lists to tensors
    all_preds = torch.tensor(all_preds)
    all_targets = torch.tensor(all_targets)
    all_confidences = torch.tensor(all_confidences)

    # Define class labels
    confusion_labels = ['B-1_TopDown', 'B-2_TopDown', 'C-130_TopDown', 'C-5_TopDown', 'E-3_TopDown']

    metrics, thresholds = calculate_confidence_curves(
        all_preds, all_confidences, all_targets, 
        num_classes=len(confusion_labels)
    )
    
    # ---------- F1, Precision, & Recall Curve ----------
    for metric_name in ['f1', 'precision', 'recall']:
        plot_confidence_curve(thresholds, metrics, metric_name, 
                            confusion_labels, log_dir)

    # ---------- Precision-Recall Curve ----------
    precision, recall, _ = metric_logger['pr_curve'].compute()
    plt.figure()
    if isinstance(precision, (list, tuple)):
        for i, (prec, rec) in enumerate(zip(precision, recall)):
            plt.plot(rec.cpu().numpy(), prec.cpu().numpy(), label=f'Class {i}')
    else:
        for i in range(len(precision)):
            plt.plot(recall[i].cpu().numpy(), precision[i].cpu().numpy(), label=f'Class {i}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(f"{log_dir}/curves/precision_recall_curve.png")
    plt.close()

    # ---------- ROC Curve ----------
    fpr, tpr, _ = metric_logger['roc_curve'].compute()
    plt.figure()
    if isinstance(tpr, (list, tuple)):
        for i, (false_pos, true_pos) in enumerate(zip(fpr, tpr)):
            plt.plot(false_pos.cpu().numpy(), true_pos.cpu().numpy(), label=f'Class {i}')
    else:
        for i in range(len(tpr)):
            plt.plot(fpr[i].cpu().numpy(), tpr[i].cpu().numpy(), label=f'Class {i}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"{log_dir}/curves/roc_curve.png")
    plt.close()

    # ---------- Confusion Matrix ----------
    cm = confusion_matrix(all_preds, all_targets)

    # Normalize by row (sum of each row = 100%)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Replace NaN with 0 (in case of empty rows)
    cm_normalized = np.nan_to_num(cm_normalized)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=confusion_labels, yticklabels=confusion_labels)
    plt.ylabel('Predicted')
    plt.xlabel('True')
    plt.title("Confusion Matrix")

    plt.tight_layout()  # Adjust layout to prevent label cutoff
    
    plt.savefig(f"{log_dir}/confusion_matrix.png")
    plt.close()

    print("Final results saved to:", log_dir)

def plot_epoch_data(num_epochs, log_dir, train_losses, test_losses, train_accuracies, test_accuracies):
    """
    Description:
        Creates and saves plots showing the progression of model performance metrics
        (loss and accuracy) over training epochs for both training and test sets.

    Args:
        num_epochs (int): Total number of training epochs
        log_dir (str): Directory path to save the generated plots
        train_losses (list): List of average training losses for each epoch
        test_losses (list): List of average test losses for each epoch
        train_accuracies (list): List of training accuracies for each epoch
        test_accuracies (list): List of test accuracies for each epoch

    Returns:
        None: Saves two plots to the specified directory:
            - avg_loss_per_epoch.png: Plot of training and test losses over epochs
            - avg_accuracy_per_epoch.png: Plot of training and test accuracies over epochs
    """

    # Plot and save the average loss per epoch
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Average Loss per Epoch")
    plt.legend()
    plt.savefig(f"./{log_dir}/epochs/avg_loss_per_epoch.png")
    plt.close()

    # Plot and save the average accuracy per epoch
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, num_epochs + 1), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Average Accuracy")
    plt.title("Average Accuracy per Epoch")
    plt.legend()
    plt.savefig(f"./{log_dir}/epochs/avg_accuracy_per_epoch.png")
    plt.close()

# -------------------------------------------------------------------------
# Object detection methods
# -------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    Description:
        A custom loss function that combines classification and bounding box regression losses.
        Inherits from nn.Module.

    Args:
        cls_weight (float): Weight factor for classification loss
        bbox_weight (float): Weight factor for bounding box regression loss

    Methods:
        forward(cls_pred, bbox_pred, cls_target, bbox_target): Computes the combined loss
    """

    def __init__(self, cls_weight, bbox_weight):
        super(CombinedLoss, self).__init__()
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.cls_criterion = nn.CrossEntropyLoss()
        self.bbox_criterion = nn.SmoothL1Loss()
        
    def forward(self, cls_pred, bbox_pred, cls_target, bbox_target):
        # Reshape classification target to be 1D
        cls_target = cls_target.view(-1)
        
        # Ensure bbox predictions match target dimensions
        bbox_pred = bbox_pred.view(bbox_target.size())
        
        cls_loss = self.cls_criterion(cls_pred, cls_target)
        bbox_loss = self.bbox_criterion(bbox_pred, bbox_target)
        return self.cls_weight * cls_loss + self.bbox_weight * bbox_loss

def draw_boxes(image, boxes, classes, class_names, confidence_threshold=0.5):
    """
    Description:
        Draws predicted bounding boxes and class labels on an image with confidence scores.

    Args:
        image (PIL.Image or numpy.ndarray): Input image to draw boxes on
        boxes (torch.Tensor): Tensor of shape (num_classes, 4) containing box coordinates
        classes (torch.Tensor): Tensor of shape (num_classes) containing class probabilities
        class_names (list): List of class names for labeling
        confidence_threshold (float, optional): Minimum confidence score to draw a box

    Returns:
        tuple: Contains:
            - image (numpy.ndarray): Image with drawn boxes
            - box_drawn (bool): Whether any boxes were drawn
    """
    
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    image = image.astype(np.uint8)
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    height, width = image.shape[:2]
    box_drawn = False

    # Move boxes to CPU before operations
    boxes = boxes.cpu()
    classes = classes.cpu()
    
    for i, (box, conf) in enumerate(zip(boxes, classes)):
        if conf < confidence_threshold:
            continue
            
        # Convert relative coordinates to absolute pixels
        dimensions = torch.tensor([width, height, width, height], device='cpu')
        x, y, w, h = box * dimensions
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label = f"{class_names[i]}: {conf:.2f}"
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        box_drawn = True
    
    return image, box_drawn

def detect_objects(model, image_tensor, class_names):
    """
    Description:
        Performs object detection on a single image using the trained model.

    Args:
        model (nn.Module): The trained detection model
        image_tensor (torch.Tensor): Input image tensor
        class_names (list): List of class names

    Returns:
        tuple: Contains:
            - class_probs (torch.Tensor): Class probabilities for detected objects
            - bbox_preds (torch.Tensor): Predicted bounding box coordinates
    """

    model.eval()
    with torch.no_grad():
        class_preds, bbox_preds = model(image_tensor)
        
        # Get class probabilities
        class_probs = F.softmax(class_preds, dim=1)
        
        # Reshape bbox predictions
        bbox_preds = bbox_preds.view(-1, len(class_names), 4)
        
        return class_probs, bbox_preds

def find_and_annotate_boxes(model, class_names, test_loader, device, output_dir='detected_boxes'):
    """
    Description:
        Processes a batch of images through the model and saves annotated versions
        with detected boxes and class labels.

    Args:
        model (nn.Module): The trained detection model
        class_names (list): List of class names
        test_loader (DataLoader): DataLoader containing images to process
        device (torch.device): Device to run detection on
        output_dir (str, optional): Directory to save annotated images

    Returns:
        None: Saves annotated images to the specified directory
    """

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        for i, (images, (targets, bbox_targets)) in enumerate(test_loader):
            images = images.to(device)
            cls_output, bbox_output = model(images)
            
            # Process each image in the batch
            for j, image in enumerate(images):
                # Convert the tensor to PIL Image
                img_np = image.cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * 255).astype(np.uint8)
                
                # Get predictions for this image
                cls_pred = F.softmax(cls_output[j], dim=0)
                bbox_pred = bbox_output[j].view(-1, 4)
                
                # Draw boxes (no need to explicitly move tensors to CPU - handled in draw_boxes)
                annotated_img, box_drawn = draw_boxes(img_np, bbox_pred, cls_pred, class_names)
                
                # Set filename based on whether a box was drawn
                prefix = "success_img" if box_drawn else "detected_img"
                output_path = os.path.join(output_dir, f'{prefix}_{i}_{j}.jpg')
                
                # Save the annotated image
                cv2.imwrite(output_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))



# -------------------------------------------------------------------------
# Main - Program Order
# -------------------------------------------------------------------------

def run_main(FLAGS):
    """
    Description:
        Main function that orchestrates the entire training and evaluation pipeline.
        Sets up the model, dataloaders, and training parameters, then runs training
        and final evaluation.

    Args:
        FLAGS (argparse.Namespace): Command line arguments containing:
            - learning_rate (float): Initial learning rate
            - num_epochs (int): Number of training epochs
            - batch_size (int): Training batch size
            - log_dir (str): Directory for saving logs and results
            - num_classes (int): Number of object classes
            - class_weight (float): Weight for classification loss
            - bbox_weight (float): Weight for bounding box loss

    Returns:
        None: Trains model and saves results to specified directory
    """

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    #device = torch.device ("cpu")     # Uncomment this if cuda is taking too long...
    print("Torch device selected: ", device)
    torch.cuda.empty_cache()
    
    # Initialize the model with 5 output classes and send to device
    model = ConvNet(FLAGS.num_classes).to(device)
    
    # Define the criterion for loss.
    criterion = CombinedLoss(cls_weight=FLAGS.class_weight, bbox_weight=FLAGS.bbox_weight)
    
    # Define optimizer function.
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
        
    # Create transformations to apply to each data sample 
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
    ])
    
    # Load datasets for training and testing
    train_dataset = CustomDataset(images_dir='./data/train/images/', labels_dir='./data/train/labels/', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
    valid_dataset = CustomDataset(images_dir='./data/valid/images/', labels_dir='./data/valid/labels/', transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=FLAGS.batch_size, shuffle=True)
    test_dataset = CustomDataset(images_dir='./data/test/images/', labels_dir='./data/test/labels/', transform=transform)
    test_loader = DataLoader(valid_dataset, batch_size=FLAGS.batch_size, shuffle=True)
    
    # Initialize metrics
    metric_logger = {
        'precision': Precision(task='multiclass', num_classes=FLAGS.num_classes, average='macro').to(device),
        'recall': Recall(task='multiclass', num_classes=FLAGS.num_classes, average='macro').to(device),
        'f1': F1Score(task='multiclass', num_classes=FLAGS.num_classes, average='macro').to(device),
        'conf_matrix': ConfusionMatrix(task='multiclass', num_classes=FLAGS.num_classes).to(device),
        'pr_curve': PrecisionRecallCurve(task='multiclass', num_classes=FLAGS.num_classes).to(device),
        'roc_curve': ROC(task='multiclass', num_classes=FLAGS.num_classes).to(device),
        'bbox_error': MeanMetric().to(device)
    }

    best_metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'bbox_error': float('inf')
    }

    # Create a directory/file for the output
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    output_file_path = os.path.join(FLAGS.log_dir, 'output.txt')
    
    # Run training for n_epochs specified in config
    start_time = time.time()
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")

    # Lists to store average loss and accuracy per epoch for training and testing
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    with open(output_file_path, 'w') as file:

        # Print header information
        print(f"Beginning training! Date and time = {formatted_now}")
        file.write("\n--------------------------------------------------------------------\n")
        file.write(f"- Date: {formatted_now}\n")
        file.write(f"- Learning rate: {FLAGS.learning_rate}\n")
        file.write(f"- Epochs: {FLAGS.num_epochs}\n")
        file.write(f"- Batch size: {FLAGS.batch_size}\n\n")
        file.write(f"- Classes: {FLAGS.num_classes}\n\n")

        # Simulate training
        for epoch in range(1, FLAGS.num_epochs + 1):
            train_loss, train_acc, train_prec, train_recall, train_f1, train_bbox = train(
                model, device, train_loader, optimizer, criterion, epoch, FLAGS.batch_size, file, metric_logger
            )
            test_loss, test_acc, test_prec, test_recall, test_f1, test_bbox = test(
                model, device, valid_loader, criterion, file, metric_logger
            )
            
            # Update model results
            best_metrics['accuracy'] = max(best_metrics['accuracy'], test_acc)
            best_metrics['precision'] = max(best_metrics['precision'], test_prec)
            best_metrics['recall'] = max(best_metrics['recall'], test_recall)
            best_metrics['f1'] = max(best_metrics['f1'], test_f1)
            best_metrics['bbox_error'] = min(best_metrics['bbox_error'], test_bbox)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)

            # Debug printing
            total_time = time.time() - start_time
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            #print(f"\nTotal training time: {int(hours):02}:{int(minutes):02}:{int(seconds):02} (hh:mm:ss)\n")
            #print(f"Epoch: {epoch}")
            #print(f"Train loss: {train_loss}, Train Accuracy: {train_acc}")
            #print(f"Test loss: {test_loss}, Test Accuracy: {test_acc}\n")

            # Write final results
            if epoch == FLAGS.num_epochs:
                file.write("\n\nFinal Results (Saved Model):")
                file.write("\nTrain accuracy: {:.2f}%".format(train_acc))
                file.write("\nTrain precision: {:.4f}".format(train_prec))
                file.write("\nTrain recall: {:.4f}".format(train_recall))
                file.write("\nTrain F1 score: {:.4f}".format(train_f1))
                file.write("\nTrain bbox error: {:.4f}\n".format(train_bbox))
                file.write("\nTest accuracy: {:.2f}%".format(test_acc))
                file.write("\nTest precision: {:.4f}".format(test_prec))
                file.write("\nTest recall: {:.4f}".format(test_recall))
                file.write("\nTest F1 score: {:.4f}".format(test_f1))
                file.write("\nTest bbox error: {:.4f}\n".format(test_bbox))
        
        # Write best results
        file.write("\nBest Results:")
        file.write("\nTest accuracy: {:.2f}%".format(best_metrics['accuracy']))
        file.write("\nTest precision: {:.4f}".format(best_metrics['precision']))
        file.write("\nTest recall: {:.4f}".format(best_metrics['recall']))
        file.write("\nTest F1 score: {:.4f}".format(best_metrics['f1']))
        file.write("\nBest bbox error: {:.4f}".format(best_metrics['bbox_error']))
        file.write("\n--------------------------------------------------------------------\n")

        # Generate final visualizations
        log_final_results(model, device, valid_loader, metric_logger, FLAGS.log_dir)
        
        # Save example detections
        classnames = ['B-1_TopDown', 'B-2_TopDown', 'C-130_TopDown', 'C-5_TopDown', 'E-3_TopDown']
        find_and_annotate_boxes(model, classnames, train_loader, device)

        # Plot average loss per epoch and average accuracy per epoch
        plot_epoch_data(FLAGS.num_epochs, FLAGS.log_dir, train_losses, test_losses, train_accuracies, test_accuracies)
 
if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN: Aircraft Segmentation.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.0001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=16,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    parser.add_argument('--num_classes',
                        type=int, default=5,
                        help='Number of output classes. Represents total number of label types.')
    parser.add_argument('--class_weight',
                        type=int, default=1.0,
                        help='Weight of class loss in overall loss calculation. Between 0 and 1')
    parser.add_argument('--bbox_weight',
                        type=int, default=0.0,
                        help='Weight of bbox loss in overall loss calculation. Between 0 and 1')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)
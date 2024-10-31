from __future__ import print_function
import argparse
from datetime import datetime
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
from torchvision import transforms as T
import cv2
from torchmetrics import (
    Precision, Recall, F1Score, MeanMetric,
    ConfusionMatrix, PrecisionRecallCurve, ROC, Accuracy
)
from ConvNet import ConvNet

# -------------------------------------------------------------------------
# Classes
# -------------------------------------------------------------------------

class CustomDataset(Dataset):
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
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''
    
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

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}\tBBox Loss: {bbox_loss.item():.6f}')

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
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
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


def log_final_results(model, device, test_loader, metric_logger, log_dir):
    """
    """
        
    model.eval()
    all_preds = []
    all_targets = []
    
    # First collect all predictions and targets
    with torch.no_grad():
        for data, (cls_target, _) in test_loader:
            data, cls_target = data.to(device), cls_target.to(device)
            cls_target = cls_target.view(-1)  # Ensure 1D for targets
            cls_output, _ = model(data)
            cls_pred = cls_output.argmax(dim=1)
            all_preds.extend(cls_pred.cpu().numpy())
            all_targets.extend(cls_target.cpu().numpy())
            
            # Update precision-recall and ROC metrics
            metric_logger['pr_curve'].update(cls_output.softmax(dim=1), cls_target)
            metric_logger['roc_curve'].update(cls_output.softmax(dim=1), cls_target)

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title("Confusion Matrix")
    plt.savefig(f"{log_dir}/confusion_matrix.png")
    plt.close()

    # Precision-Recall Curve
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
    plt.savefig(f"{log_dir}/precision_recall_curve.png")
    plt.close()

    # ROC Curve
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
    plt.savefig(f"{log_dir}/roc_curve.png")
    plt.close()

    print("Final results saved to:", log_dir)


# -------------------------------
# Object detection methods
# -------------------------------

class CombinedLoss(nn.Module):
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


# Helper function to draw bounding boxes on images
def draw_boxes(image, boxes, classes, class_names, confidence_threshold=0.5):
    """
    Draw predicted bounding boxes on the image
    
    Args:
        image: PIL Image or numpy array
        boxes: tensor of shape (num_classes, 4) with (x, y, w, h) coordinates
        classes: tensor of shape (num_classes) with class probabilities
        class_names: list of class names
        confidence_threshold: minimum confidence to draw a box
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

# Example usage
def detect_objects(model, image_tensor, class_names):
    model.eval()
    with torch.no_grad():
        class_preds, bbox_preds = model(image_tensor)
        
        # Get class probabilities
        class_probs = F.softmax(class_preds, dim=1)
        
        # Reshape bbox predictions
        bbox_preds = bbox_preds.view(-1, len(class_names), 4)
        
        return class_probs, bbox_preds

# Function to find and annotate boxes on test images
def find_and_annotate_boxes(model, class_names, test_loader, device, output_dir='detected_boxes'):
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

# -------------------------------


def run_main(FLAGS):
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
            print(f"\nTotal training time: {int(hours):02}:{int(minutes):02}:{int(seconds):02} (hh:mm:ss)\n")
            print(f"Epoch: {epoch}")
            print(f"Train loss: {train_loss}, Train Accuracy: {train_acc}")
            print(f"Test loss: {test_loss}, Test Accuracy: {test_acc}\n")

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

        # Plot and save the average loss per epoch
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, FLAGS.num_epochs + 1), train_losses, label="Train Loss")
        plt.plot(range(1, FLAGS.num_epochs + 1), test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.title("Average Loss per Epoch")
        plt.legend()
        plt.savefig(f"./{FLAGS.log_dir}/avg_loss_per_epoch.png")
        plt.close()

        # Plot and save the average accuracy per epoch
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, FLAGS.num_epochs + 1), train_accuracies, label="Train Accuracy")
        plt.plot(range(1, FLAGS.num_epochs + 1), test_accuracies, label="Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Average Accuracy")
        plt.title("Average Accuracy per Epoch")
        plt.legend()
        plt.savefig(f"./{FLAGS.log_dir}/avg_accuracy_per_epoch.png")
        plt.close()
    
    
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
                        type=int, default=64,
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
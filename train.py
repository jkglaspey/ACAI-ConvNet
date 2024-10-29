from __future__ import print_function
import argparse
from datetime import datetime
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
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
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = os.listdir(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.image_files[idx].replace('.jpg', '.txt'))

        image = Image.open(img_path).convert("RGB")
        with open(label_path, 'r') as f:
            labels = f.readlines()

        # Process labels to get a single target class
        if labels:  # Check if there are any labels
            # Assume the format is <class_id> <x_center> <y_center> <width> <height>
            class_id, _, _, _, _ = map(float, labels[0].strip().split())
            target = class_id  # Use only the first class ID
        else:
            target = 0  # Default or placeholder class ID if no labels are found

        # Convert target to tensor
        target_tensor = torch.tensor(int(target), dtype=torch.long)  # Make sure it's long type for CrossEntropyLoss

        if self.transform:
            image = self.transform(image)

        return image, target_tensor
    

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
    
    # Empty list to store losses 
    losses = []
    correct = 0

    # Reset metrics at the start of each epoch
    metric_logger['precision'].reset()
    metric_logger['recall'].reset()
    metric_logger['f1'].reset()
    
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample
        
        # Push data/label to correct device
        data, target = data.to(device), target.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(data)
        
        # Compute loss based on criterion
        loss = criterion(output, target)
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=True)

        # Convert the target to be the same dimensions as the prediction
        target_size_pred = target.view(pred.size())
        
        # Count correct predictions overall
        correct += pred.eq(target_size_pred).sum().item()   # Utilize eq() to sum the number of matching elements
        
        # Update metrics
        pred = output.argmax(dim=1)
        target = target.view(-1)
        metric_logger['precision'].update(pred, target)
        metric_logger['recall'].update(pred, target)
        metric_logger['f1'].update(pred, target)

        # Add some logging every N batches
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    train_loss = float(np.mean(losses))
    train_acc = correct / ((batch_idx+1) * batch_size)
    precision = metric_logger['precision'].compute()
    recall = metric_logger['recall'].compute()
    f1 = metric_logger['f1'].compute()

    file.write(f'Epoch {epoch}/{FLAGS.num_epochs}\n')
    file.write(f'Train set: Loss: {train_loss:.4f}, Accuracy: {correct}/{(batch_idx+1) * batch_size} ({train_acc*100:.2f}%)\n')
    file.write(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n')

    return train_loss, train_acc, precision, recall, f1
    


def test(model, device, test_loader, criterion, file, metric_logger):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    correct = 0

    # Reset metrics at the start of testing
    metric_logger['precision'].reset()
    metric_logger['recall'].reset()
    metric_logger['f1'].reset()
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)
            
            # Predict for data by doing forward pass
            output = model(data)
            
            # Compute loss based on criterion
            loss = criterion(output, target)
            
            # Append loss to overall test loss
            losses.append(loss.item())
            
            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=True)
            
            # Convert the target to be the same dimensions as the prediction
            target_size_pred = target.view(pred.size())
            
            # Count correct predictions overall
            correct += pred.eq(target_size_pred).sum().item()   # Utilize eq() to sum the number of matching elements

            # Update metrics
            pred = output.argmax(dim=1)
            target = target.view(-1)
            metric_logger['precision'].update(pred, target)
            metric_logger['recall'].update(pred, target)
            metric_logger['f1'].update(pred, target)

    test_loss = float(np.mean(losses))
    accuracy = 100. * correct / len(test_loader.dataset)
    precision = metric_logger['precision'].compute()
    recall = metric_logger['recall'].compute()
    f1 = metric_logger['f1'].compute()

    file.write(f'Test set: Loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    file.write(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n\n')

    return test_loss, accuracy, precision, recall, f1


def log_final_results(model, device, test_loader, metric_logger):
    """
    """
        
    model.eval()
    all_preds = []
    all_targets = []
    
    # First collect all predictions and targets
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # Update the PR and ROC curves
            metric_logger['pr_curve'].update(output.softmax(dim=1), target)
            metric_logger['roc_curve'].update(output.softmax(dim=1), target)

    # Now compute the curves
    try:
        # Plot confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title("Confusion Matrix")
        plt.savefig(f"{FLAGS.log_dir}/confusion_matrix.png")
        plt.close()
        
        # Get PR curve data
        precision, recall, _ = metric_logger['pr_curve'].compute()
        
        # Plot PR curve (handling both tensor and list formats)
        plt.figure()
        if isinstance(precision, (list, tuple)):
            for i, (prec, rec) in enumerate(zip(precision, recall)):
                plt.plot(rec.cpu().numpy(), prec.cpu().numpy(), label=f'Class {i}')
        else:
            # If they're tensors
            for i in range(len(precision)):
                plt.plot(recall[i].cpu().numpy(), precision[i].cpu().numpy(), label=f'Class {i}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.savefig(f"{FLAGS.log_dir}/precision_recall_curve.png")
        plt.close()
        
        # Get ROC curve data
        fpr, tpr, _ = metric_logger['roc_curve'].compute()
        
        # Plot ROC curve (handling both tensor and list formats)
        plt.figure()
        if isinstance(tpr, (list, tuple)):
            for i, (false_pos, true_pos) in enumerate(zip(fpr, tpr)):
                plt.plot(false_pos.cpu().numpy(), true_pos.cpu().numpy(), label=f'Class {i}')
        else:
            # If they're tensors
            for i in range(len(tpr)):
                plt.plot(fpr[i].cpu().numpy(), tpr[i].cpu().numpy(), label=f'Class {i}')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(f"{FLAGS.log_dir}/roc_curve.png")
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not generate curves due to error: {e}")
        
    print("Results saved to:", FLAGS.log_dir)


def load_image(image_path):
    image = cv2.imread(image_path)
    return image

def transform_image(image):
    img_transform = T.ToTensor()
    image_tensor = img_transform(image)
    return image_tensor

def find_and_annotate_boxes(model, classnames, test_loader, device):
    model.eval()
    # Detect objects
    bbox, scores, labels = detect_objects(model, test_loader, device)
    # Draw bounding boxes and labels
    #result_img = draw_boxes_and_labels(img, bbox, labels, classnames)
    # Display the result
    #cv2.imshow('Detected Objects', result_img)

def detect_objects(model, test_loader, device, confidence_threshold=0.80):
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)
            y_pred = model(data[0])
            break

    print(f"y_pred = {y_pred}")
    
    bbox, scores, labels = y_pred[0]['boxes'], y_pred[0]['scores'], y_pred[0]['labels']
    indices = torch.nonzero(scores > confidence_threshold).squeeze(1)

    filtered_bbox = bbox[indices]
    filtered_scores = scores[indices]
    filtered_labels = labels[indices]

    for i in range(len(filtered_bbox)):
        print(f"Bounding Box: {filtered_bbox[i].cpu().numpy()}")
        print(f"Score: {filtered_scores[i].cpu().item()}")
        print(f"Label: {filtered_labels[i].cpu().item()}\n")

    return filtered_bbox, filtered_scores, filtered_labels

def draw_boxes_and_labels(image, bbox, labels, class_names):
    img_copy = image.copy()

    for i in range(len(bbox)):
        x, y, w, h = bbox[i].numpy().astype('int')
        cv2.rectangle(img_copy, (x, y), (w, h), (0, 0, 255), 5)

        class_index = labels[i].numpy().astype('int')
        class_detected = class_names[class_index - 1]

        cv2.putText(img_copy, class_detected, (x, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)

    return img_copy


def run_main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    torch.cuda.empty_cache()
    
    # Initialize the model with 5 output classes and send to device
    model = ConvNet(FLAGS.num_classes).to(device)
    
    # Define the criterion for loss.
    criterion = nn.CrossEntropyLoss()
    
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
        'roc_curve': ROC(task='multiclass', num_classes=FLAGS.num_classes).to(device)
    }

    best_accuracy = 0.0
    best_prec = 1.0
    best_recall = 1.0
    best_f1 = 1.0

    # Create a directory/file for the output
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    output_file_path = os.path.join(FLAGS.log_dir, 'output.txt')
    
    # Run training for n_epochs specified in config
    start_time = time.time()
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
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
            train_loss, train_acc, train_prec, train_recall, train_f1 = train(
                model, device, train_loader, optimizer, criterion, epoch, FLAGS.batch_size, file, metric_logger
            )
            test_loss, test_acc, test_prec, test_recall, test_f1 = test(
                model, device, valid_loader, criterion, file, metric_logger
            )
            
            # Update model results
            if test_acc > best_accuracy:
                best_accuracy = test_acc
            if test_prec < best_prec:
                best_prec = test_prec
            if test_recall < best_recall:
                best_recall = test_recall
            if test_f1 < best_f1:
                best_f1 = test_f1

            # Debug printing
            total_time = time.time() - start_time
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"\nTotal training time: {int(hours):02}:{int(minutes):02}:{int(seconds):02} (hh:mm:ss)\n")
            print(f"Epoch: {epoch}")
            print(f"Train loss: {train_loss}, Train Accuracy: {train_acc}, Train Precision: {train_prec}, Train Recall: {train_recall}, Train F1: {train_f1}")
            print(f"Test loss: {test_loss}, Test Accuracy: {test_acc}, Test Precision: {test_prec}, Test Recall: {test_recall}, Test F1: {test_f1}\n")
        
        file.write("\nBest test accuracy: {:.2f}%".format(best_accuracy))
        file.write("\nBest test precision: {:.2f}".format(best_prec))
        file.write("\nBest test recall: {:.2f}".format(best_recall))
        file.write("\nBest test f1 score: {:.2f}".format(best_f1))
        log_final_results(model, device, valid_loader, metric_logger)
        #classnames = ['B-1_TopDown', 'B-2_TopDown', 'C-130_TopDown', 'C-5_TopDown', 'E-3_TopDown']
        #find_and_annotate_boxes(model, classnames, test_loader, device)
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        file.write("\nTotal time: {:02}:{:02}:{:02} (hh:mm:ss)\n".format(int(hours), int(minutes), int(seconds)))
        file.write("------------------------------------------------------\n\n\n\n")
    
    
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
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)
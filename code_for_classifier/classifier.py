import argparse
parser = argparse.ArgumentParser(description="Train a ResNet model for binary classification of images.")
parser.add_argument('--data_dir', type=str, required=True, help='Directory with the dataset.')
parser.add_argument('--test_dir', type=str, required=True, help='Directory with the dataset.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and testing.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
parser.add_argument('--local', type=str, default='', help='Directory with the dataset.')
parser.add_argument('--job_id', type=str, default='local', help='Directory with the dataset.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--to_test_method', type=str, default='ours')
parser.add_argument('--arch', type=str, default='resnet18')
parser.add_argument('--val_epoch', type=int, default=1)
parser.add_argument("--split_val", action="store_true", default=False)
parser.add_argument("--test_transform", action="store_true", default=False)
parser.add_argument("--no_aug", action="store_true", default=False)
parser.add_argument('--save_name', type=str, default='')

args = parser.parse_args()

import os
if args.local != '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.local

import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import timm

from tqdm import tqdm

def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(args.seed)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.no_aug:
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
else:
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomApply([
            transforms.RandomResizedCrop(224, scale=(0.6, 1)),
        ], p=0.7),
        transforms.RandomApply([
            transforms.GaussianBlur(3),
        ], p=0.3),
        transforms.RandomApply([
            transforms.ElasticTransform(),
        ], p=0.1),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        ], p=0.3),
        transforms.RandomApply([
            transforms.Grayscale(3)
        ], p=0.3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

if args.arch == "resnet18":
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 101)

elif args.arch == "resnet50":
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 101)

elif args.arch == "vit":
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)

if args.mode == "train":

    # Load datasets

    if args.split_val:
        # import pdb ; pdb.set_trace()
        if args.test_transform:
            dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, _ = random_split(dataset, [train_size, test_size])

            dataset = datasets.ImageFolder(root=args.data_dir, transform=test_transform)
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            _, test_dataset = random_split(dataset, [train_size, test_size])
        else:
            dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    else:
        train_dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
        test_dataset = datasets.ImageFolder(root=args.test_dir, transform=test_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=32)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32)

    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.arch == "resnet18":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-5)
    elif args.arch == "resnet50":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-5)
    elif args.arch == "vit":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_correct = 0

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        batch_counter = 0
        correct = 0
        total = 0
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            batch_counter += 1

            running_loss += loss.item()

            train_bar.set_description(f"Epoch: [{epoch+1}/{args.epochs}] Loss: [{(running_loss / batch_counter):.04f}] Accuracy: {(100 * correct / total):.02f}%")

        # print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

        if (epoch + 1) % args.val_epoch == 0:

            # Evaluate the model
            model.eval()
            correct = 0
            total = 0
            test_bar = tqdm(test_loader)
            with torch.no_grad():
                for inputs, labels in test_bar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    test_bar.set_description(f"Test Epoch: [{epoch+1}/{args.epochs}] Accuracy: {(100 * correct / total):.02f}%")

            if correct >= best_correct:
                best_correct = correct
                torch.save(model.state_dict(), f"./results/best_classifier_{args.arch}_ours_0530_{args.save_name}_{epoch}.pt")
                print("saved")

elif args.mode == "test":

    model_path = f"./results/{args.to_test_method}.pt"
    
    dataset = datasets.ImageFolder(root=args.test_dir, transform=test_transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=32)

    '''dataset.imgs'''

    # Model setup
    # model = models.resnet18(pretrained=False)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 2)
    params = torch.load(model_path)
    # import pdb ; pdb.set_trace()
    model.load_state_dict(params)
    model = model.to(device)
    model.eval()

    # Test the model
    correct = 0
    total = 0
    label_list = []
    predicted_list = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            label_list.append(labels.cpu().numpy())
            predicted_list.append(predicted.cpu().numpy())

    print(f'Tested Model Accuracy: {100 * correct / total}%')


elif args.mode == "test_gamma":


    model_path = f"./results/{args.to_test_method}.pt"
    
    dataset = datasets.ImageFolder(root=args.test_dir, transform=test_transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    '''dataset.imgs'''

    # Model setup
    # model = models.resnet18(pretrained=False)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 2)
    params = torch.load(model_path)
    # import pdb ; pdb.set_trace()
    model.load_state_dict(params)
    model = model.to(device)
    model.eval()

    # Test the model
    correct = 0
    total = 0
    label_list = []
    predicted_list = []
    clean_counter = 0
    wrong_clean_counter = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # import pdb ; pdb.set_trace()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            label_list.append(labels.cpu().numpy())
            
            predicted_list.append(predicted.cpu().numpy())  

            # import pdb ; pdb.set_trace()

    print(f'Tested Model Accuracy: {100 * correct / total}%')


    from sklearn.metrics import precision_score, recall_score

    labels = np.concatenate(label_list, axis=0)
    predictions = np.concatenate(predicted_list, axis=0)

    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)

    file_path = f"./results/test/{args.test_dir.split('results/')[1].replace('/', '')}.npy"

    save_label_pred = np.concatenate([labels, predictions], axis=0)
    np.save(file_path, save_label_pred)

    # import pdb ; pdb.set_trace()

    print("Precision:", precision)
    print("Recall:", recall)

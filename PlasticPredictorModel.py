import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import kagglehub

import cv2
import os

class PlasticPredictorModel:
    def __init__(self, num_labels=8, save_path='plastic_predictor.pth'):
        model_name = 'google/vit-base-patch16-224-in21k'
        self.config = ViTConfig.from_pretrained(model_name, num_labels=num_labels)
        self.model = ViTForImageClassification.from_pretrained(model_name, config=self.config)
        self.extractor = ViTImageProcessor.from_pretrained(model_name)
        self.save_path = save_path

        self.batch_size = 64
        self.epochs = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.classes = [
            '1_polyethylene_PET',
            '2_high_density_polyethylene_PE-HD',
            '3_polyvinylchloride_PVC',
            '4_low_density_polyethylene_PE-LD',
            '5_polypropylene_PP',
            '6_polystyrene_PS',
            '7_other_resins',
            '8_no_plastic'
        ]

    def train(self, dataset_path=None):
        if dataset_path is None:
            dataset_path = "C:\\Users\\eric huang\\.cache\\kagglehub\\datasets\\piaoya\\plastic-recycling-codes\\versions\\9\\seven_plastics"

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
        dataset_size = len(full_dataset)
        val_size = int(0.2 * dataset_size)
        train_size = dataset_size - val_size

        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images).logits
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Validation phase
            self.model.eval()
            correct = 0
            total = 0
            val_loss = 0.0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images).logits
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = correct / total
            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {running_loss/len(train_loader):.4f}, "
                f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.4f}")

        torch.save(self.model.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")

        return val_dataset


    def file_classify(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image path '{path}' does not exist.")

        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to read the image at '{path}'.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        img_tensor = img_transform(img).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img_tensor).logits
            _, predicted = torch.max(outputs, 1)

        return self.classes[predicted.item()]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig # pytorch based model
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import cv2
import os

class PlasticPredictorModel:

    def __init__(self):
        model_name = 'google/vit-base-patch16-224-in21k'
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.config = ViTConfig.from_pretrained(model_name, num_labels = 3)
        self.extractor = ViTImageProcessor.from_pretrained(model_name)
        self.model.classifier = torch.nn.Linear(self.model.config.hidden_size, 3)

        self.batch_size = 64
        self.epochs = 10

        self.model.eval()

    def train(self):
        
        transform = transforms.Compose([
            transform.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
        ])

        path = os.path('enter path for the folder containing all images. download the kaggle plastic dataset')
        training_dataset = datasets.ImageFolder(root = path, transform = transform)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-4)

        train_loader = DataLoader(training_dataset, batch_size = self.batch_size, shuffle = True)

        for epoch in range(self.epochs):
            self.model.train()
            loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                loss += loss.item()
            
            
            print("Epoch: " + str(epoch+1) + "/" + str(self.epochs))
            print("Loss: " + str(loss/len(train_loader)))

        torch.save(self.model.state_dict(), 'folder u wish to save ur model to')

    def file_classify(self, path):

        # Returns an int 0-2
        # 0 - Polyethylene PET
        # 1 - High Density Polyethylene
        # 2 - Polyvinylchloride PVC
        # 3 - Low Density Polyethylene
        # 4 - Polypropylene PP
        # 5 - Polystyrene PS
        # 6 - Other Resins
        # 7 - No Plastic
        
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        assert img is not None, "Path is inaccessible. Try backslashing or os.path.exists()"
        img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = img_transform(img)
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            _, predicted = torch.max(self.model(img_tensor).logits, 1)

        classes = ['1_polyethylene_PET', '2_high_density_polyethylene_PE-HD', '3_polyvinylchloride_PVC', '4_low_density_polyethylene_PE-LD', '5_polypropylene_PP', '6_polystyrene_PS', '7_other_resins', '8_no_plastic']

        return classes[predicted.item()]

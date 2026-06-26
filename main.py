import os
import argparse
import pandas as pd
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn

from conversion import MainDataset
from CNN import CNN

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

def train_model(csv_path, data_dir, epochs=10):

    print("Инициализация датасета...")
    dataset = MainDataset(csv_file=csv_path, root_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    num_classes = len(dataset.classes)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Обучение пойдёт на: {device}")
    
    model = CNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Найдено классов: {num_classes}, всего картинок: {len(dataset)}")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()    
            outputs = model(images)    
            loss = criterion(outputs, labels) 
            loss.backward()            
            optimizer.step()          
            
            running_loss += loss.item()
            
        print(f"У {epoch+1}/{epochs} | Loss: {running_loss/len(dataloader):.4f}")
        
    torch.save({
        'dict': model.state_dict(),
        'classes': dataset.classes
    }, 'weights.pth')

    print("Обучение пршло успешно")

def predict_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    checkpoint = torch.load('weights.pth', map_location=device)
    classes = checkpoint['classes']
    num_classes = len(classes)
    
    model = CNN(num_classes).to(device)
    
    model.load_state_dict(checkpoint['dict'])
    model.eval() 
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device) 
    
    with torch.no_grad(): 
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class_idx = torch.max(probabilities, 1)
        
    predicted_class = classes[top_class_idx.item()]
    print(f"[{image_path}]")
    print(f"=== Результат: {predicted_class.upper()} ===")
    print(f"Вероятность: {top_prob.item() * 100:.2f}%\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', type=str)
    parser.add_argument('--csv', type=str, default='wonders_of_world_images.csv')
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--epochs', type=int, default=10)
   
    
    args = parser.parse_args()
    
    if args.train:
        train_model(args.csv, args.data_dir, args.epochs)
    elif args.predict:
        predict_image(args.predict)

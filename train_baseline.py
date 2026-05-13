import torch
import pandas as pd
from tqdm import tqdm # Để hiện thanh tiến trình

# Khởi tạo scaler cho Mixed Precision (Cực tốt cho CMP 40HX)
scaler = torch.cuda.amp.GradScaler()

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Sử dụng Mixed Precision
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        
        
    return running_loss/len(dataloader), 100.*correct/total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss/len(dataloader), 100.*correct/total

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

# Trong vòng lặp chính (ví dụ 30 epochs)
for epoch in range(30):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Save logs (Sub-task 3)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    
    print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}%")
    
    # Save model checkpoint
    if val_acc >= max(history['val_acc'] + [0]):
        torch.save(model.state_dict(), 'best_baseline_model.pth')
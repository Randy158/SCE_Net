from utils.DataLoade import MedSegment
from model.net import SCE_Net
from utils.transform import transform
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import torch
from Train_model import train

if __name__ == "__main__":
    
    train_img_dir = ""
    train_label_dir = ""
    val_img_dir = ""
    val_label_dir = ""

    epoch = 200
    classes_num = 2
    log_dir = r"logs"
    model_save = r"model_save"
    model_save_dir = os.path.join(model_save)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    train_data = MedSegment(train_img_dir, train_label_dir, transform['train'])
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4, drop_last=True)
    val_data = MedSegment(val_img_dir, val_label_dir, transform['val'])
    val_dataloader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=4, drop_last=False)
    net = SCE_Net(num_classes=classes_num)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)

    lr_scheduler_choice = "Exponential"  
    if lr_scheduler_choice == "Cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-5)
    elif lr_scheduler_choice == "Step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    elif lr_scheduler_choice == "Exponential":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        print("Invalid lr_scheduler_choice, using default CosineAnnealingLR.")
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-5)

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)
    criterion = criterion.to(device)

    train(net, train_dataloader, val_dataloader, epoch, criterion, optimizer, log_dir, model_save_dir, device, lr_scheduler)

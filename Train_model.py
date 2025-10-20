import os
import pandas as pd
import torch
from torch.autograd import Variable
from tqdm import tqdm
from torch.cuda.amp import GradScaler
def train(net, train_dataloader, val_dataloader, epoch, criterion, optimizer, log_dir, model_save_dir,
           device, scheduler):
    best_loss = float('inf')
    scaler = GradScaler()
    
    columns = ['epoch', 'train_loss', 'val_loss']
    result_df = pd.DataFrame(columns=columns)

    for e in range(epoch):
        if scheduler is not None:
            scheduler.step()

        net.train()
        train_loss = 0.0

        for i, batchdata in tqdm(enumerate(train_dataloader)):
            batchdata, batchlabel = Variable(batchdata['image'], requires_grad=True).to(device), Variable(batchdata['mask'].squeeze(1), requires_grad=True).to(device)
            
            output = net(batchdata)
            loss = criterion(output, batchlabel.type(torch.cuda.LongTensor))
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            optimizer.step()
            
            train_loss += loss.cpu().item() * batchlabel.size(0)

        train_loss /= len(train_dataloader.dataset)
        print(f'\nepoch:{e+1}, train_loss:{train_loss:.4f}')

        net.eval()
        val_loss = 0.0

        with torch.no_grad():
            for i, batchdata in tqdm(enumerate(val_dataloader)):
                batchdata, batchlabel = Variable(batchdata['image'], requires_grad=True).to(device), Variable(batchdata['mask'].squeeze(1), requires_grad=True).to(device)
                
                output = net(batchdata)
                loss = criterion(output, batchlabel.type(torch.cuda.LongTensor))
                
                val_loss += loss.cpu().item() * batchlabel.size(0)
            
            val_loss /= len(val_dataloader.dataset)
        
        print(f'epoch:{e+1}, val_loss:{val_loss:.4f}')

        result_df = result_df.append({'epoch': e + 1, 'train_loss': train_loss, 'val_loss': val_loss}, ignore_index=True)
        result_df.to_excel(os.path.join(log_dir, "_train_results.xlsx"), index=False)

        if val_loss < best_loss:
            best_loss = val_loss
            model_path = os.path.join(model_save_dir, "_best.pth")
            torch.save(net.state_dict(), model_path)

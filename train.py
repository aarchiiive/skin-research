# torch library
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt

# library
import time
import os
import copy
from tqdm import tqdm
import gc
import pandas as pd

# local library
from model import _models
from utils import *
from test import *

def train_model(model, 
                dataloaders, 
                criterion, 
                optimizer, 
                scheduler,
                device,
                save_path="", 
                num_epochs=25, 
                resume=False,
                start=0):

    if torch.cuda.device_count() > 1:
        model = nn.Parallel(model)
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    model = copy.deepcopy(model)
    model = model.to(device)
    criterion = criterion.to(device)
    
    if not os.path.isdir("weights/{}".format(save_path)):
        os.mkdir("weights/{}".format(save_path))
        
    if resume: 
        # 학습을 다시 시작한다면 가중치(.pt)를 불러옴
        model.load_state_dict(torch.load("weights/{}/last.pt".format(save_path)))
        f = open("weights/{}/log.txt".format(save_path), 'r')
        start = int(len(f.readlines()) / 2)
        f.close()
        f = open("weights/{}/log.txt".format(save_path), 'a')
        
        print("Restart training model....")
    else: 
        # 학습을 처음 시작할 경우
        if os.path.isfile("weights/{}/log.txt".format(save_path)):
            print("log file already exist.")
            exit()
            
        f = open("weights/{}/log.txt".format(save_path), 'w')
        
    try:
        for epoch in range(start, num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 120)

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                
                gc.collect()
                torch.cuda.empty_cache()

                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels.long())

                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                        
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                if phase == "train":
                    scheduler.step()
                    
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    model.load_state_dict(best_model_wts)
                    torch.save(model.state_dict(), "weights/{}/best.pt".format(save_path))
                    
                if phase == "train":
                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                elif phase == "val":
                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Best: {best_acc:.4f}')
                    
                f.write(f'{phase} Loss {epoch_loss:.4f} Acc {epoch_acc:.4f}\n')
                
            torch.save(model.state_dict(), "weights/{}/last.pt".format(save_path))
                
            print()
            
    except KeyboardInterrupt:
        time_elapsed = time.time() - since
        print(f'Training stopped in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        
        torch.save(model.state_dict(), "weights/{}/last.pt".format(save_path))
        f.close()

    time_elapsed = time.time() - since
    
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)

    f.close()
    
def train(dataset_path,
          save_path,
          model_name,
          num_epochs=25,
          input_size=600,
          num_classes=4,
          optimizer="Adam",
          learning_rate=0.0001,
          weight_decay=0.0005,
          drop_rate=0.2,
          batch_size=8,
          num_workers=10,
          resume=False,
          start=0):
    
    if not os.path.isdir("weights"):
        os.mkdir("weights")
        
    if not os.path.isdir("data"):
        os.mkdir("data")
        
    if not os.path.isdir(os.path.join("weights", save_path)):
        os.mkdir(os.path.join("weights", save_path))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    print("save weights in {}....".format(save_path))
    print("save hyperparmeters......")
    save_params(os.path.join("weights", save_path),
                model_name,
                num_epochs,
                input_size,
                num_classes,
                optimizer,
                learning_rate,
                weight_decay,
                drop_rate,
                batch_size,
                num_workers)
    
    # dataloaders
    trainloader, testloader = dataloader(os.path.join(dataset_path, "train"),
                                         os.path.join(dataset_path, "train"),
                                        [os.path.join(dataset_path, "label/train.csv"),
                                         os.path.join(dataset_path, "label/val.csv")],
                                         input_size=input_size,
                                         batch_size=batch_size,
                                         num_workers=num_workers)

    dataloaders = {'train': trainloader, 
                    'val' : testloader}
        
    # model
    model = _models(model_name=model_name, 
                    num_classes=num_classes,
                    drop_rate=drop_rate)
    
    # criterion
    df = pd.read_csv(os.path.join(dataset_path, "label/train.csv"))
    weights = compute_class_weight(class_weight = "balanced", classes=np.array(range(num_classes)), y = df["level_code"])
    weights = torch.FloatTensor(weights)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # opimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=weight_decay)
    
    # scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # train model
    train_model(model,
                dataloaders,
                criterion,
                optimizer,
                scheduler,
                device,
                save_path,
                num_epochs,
                resume=resume,
                start=start)
    
    # test model
    test(os.path.join(dataset_path, "test"),
         os.path.join(dataset_path, "test"),
         os.path.join("weights", save_path),
         model_name=model_name,
         device=device)

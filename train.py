# torch library
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from sklearn.utils.class_weight import compute_class_weight
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
from parallel import DataParallelModel, DataParallelCriterion

def train_model(model, 
                dataloaders, 
                criterion, 
                optimizer, 
                scheduler,
                device,
                save_path="", 
                num_epochs=25,
                batch_size=16,
                num_classes=4, 
                class_names=None,
                resume=False,
                use_wandb=False,
                start=0):
    
    n_gpus = torch.cuda.device_count()
    since = time.time()
    weights_path = os.path.join("weights", save_path)
    best_acc = 0
    max_acc = [0 for _ in range(num_classes)]
    
    if not os.path.isdir(weights_path):
        os.mkdir(weights_path)
    
    if use_wandb:
        wandb.watch(model, log='all')
    
    learning_dict = {"train" : [],
                     "val" : []}
    learning_columns = ["epoch"] \
                     + get_columns(num_classes) \
                     + ["best", "accuracy", "loss"]
    
    if resume or os.path.isfile(os.path.join(weights_path, "train.csv")):
        print("Restart training model saved in {}".format(save_path))
        if n_gpus > 1:
            model.load_state_dict(torch.load(os.path.join(weights_path, "last.pt"), \
                map_location={f"cuda:{i}": f"cuda:{i}" for i in range(n_gpus)}))
        else:
            model.load_state_dict(torch.load(os.path.join(weights_path, "last.pt")))
        
        _train = pd.read_csv(os.path.join(weights_path, "train.csv"))
        _val = pd.read_csv(os.path.join(weights_path, "val.csv"))
        # df = {"train" : _train, "val" : _val}
        for t, v in zip(_train.values, _val.values):
            learning_dict["train"].append(t)
            learning_dict["val"].append(v)

        best_acc = int(_val["best"].tolist()[-1])
        for i in range(num_classes):
            max_acc[i] = _val["acc_{}".format(i)].max()
        start = len(_train)

    loss_fn = copy.deepcopy(criterion)
    loss_fn = loss_fn.to(device)

    if n_gpus > 1:
        print("Training with multiple devices")
        model = DataParallelModel(model, device_ids=range(n_gpus))
        criterion = DataParallelCriterion(criterion, device_ids=range(n_gpus))

    model = model.to(device)
    criterion = criterion.to(device)
        
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    
    try:
        for epoch in range(start, num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 120)
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = running_corrects = 0
                epoch_labels = epoch_probs = epoch_preds = None
                
                gc.collect()
                torch.cuda.empty_cache()

                for inputs, labels in tqdm(dataloaders[phase]):
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):  
                        output = model(inputs) # 학습 시킬 사진들(batch_size만큼)을 모델에 넣고 outputs를 반환
                        outputs = None
                        for o in output:
                            if outputs is None: outputs = o.cuda()
                            else: outputs = torch.cat((outputs, o.cuda()),dim=0)
                        _, preds = torch.max(outputs, 1) # label과 같은 정수 형태로 반환(0 or 1 or 2....)
                        loss = criterion(output, labels) # loss 계산

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0) # loss 계산
                    running_corrects += torch.sum(preds == labels.cuda().data) # 정답이 맞은 개수 계산
                    
                    # 성능 검증을 위해 사용할 labels, probs, preds 저장
                    if epoch_labels is None:
                        epoch_labels = labels
                        epoch_probs = outputs
                        epoch_preds = preds
                    else:
                        epoch_labels = torch.cat((epoch_labels, labels),dim=0)
                        epoch_probs = torch.cat((epoch_probs, outputs),dim=0)
                        epoch_preds = torch.cat((epoch_preds, preds),dim=0)
                
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset) * 100
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                
                print("\n- {}".format(phase))
                row = [epoch]
                
                _, total_counts = torch.unique(epoch_labels, return_counts=True)
                
                # class별 accuracy, loss 계산
                for i in range(num_classes):
                    acc = loss = 0 
                    for prob, pred, label in zip(epoch_probs, epoch_preds, epoch_labels):
                        if label == i:
                            # loss += criterion(prob.reshape(1, -1), label.reshape(1))
                            loss += loss_fn(prob.float(), label.cuda())
                            if pred == i: acc += 1

                    acc = acc / total_counts[i] * 100
                    loss = loss / total_counts[i]
                    
                    if acc > max_acc[i] and phase == "val":
                        max_acc[i] = acc
                        if num_classes > 2:
                            torch.save(model.module.state_dict(), os.path.join(weights_path, "best_{}.pt".format(i)))
                        
                    if phase == "train": print("[{}] Accuracy : {:.6f} | Loss : {:.6f}".format(class_names[i], acc, loss))
                    elif phase == "val": print("[{}] Accuracy : {:.6f} | Loss : {:.6f} | Best : {:.6f}".format(class_names[i], acc, loss, max_acc[i]))
                    row.append(acc.item())
                    row.append(loss.item())
                            
                print()
                
                if phase == "train":
                    scheduler.step()
                    print("Total accuracy : {:.6f}%".format(epoch_acc))
                    print("Total loss     : {:.6f}".format(epoch_loss))
                    
                elif phase == "val":
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc.item()
                        torch.save(model.module.state_dict(), os.path.join(weights_path, "best.pt"))
                        
                    if use_wandb:
                        wandb.log({"pr" : wandb.plot.pr_curve(epoch_labels.cpu(), epoch_probs.cpu(), labels=class_names, classes_to_plot=None)})
                        wandb.log({"roc" : wandb.plot.roc_curve(epoch_labels.cpu(), epoch_probs.cpu(), labels=class_names, classes_to_plot=None)})
                        
                    print("Total accuracy : {:.6f}%".format(epoch_acc))
                    print("Best accuracy  : {:.6f}%".format(best_acc))
                    print("Total loss     : {:.6f}".format(epoch_loss))
                    
                row.append(best_acc)
                row.append(epoch_acc.item())
                row.append(epoch_loss)
                learning_dict[phase].append(row)
                    
            torch.save(model.module.state_dict(), os.path.join(weights_path, "last.pt"))
            train_dict = pd.DataFrame(learning_dict["train"], columns=learning_columns)
            val_dict = pd.DataFrame(learning_dict["val"], columns=learning_columns)
            
            try:
                train_dict.to_csv(os.path.join(weights_path, "train.csv"), index=False)
                val_dict.to_csv(os.path.join(weights_path, "val.csv"), index=False)
            except:
                print("Failed saving log files : permission denied")
                
            if use_wandb:
                log_to_wandb(train_dict, val_dict, num_classes)
                
            print()
            
    except KeyboardInterrupt:
        time_elapsed = time.time() - since
        print(f'Training stopped in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        torch.save(model.module.state_dict(), os.path.join(weights_path, "last.pt"))
        wandb.finish(1, quiet=True)
        
    # except torch.cuda.OutOfMemoryError:
    #     wandb.finish(1, quiet=True)
    #     print("Error : torch.cuda.OutOfMemoryError")
    #     exit()
    # except:
    #     time_elapsed = time.time() - since
    #     print(f'Training stopped in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    #     print(f'Best val Acc: {best_acc:4f}')
    #     torch.save(model.state_dict(), os.path.join(weights_path, "last.pt"))
    #     wandb.finish(1, quiet=True)

    time_elapsed = time.time() - since
    
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    learning_dict["train"] = pd.DataFrame(learning_dict["train"], columns=learning_columns)
    learning_dict["val"] = pd.DataFrame(learning_dict["val"], columns=learning_columns)
    learning_dict["train"].to_csv(os.path.join(weights_path, "train.csv"), index=False)
    learning_dict["val"].to_csv(os.path.join(weights_path, "val.csv"), index=False)
    torch.save(model.module.state_dict(), os.path.join(weights_path, "last.pt"))
    wandb.finish(1, quiet=True)
    
def train(dataset_path,
          save_path,
          model_name,
          num_epochs=25,
          input_size=600,
          class_names=["level 0", "level 1", "level 2", "level 3"],
          optimize="adam",
          learning_rate=0.0001,
          weight_decay=0.0005,
          drop_rate=0.2,
          batch_size=8,
          num_workers=10,
          resume=False,
          use_wandb=False,
          project_name=None,
          start=0):
    
    
    num_classes = len(class_names)
    if not os.path.isdir("weights"):
        os.mkdir("weights")
    if not os.path.isdir(os.path.join("weights", save_path)):
        os.mkdir(os.path.join("weights", save_path))
    if num_classes != len(class_names):
        print("length of class names must be equal to number of classes")
        exit()
    if use_wandb and project_name == None:
        print("Set project name for using wandb")
        exit()
            
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # device = torch.device("mps")
    
    print("save weights in {}....".format(save_path))
    print("save hyperparmeters......")
    save_params(os.path.join("weights", save_path),
                save_path,
                model_name,
                num_epochs,
                input_size,
                num_classes,
                optimize,
                learning_rate,
                weight_decay,
                drop_rate,
                batch_size,
                num_workers,
                use_wandb,
                project_name)
    
    # dataloaders
    trainloader, testloader = dataloader(os.path.join(dataset_path, "train"),
                                         os.path.join(dataset_path, "val"),
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
    if optimize=="adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=weight_decay)
    elif optimize=="sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
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
                batch_size,
                num_classes,
                class_names,
                resume=resume,
                use_wandb=use_wandb,
                start=start)

    wandb.finish(1, quiet=True)
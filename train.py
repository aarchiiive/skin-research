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
                resume=False,
                start=0):

    weights_path = os.path.join("weights", save_path)
    
    if not os.path.isdir(weights_path):
        os.mkdir(weights_path)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=range(8))
    
    since = time.time()
    best_acc = 0.0
    
    model = copy.deepcopy(model)
    model = model.to(device)
    wandb.watch(model, log='all')
    criterion = criterion.to(device)
    
    learning_dict = {"train" : [],
                     "val" : []}
    learning_columns = ["epoch"] \
                     + get_columns(num_classes) \
                     + ["best", "accuracy", "loss"]
    
    if resume or os.path.isfile(os.path.join(weights_path, "train.csv")):
        print("Restart training model....")
        model.load_state_dict(torch.load(os.path.join(weights_path, "last.pt")))
        
        _train = pd.read_csv(os.path.join(weights_path, "train.csv"))
        _val = pd.read_csv(os.path.join(weights_path, "val.csv"))
        # df = {"train" : _train, "val" : _val}
        for t, v in zip(_train.values, _val.values):
            learning_dict["train"].append(t)
            learning_dict["val"].append(v)

        start = len(_train)
        
    class_names = [str(i) for i in range(num_classes)]
    max_acc = {"train" : [0 for _ in range(num_classes)],
               "val" : [0 for _ in range(num_classes)]}  
    running_losses = {"train" : [0 for _ in range(num_classes)],
                      "val" : [0 for _ in range(num_classes)]}
    total_preds = {"train" : [0 for _ in range(num_classes)],
                  "val" : [0 for _ in range(num_classes)]}
    correct_preds = {"train" : [0 for _ in range(num_classes)],
                    "val" : [0 for _ in range(num_classes)]}
    
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
                
                for l in [running_losses, total_preds, correct_preds]:
                    for p in ["train", "val"]:
                        l[p] = [0 for _ in range(num_classes)]
                        
                # metrics = dict()
                running_labels = None
                running_probs = None
                running_preds = None
                
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
                        
                        for label, output, pred in zip(labels, outputs, preds):
                            if label == pred:
                                correct_preds[phase][label] += 1
                            total_preds[phase][label] += 1
                            running_losses[phase][label] += criterion(output, label.long())

                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                        else:
                            if running_labels is None:
                                running_labels = labels
                                running_probs = outputs
                                running_preds = preds
                            else:
                                running_labels = torch.cat((running_labels, labels),dim=0)
                                running_probs = torch.cat((running_probs, outputs),dim=0)
                                running_preds = torch.cat((running_preds, preds),dim=0)
                        
                    running_loss += loss.item() * inputs.size(0)
                    
                print("\n[{}]".format(phase))
                
                row = [epoch]
                
                for i, correct_count in enumerate(correct_preds[phase]):
                    accuracy = 100 * float(correct_count) / total_preds[phase][i]
                    loss = running_losses[phase][i].item() / total_preds[phase][i]
                    
                    if accuracy > max_acc[phase][i]:
                        max_acc[phase][i] = accuracy
                        torch.save(model.state_dict(), os.path.join(weights_path, "best_{}.pt".format(i)))
                        
                    row.append(accuracy)
                    row.append(loss)
                    # if phase == "val":
                    #     metrics['val_acc_{}'.format(i)] = accuracy
                    #     metrics['val_loss_{}'.format(i)] = loss
                    
                    print("Accuracy for class(level {}) : {:.4f}% ({}/{}), best : {:.4f}%, loss : {:.4f}".\
                        format(i, accuracy, correct_count, 
                               total_preds[phase][i], max_acc[phase][i], loss))    
                
                total = sum(correct_preds[phase]) / sum(total_preds[phase]) * 100
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                if total > best_acc  and phase == "val":
                    best_acc = total
                    torch.save(model.state_dict(), os.path.join(weights_path, "best.pt"))
                
                row.append(best_acc)
                row.append(total)
                row.append(epoch_loss)
                
                if phase == "val":
                    # metrics["val_acc"] = total
                    # metrics["val_loss"] = epoch_loss
                    # wandb.log(metrics, step=epoch)
                    wandb.log({"pr" : wandb.plot.pr_curve(running_labels.cpu(), running_probs.cpu(), labels=class_names, classes_to_plot=None)})
                    wandb.log({"roc" : wandb.plot.roc_curve(running_labels.cpu(), running_probs.cpu(), labels=class_names, classes_to_plot=None)})
                    print("Best accuracy  : {:.4f}%".format(best_acc))
                else:
                    pass
                    # metrics["train_acc"] = total
                    # metrics["train_loss"] = epoch_loss
                    # wandb.log(metrics, step=epoch)
                    
                print("Total accuracy : {:.4f}%".format(total))
                print("Total loss     : {:.4f}%".format(epoch_loss))
                
                if phase == "train":
                    scheduler.step()
            
                learning_dict[phase].append(row)
                
            torch.save(model.state_dict(), os.path.join(weights_path, "last.pt"))
            train_dict = pd.DataFrame(learning_dict["train"], columns=learning_columns)
            val_dict = pd.DataFrame(learning_dict["val"], columns=learning_columns)
            train_dict.to_csv(os.path.join(weights_path, "train.csv"), index=False)
            val_dict.to_csv(os.path.join(weights_path, "val.csv"), index=False)
            log_to_wandb(train_dict, val_dict, num_classes)
                
            print()
            
    except KeyboardInterrupt:
        time_elapsed = time.time() - since
        print(f'Training stopped in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        torch.save(model.state_dict(), os.path.join(weights_path, "last.pt"))
        
    except torch.cuda.OutOfMemoryError:
        if epoch == 0:
            os.remove(os.path.join(weights_path, "log.txt"))
        print("Error : torch.cuda.OutOfMemoryError")
        exit()
        
    finally:
        time_elapsed = time.time() - since
        print(f'Training stopped in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        torch.save(model.state_dict(), os.path.join(weights_path, "last.pt"))

    time_elapsed = time.time() - since
    
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    learning_dict["train"] = pd.DataFrame(learning_dict["train"], columns=learning_columns)
    learning_dict["val"] = pd.DataFrame(learning_dict["val"], columns=learning_columns)
    learning_dict["train"].to_csv(os.path.join(weights_path, "train.csv"), index=False)
    learning_dict["val"].to_csv(os.path.join(weights_path, "valid.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(weights_path, "last.pt"))
    
def train(dataset_path,
          save_path,
          model_name,
          num_epochs=25,
          input_size=600,
          num_classes=4,
          optimize="adam",
          learning_rate=0.0001,
          weight_decay=0.0005,
          drop_rate=0.2,
          batch_size=8,
          num_workers=10,
          resume=False,
          start=0):

    
    if not os.path.isdir("weights"):
        os.mkdir("weights")
    if not os.path.isdir(os.path.join("weights", save_path)):
        os.mkdir(os.path.join("weights", save_path))
    
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
                num_workers)
    
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
                resume=resume,
                start=start)
    
    wandb.finish(1, quiet=True)

import torch
import torch.nn as nn

import gc
import time
import wandb
import pickle
from tqdm import tqdm
import operator
from sklearn import metrics

from utils import *
from model import _models

def hard_voting(probs):
    probs = torch.stack(probs, dim=1)
    probs = probs.permute(0, 2, 1)
    probs, _ = torch.max(probs, dim=2)
    _, preds = torch.max(probs, dim=1)
    return probs.numpy(), preds.numpy()
    
def soft_voting(probs):
    probs = torch.stack(probs, dim=1)
    probs = probs.permute(0, 2, 1)
    probs = torch.mean(probs, dim=2)
    _, preds = torch.max(probs, dim=1)
    return probs.numpy(), preds.numpy()
    
def log(project_name, preds, probs, labels, model_list, num_classes, class_names, mode="hard", use_wandb=False):
    print(f"[{mode} voting]") 
    accuracy = metrics.accuracy_score(labels, preds)
    f1_score = metrics.f1_score(labels, preds, average=None)
    confusion_matrix = metrics.multilabel_confusion_matrix(labels, preds, labels=range(num_classes))
    _, total_counts = np.unique(labels, return_counts=True)
    _sensitivity = _specificity = _precision = _recall = _fpr = _fnr = _auc = _acc = []
    
    for i in range(num_classes):
        acc = 0 
        for pred, label in zip(preds, labels):
            if label == i:
                if pred == i: acc += 1
        acc = acc / total_counts[i] * 100
        
        print(f"- class {i}")
        [tn, fp], [fn, tp] = confusion_matrix[i]
        fpr, tpr, _ = metrics.roc_curve(labels, probs[:, i], pos_label=i)
        auc = metrics.auc(fpr, tpr)
        print("Sensitivity : {}".format((tp/(tp+fn))*100))
        print("Specificity : {}".format((tn/(tn+fp))*100))
        print("Precision   : {}".format((tp/(tp+fp))*100))
        print("Recall      : {}".format((tp/(tp+fn))*100))
        print("FPR         : {}".format((fp/(tn+fp))*100))
        print("FNR         : {}".format((fn/(tp+fn))*100))
        print("AUC         : {}".format(auc*100))
        print("F1 score    : {}".format(f1_score[i]*100))
        print("Accuracy    : {}".format(acc))
        _sensitivity.append((tp/(tp+fn))*100)
        _specificity.append((tn/(tn+fp))*100)
        _precision.append((tp/(tp+fp))*100)
        _recall.append((tp/(tp+fn))*100)
        _fpr.append((fp/(tn+fp))*100)
        _fnr.append((fn/(tp+fn))*100)
        _auc.append(auc*100)
        _acc.append(acc*100)
    # print("95% confidence interval of AUC : {}".format(confidence_interval(auc, labels)))
    print()

    if use_wandb:
        wandb.init(project=project_name, name="+".join(model_list)+"({})".format(mode), tags=model_list, group="ensemble")
        table = wandb.Table(data=[[x, y] for (x, y) in \
                              zip(class_names, f1_score)], columns = ["label", "f1 score"])
        wandb.log({"f1 score/class" : wandb.plot.bar(table, "label", "f1 score", title="F1 score")})
        table = wandb.Table(data=[[x, y] for (x, y) in \
                              zip(class_names, _sensitivity)], columns = ["label", "sensitivity"])
        wandb.log({"sensitivity/class" : wandb.plot.bar(table, "label", "sensitivity", title="Sensitivity")})
        table = wandb.Table(data=[[x, y] for (x, y) in \
                              zip(class_names, _specificity)], columns = ["label", "specificity"])
        wandb.log({"specificity/class" : wandb.plot.bar(table, "label", "specificity", title="Specificity")})
        table = wandb.Table(data=[[x, y] for (x, y) in \
                              zip(class_names, _precision)], columns = ["label", "precision"])
        wandb.log({"precision/class" : wandb.plot.bar(table, "label", "precision", title="Precision")})
        table = wandb.Table(data=[[x, y] for (x, y) in \
                              zip(class_names, _recall)], columns = ["label", "recall"])
        wandb.log({"recall/class" : wandb.plot.bar(table, "label", "recall", title="Recall")})
        table = wandb.Table(data=[[x, y] for (x, y) in \
                              zip(class_names, _fpr)], columns = ["label", "fpr"])
        wandb.log({"fpr/class" : wandb.plot.bar(table, "label", "fpr", title="FPR")})
        table = wandb.Table(data=[[x, y] for (x, y) in \
                              zip(class_names, _fnr)], columns = ["label", "fnr"])
        wandb.log({"fnr/class" : wandb.plot.bar(table, "label", "fnr", title="FNR")})
        table = wandb.Table(data=[[x, y] for (x, y) in \
                              zip(class_names, _auc)], columns = ["label", "auc"])
        wandb.log({"auc/class" : wandb.plot.bar(table, "label", "auc", title="AUC")})
        table = wandb.Table(data=[[x, y] for (x, y) in \
                              zip(class_names, _acc)], columns = ["label", "accuracy"])
        wandb.log({"accuracy/class" : wandb.plot.bar(table, "label", "accuracy", title="Accuracy")})
        
        wandb.log({"accuracy" : accuracy*100})
        wandb.log({"f1 score" :  metrics.f1_score(labels, preds, average="micro")*100})
        wandb.log({"confusion matrix" : wandb.plot.confusion_matrix(probs=probs, y_true=labels, preds=None, class_names=class_names)})
        wandb.log({"pr" : wandb.plot.pr_curve(labels, probs, labels=class_names)})
        wandb.log({"roc" : wandb.plot.roc_curve(labels, probs, labels=class_names)})
        wandb.finish(1, quiet=True)

def ensemble(weights_paths, 
             class_names, 
             project_name=None, 
             use_wandb=True,
             select="best"):
    
    probs = []
    preds = []
    labels = None
    num_classes = len(class_names)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    testloader = get_testloader("data/test/", "data/label/test.csv", input_size=600, batch_size=16)
    model_list = [pd.read_csv(os.path.join("weights", w, "hyperparameters.csv"))["model_name"][0] for w in weights_paths]
    models = [_models(m, num_classes=num_classes, drop_rate=0) for m in model_list]

    for p, model in zip(weights_paths, models):
        if select == "last":
            model.load_state_dict(torch.load(os.path.join("weights", p, "last.pt")))
        else:
            model.load_state_dict(torch.load(os.path.join("weights", p, "best.pt")))
            
        if not os.path.isfile(os.path.join("weights", p, "preds.pickle")):
            _probs, _preds, labels = predictions(model, testloader, device)
            with open(os.path.join("weights", p, "preds.pickle"), 'wb') as f:
                pickle.dump([_probs, _preds, labels], f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(os.path.join("weights", p, "preds.pickle"), 'rb') as f:
                _probs, _preds, labels = pickle.load(f)
                
        probs.append(_probs)
        preds.append(_preds)
        
    print("[single model accuracy]")
    for i, m in enumerate(model_list):
        print("{} : {}".format(m, metrics.accuracy_score(labels, preds[i].ravel())))
    print()
    
    hard_probs, hard_preds = hard_voting(probs)
    soft_probs, soft_preds = soft_voting(probs)
    
    log(project_name, hard_preds, hard_probs, labels, model_list, num_classes, class_names, mode="hard", use_wandb=use_wandb)
    log(project_name, soft_preds, soft_probs, labels, model_list, num_classes, class_names, mode="soft", use_wandb=use_wandb)
    
if __name__ == "__main__":
    """_summary_

    Args:
        weights_paths (List): weights(.pt) 파일이 저장된 폴더이름들(학습 때 넣었던 폴더 이름과 동일) -> 두개 이상 넣으면 됨
        class_names (List): label의 이름이 담긴 리스트
        project_name (str, optional): wandb에 사용할 프로젝트 이름. Defaults to None.
        use_wandb (bool, optional): wandb를 사용한다면 True. Defaults to True.
        select (str, optional): "best"(가장 나은 가중치)와 "last"(학습 중 마지막 가중치)중 선택. Defaults to "best".
    """
    project_name = "skin"
    class_names = ["level 0", "level 1", "level 2", "level 3"]
    ensemble(["lab01", "lab02", "lab03"], class_names, project_name=project_name, use_wandb=True) # using wandb
    ensemble(["lab01", "lab02"], class_names, project_name=project_name, use_wandb=True) # using wandb
    ensemble(["lab01", "lab02", "lab03"], class_names) # without wandb
    
    
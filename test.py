import torch
import numpy as np
from sklearn import metrics
import pickle

from utils import *
from model import _models

def log(project_name, weights_path, preds, probs, labels, model_name, num_classes, class_names, use_wandb=False):
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
        
    print("- total")
    print("F1 score    : {}".format(metrics.f1_score(labels, preds, average="micro")*100))
    print("Accuracy    : {}".format(accuracy*100))
    # print("95% confidence interval of AUC : {}".format(confidence_interval(auc, labels)))
    print()

    if use_wandb:
        wandb.init(project=project_name, name=weights_path+"-"+model_name, tags=[model_name], group="single")
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

def test_model(weights_path, 
               class_names, 
               project_name=None, 
               use_wandb=True,
               select="best"):
    
    num_classes = len(class_names)
    model_name = pd.read_csv(os.path.join("weights", weights_path, "hyperparameters.csv"))["model_name"].item()
    model = _models(model_name, num_classes, drop_rate=0)
    
    if select == "last":
        model.load_state_dict(torch.load(os.path.join("weights", weights_path, "last.pt")))
    else:
        model.load_state_dict(torch.load(os.path.join("weights", weights_path, "best.pt")))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    testloader = get_testloader("data/test/", "data/label/test.csv", input_size=600, batch_size=16)
    model = model.to(device)
    
    if not os.path.isfile(os.path.join("weights", weights_path, "preds.pickle")):
        probs, preds, labels = predictions(model, testloader, device)
        with open(os.path.join("weights", weights_path, "preds.pickle"), 'wb') as f:
            pickle.dump([probs, preds, labels], f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(os.path.join("weights", weights_path, "preds.pickle"), 'rb') as f:
            probs, preds, labels = pickle.load(f)
            
    probs = probs.numpy()
    preds = preds.numpy()
            
    print("{} : {}".format(model_name, metrics.accuracy_score(labels, preds.ravel())))
    log(project_name, weights_path, preds, probs, labels, model_name, num_classes, class_names, use_wandb)

if __name__ == "__main__":
    """_summary_

    Args:
        weights_paths (List): weights(.pt) 파일이 저장된 폴더이름
        class_names (List): label의 이름이 담긴 리스트
        project_name (str, optional): wandb에 사용할 프로젝트 이름. Defaults to None.
        use_wandb (bool, optional): wandb를 사용한다면 True. Defaults to True.
        select (str, optional): "best"(가장 나은 가중치)와 "last"(학습 중 마지막 가중치)중 선택. Defaults to "best".
    """
    project_name = "skin"
    class_names = ["level 0", "level 1", "level 2", "level 3"]
    test_model("lab01", class_names, project_name, use_wandb=True)
    
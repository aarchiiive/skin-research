from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F

from utils import *
from model import _models


def test(image_path,
         label_path,
         save_path, 
         model_name, 
         device,
         num_classes=4):
    
    print(save_path)
    
    dataloaders = {"test" : get_testloader(image_path, label_path)} # test set 이미지와 label 경로
        
    # model
    model = _models(model_name=model_name, num_classes=num_classes)
    model = model.to(device)
    
    # 학습한 resnet 가중치를 불러옴
    model.load_state_dict(torch.load(os.path.join(save_path, "last.pt"))) 
    model.eval()
    
    CM = 0
    y_test = []
    y_score = []

    with torch.no_grad():
        for data in tqdm(dataloaders['test']):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images) #file_name
            preds = torch.argmax(outputs.data, 1)
            CM += confusion_matrix(labels.cpu(), preds.cpu(),labels=[0,1])
            labels = labels.cpu().numpy()
            preds = preds.cpu().numpy()
            for l, p in zip(labels, preds):
                y_test.append(l)
                y_score.append(p)
            
        tn = CM[0][0]
        tp = CM[1][1]
        fp = CM[0][1]
        fn = CM[1][0]
        acc = np.sum(np.diag(CM)/np.sum(CM))
        sensitivity = tp/(tp+fn)
        precision = tp/(tp+fp)

        print('\nTest set Accuracy(mean): %f %%' % (100 * acc))
        print()
        print('[Confusion Matrix]')
        print(CM)
        print()
        print('- Sensitivity : ',(tp/(tp+fn))*100)
        print('- Specificity : ',(tn/(tn+fp))*100)
        print('- Precision   : ',(tp/(tp+fp))*100)
        print('- NPV         : ',(tn/(tn+fn))*100)
        print('- F1          : ',((2*sensitivity*precision)/(sensitivity+precision))*100)
        print('- FPR         : ', (fp/(tn+fp))*100)
        print('- FNR         : ', (fn/(tp+fn))*100)
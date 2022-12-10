import timm
import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary
import torch.nn.functional as F

def set_parameter_requires_grad(model, feature_extracting=True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False
            
    return model
            

def _models(model_name, 
            num_classes=4, 
            drop_rate=0.2, 
            feature_extract=True):

    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=True)
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=True)
    elif model_name == "resnext50":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
    elif model_name == "resnext101":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d', pretrained=True)
    elif "densenet" in model_name:
        if model_name == "densenet121":
            model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True, drop_rate=drop_rate)
        elif model_name == "densenet161":
            model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet161', pretrained=True, drop_rate=drop_rate)
        elif model_name == "densenet169":
            model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet169', pretrained=True, drop_rate=drop_rate)
        elif model_name == "densenet201":
            model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True, drop_rate=drop_rate)
            
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        
        return model
            
    elif 'efficientnet' in model_name:
        if model_name == 'efficientnet_b5':
            model = models.efficientnet_b5(weights='IMAGENET1K_V1', drop_rate=drop_rate)
        elif model_name == 'efficientnet_b6':
            model = models.efficientnet_b6(weights='IMAGENET1K_V1', drop_rate=drop_rate)
        elif model_name == 'efficientnet_b7':
            model = models.efficientnet_b7(weights='IMAGENET1K_V1', drop_rate=drop_rate)
        elif model_name == 'efficientnet_v2_s':
            model = models.efficientnet_v2_s(weights='IMAGENET1K_V1', drop_rate=drop_rate)
        elif model_name == 'efficientnet_v2_m':
            model = models.efficientnet_v2_m(weights='IMAGENET1K_V1', drop_rate=drop_rate)
        elif model_name == 'efficientnet_v2_l':
            model = models.efficientnet_v2_l(weights='IMAGENET1K_V1', drop_rate=drop_rate)
            
        model = set_parameter_requires_grad(model, feature_extract)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, out_features=num_classes, bias=True)
        
        return model

    elif 'xception' in model_name:
        if model_name == 'xception':
            model = timm.create_model('xception', pretrained=True, drop_rate=drop_rate)
        elif model_name == 'xception41':
            model = timm.create_model('xception41', pretrained=True, drop_rate=drop_rate)
        elif model_name == 'xception41p':
            model = timm.create_model('xception41p', pretrained=True, drop_rate=drop_rate)
        elif model_name == 'xception65':
            model = timm.create_model('xception65', pretrained=True, drop_rate=drop_rate)
        elif model_name == 'xception65p':
            model = timm.create_model('xception65p', pretrained=True, drop_rate=drop_rate)
        elif model_name == 'xception71':
            model = timm.create_model('xception71', pretrained=True, drop_rate=drop_rate)
            
        model = set_parameter_requires_grad(model, feature_extract)
        
        if model_name == 'xception':
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, out_features=num_classes, bias=True)
        else:
            in_features = model.head.fc.in_features
            model.head.fc = nn.Linear(in_features, out_features=num_classes, bias=True)
            
        return model
            

    # tuning resnet, resnext
    # layer를 freeze
    model = set_parameter_requires_grad(model, feature_extract)
    
    # 마지막 fc layer를 변경
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(drop_rate),
        nn.Linear(num_ftrs, num_classes)
    )


    return model




if __name__ == "__main__":
    num_classes = 2
    
    model = _models("xception41")
    print(model)

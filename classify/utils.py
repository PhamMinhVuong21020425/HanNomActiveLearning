import os
import torch
import matplotlib.pyplot as plt
from typing import Type

from classify.hyperparams import *
from classify.config import device
from classify.models import EfficientNetB7_Dropout

from sampling_strategies import (
    BaseStrategy,
    EntropySampling,
    LeastConfidenceSampling,
    MarginSampling,
    RandomSampling,
    RatioSampling,
    DropoutSampling,
    BALDSampling,
)


def get_image_paths(data_dir: str, phase='train'):
    image_paths = []
    phase_dir = os.path.join(data_dir, phase)
    for label_name in os.listdir(phase_dir):
        laber_dir = os.path.join(phase_dir, label_name)
        for image_file in os.listdir(laber_dir):
            image_path = os.path.join(laber_dir, image_file).replace("\\", "/")
            image_paths.append(image_path)

    return image_paths


def get_model():
    # model = models.efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
    # model.features[0][0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    # model.classifier[1] = nn.Linear(in_features=2560, out_features=2139, bias=True)
    
    model = EfficientNetB7_Dropout(num_classes=2139)

    # model = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
    # model.classifier[6] = nn.Linear(in_features=4096, out_features=2139, bias=True)

    model.to(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    return model


def get_strategy(name: str) -> Type[BaseStrategy]:
    strategies = {
        "RandomSampling": RandomSampling,
        "EntropySampling": EntropySampling,
        "RatioSampling": RatioSampling,
        "MarginSampling": MarginSampling,
        "LeastConfidenceSampling": LeastConfidenceSampling,
        "DropoutSampling": DropoutSampling,
        "BALDSampling": BALDSampling,
    }

    strategy = strategies.get(name)
    if not strategy:
        raise NotImplementedError(f"Strategy '{name}' is not implemented")
    
    return strategy


def save_check_point(epochs, model, optimizer, criterion, CHECK_POINT_PATH):
    if not os.path.exists(CHECK_POINT_PATH):
        os.mkdir(CHECK_POINT_PATH)

    torch.save({
                  'epoch': epochs,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': criterion,
               }, CHECK_POINT_PATH)


def save_model(model, MODEL_PATH):
    directory = os.path.dirname(MODEL_PATH)
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(model.state_dict(), MODEL_PATH)


def load_model(model, MODEL_PATH):
    weights = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(weights)
    model.to(device)

    # print(model)
    # for name, param in model.named_parameters():
    #     print(name, param)

    return model


def load_check_point(model, optimizer, CHECK_POINT_PATH):
    check_point = torch.load(CHECK_POINT_PATH, map_location=device)
    model.load_state_dict(check_point['model_state_dict'])
    optimizer.load_state_dict(check_point['optimizer_state_dict'])

    return model, optimizer, check_point['epoch'], check_point['loss']

def imshow(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    npimg = img.numpy().transpose(1, 2, 0)
    npimg = npimg * std + mean     # unnormalize
    npimg = npimg.clip(0, 1)
    plt.imshow(npimg)
    plt.show()

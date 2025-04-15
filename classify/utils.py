import os
import torch
import matplotlib.pyplot as plt
from typing import Type

from .hyperparams import *
from .config import device
from .models import EfficientNetB7_Dropout

from sampling_strategies import (
    BaseStrategy,
    EntropySampling,
    LeastConfidenceSampling,
    MarginSampling,
    RandomSampling,
    RatioSampling,
    DropoutSampling,
    BALDSampling,
    BatchBALDSampling,
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


def get_model(num_classes=2139):
    # model = models.efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
    # model.features[0][0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    # model.classifier[1] = nn.Linear(in_features=2560, out_features=2139, bias=True)
    
    model = EfficientNetB7_Dropout(num_classes)

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
        "BatchBALDSampling": BatchBALDSampling,
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


def load_dynamic_model(model, model_path, num_classes, pretrained_n_classes):
    print(f"[*] Pretrained model num_classes: {pretrained_n_classes}, Current model num_classes: {num_classes}")
    
    # Load raw state_dict
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    if num_classes != pretrained_n_classes:
        print("[!] Number of classes differs — loading partial weights (excluding final layer)")
        model_state = model.state_dict()
        filtered_dict = {}

        for k, v in state_dict.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered_dict[k] = v
            else:
                print(f"[!] Skipping: {k} | pre-trained shape: {v.shape}, model shape: {model_state.get(k, None)}")

        model.load_state_dict(filtered_dict, strict=False)
    else:
        print("[*] Number of classes matches — loading all weights")
        model.load_state_dict(state_dict)

    model.to(device)
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

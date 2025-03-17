import json
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image

from hyperparams import *
from utils import load_model
from transform import ImageTransform
from models import EfficientNetB7_Dropout


resize = 64
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = ImageTransform(resize, mean, std)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare network
model = EfficientNetB7_Dropout(num_classes=2139)

# prepare model
model = load_model(model, f'{WEIGHT_DIR}/efficientnet_b7_last.pth')
model.eval()

# read json file
with open(f'{DATA_DIR}/map/mapping.json', mode='r', encoding='utf-8') as file:
    class_index = json.load(file)

    # print class index have duplicate value
    cnt = 0
    for key, value in class_index.items():
        amount = list(class_index.values()).count(value)
        if amount > 1:
            cnt += 1
            print(f'Class: {value}, Index: {key}, Amount: {amount}')
    
    print(f'Total duplicate class: {cnt}')

class Predictor():
    def __init__(self, class_index):
        self.clas_index = class_index

    def predict(self, img):
        # Calculate time of prediction
        start = time.time()

        # Load and preprocess the input image
        img = Image.open(img).convert('RGB')
        img_tensor = transform(img, phase='val').unsqueeze(0) # (channel, height, width) -> (1, channel, height, width)

        # Make a prediction
        with torch.no_grad():
            output = model(img_tensor.to(device))

        # Calculate probabilities using softmax
        probabilities = F.softmax(output, dim=1)
        prob, predict = torch.max(probabilities, 1) # torch.max return tuple (values, indices)

        class_id = predict.item()
        predicted_label = self.clas_index[str(class_id)]

        # Calculate time of prediction
        end = time.time()

        # Get top 5 probabilities and indices
        top5_probs, top5_indices = probabilities.sort(dim=1, descending=True)
        top5_probs = top5_probs[0][:5]
        top5_indices = top5_indices[0][:5]

        # Print top 5 classes and probabilities
        for i in range(5):
            print(f"Class: {self.clas_index[str(top5_indices[i].item())]}, Probability: {top5_probs[i].item():.4f}")

        # Print prediction and probabilities
        print(f'Predict: {predicted_label}, Probability: {prob.item()}, Time Infer: {end - start:.4f}s')

        # Display the image with prediction
        img = transform(img, phase='val')
        img = img.numpy().transpose(1, 2, 0)
        img = img.clip(0, 1) # Giới hạn giá trị trong khoảng [0, 1]
        rcParams['font.family'] = 'Meiryo'
        plt.imshow(img)
        plt.title(f'Predicted: {predicted_label}, Class Id: {class_id}, Probability: {prob.item():.4f}')
        plt.axis('off')
        plt.show()

        return predicted_label, prob.item(), end - start

if __name__ == '__main__':
    # Predict on an image
    predictor = Predictor(class_index)
    image_path = './data/images/chi.png'
    predictor.predict(image_path)

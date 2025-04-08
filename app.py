# This file is part of https://github.com/jainamoswal/Flask-Example.
# Usage covered in <IDC lICENSE>
# Jainam Oswal. <jainam.me> 


# Import Libraries 
import os
import sys
import json
import time
import yaml
import datetime
import zipfile
import shutil
import random
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from typing import Literal
from tqdm import tqdm
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS, cross_origin 
from pyngrok import ngrok
from dotenv import load_dotenv

from firebase import db
from predict import Predictor
from yolov5.train import main, parse_opt
from classify.transform import ImageTransform
from classify.dataset import SinoNomDataset, ActiveLearningDataset
from classify.models import ActiveLearningNet
from classify.config import *
from classify.utils import *

load_dotenv()
predictor = Predictor()

# Image transforms
resize = 64
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = ImageTransform(resize, mean, std)

# read json file
with open(f'{HOME}/classify/mapping.json', mode='r', encoding='utf-8') as file:
    class_index = json.load(file)
    label_to_index = {v: k for k, v in class_index.items()}
    existing_labels = set(class_index.values())

# Active learning strategies
strategy_names = [
    "RandomSampling",
    "LeastConfidenceSampling",
    "MarginSampling",
    "EntropySampling",
    "RatioSampling",
    "DropoutSampling",
    "BALDSampling",
    "BatchBALDSampling",
]

# Define app.
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
DATASET_FOLDER = "datasets"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Config CORS.
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["SECRET_KEY"] = os.environ.get('SECRET_KEY')

# Define root route.
@app.route("/", methods=['GET'])
@cross_origin() # Allow Cross-Origin Resource Sharing
def home():
    return jsonify({
        "data": "We support Han Nom image detection and recognition.",
    })

@app.route('/favicon.ico')
def favicon():
    path = os.path.join(app.root_path, 'static')
    return send_from_directory(path, 'favicon.ico')

@app.route("/test", methods=['GET'])
@cross_origin()
def test():
    source = './yolov5/data/images/test4.png'
    json_objects = predictor.predict(source, classify=False)
    return jsonify(json_objects)

@app.route("/api/detect", methods=['POST'])
@cross_origin()
def object_detection():
    start = time.time()
    files = request.files.getlist("files")

    output = []
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, f'temp.{file.filename.split(".")[-1]}')
        file.save(file_path)

        json_objects = predictor.predict(source=file_path, classify=True, batch_size=64)
        output.append({
            'objects_detection': json_objects,
            'url_image': os.path.join(UPLOAD_FOLDER, file.filename),
            'image_name': file.filename
        })

        # Remove image after processing
        os.remove(file_path)

    end = time.time()
    print(f"Time taken: {end - start:.3f}s")
    return jsonify(output)

@app.route("/api/classify", methods=['POST'])
@cross_origin()
def image_classification():
    start = time.time()
    img = request.files["img"]
    
    img_path = os.path.join(UPLOAD_FOLDER, f'temp.{img.filename.split(".")[-1]}')
    img.save(img_path)

    class_id, predicted_label, confidence = predictor.classify_single(img_path)

    # Remove image after processing
    os.remove(img_path)

    end = time.time()
    print(f"Time taken: {end - start:.3f}s")
    return jsonify({
        "class_id": class_id,
        "predicted_label": predicted_label,
        "confidence": confidence,
    })

@app.route("/api/active-learning", methods=['POST'])
@cross_origin()
def active_learning():
    try:
        # Parse request parameters with defaults
        data = request.form.to_dict()
        
        # Validate and extract parameters
        strategy_name = data.get('strategy')
        n_samples = int(data.get('n_samples'))
        model_infer = data.get('modelInference')
        
        # Validate strategy
        if strategy_name not in strategy_names:
            return jsonify({
                "status": "error",
                "message": f"Invalid strategy. Supported strategies are: {strategy_names}"
            }), 200
        
        pool = request.files['pool']
        pool_name = data.get('poolName')
        pool_path = os.path.join(DATASET_FOLDER, pool.filename)
        pool.save(pool_path)
        print("Received pool ZIP file:", pool.filename)

        # Unzip the pool data
        extract_path = os.path.join(DATASET_FOLDER, pool_name, 'train', '0')
        os.makedirs(extract_path, exist_ok=True)
        
        with zipfile.ZipFile(pool_path, 'r') as zip_ref:
            for member in zip_ref.infolist():
                filename = member.filename
                ext = filename.split('.')[-1].lower()
                if ext in {'jpg', 'jpeg', 'png'}:
                    with zip_ref.open(member) as source, open(os.path.join(extract_path, os.path.basename(filename)), 'wb') as target:
                        target.write(source.read())

        # Remove temporary files
        os.remove(pool_path)
        print(f"[*] Unzipped pool data to {extract_path}")

        # Use existing classification training approach
        data_path = os.path.join(DATASET_FOLDER, pool_name)
        data_list = get_image_paths(data_path, phase='train')
        
        # Create datasets
        dataset = SinoNomDataset(data_list, transform=transform, phase='train')
        
        # Create active learning dataset
        al_dataset = ActiveLearningDataset(dataset, initial_labeled=0)
        
        # Initialize model and active learning components
        net = ActiveLearningNet(
            model=load_model(get_model(), f'{KAGGLE_DIR}/efficientnet_b7_best.pth'),
            device=device,
            criterion=nn.CrossEntropyLoss(),
            optimizer_cls=optim.AdamW,
            optimizer_params={
                "lr": LEARNING_RATE, 
                "betas": (0.9, 0.999), 
                "eps": 1e-9, 
                "weight_decay": 0.0
            },
        )
        
        # Get the specified strategy
        strategy = get_strategy(strategy_name)(al_dataset, net)
        
        # Get images need to label using active learning
        query_indices = strategy.query(n_samples)
        strategy.dataset.label_samples(query_indices)

        # Get list path of labeled and unlabeled data
        labeled_idxs, _ = strategy.dataset.get_labeled_data()
        labeled_images = [dataset.file_list[int(i)] for i in labeled_idxs]

        # Zip the labeled images for download
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_file_path = os.path.join(DATASET_FOLDER, f'labeled_images_{timestamp}.zip')

        with zipfile.ZipFile(zip_file_path, 'w') as zipf:
            for img_path in labeled_images:
                filename = os.path.basename(img_path)
                zipf.write(img_path, filename)

        print(f"[*] Labeled images saved to {zip_file_path}")

        return jsonify({
            "status": "success",
            "message": f"Active Learning with {strategy_name} completed successfully.",
            "strategy": strategy_name,
            "samples": n_samples,
            "labeled_images_path": zip_file_path,
        }), 200
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": str(sys.exc_info())
        }), 200

def create_yaml_config(dataset_name, num_classes, class_names, train_path, val_path):
    """
    Create a dataset configuration YAML file for YOLOv5 training
    """
    config = {
        'path': os.path.abspath(os.path.join(DATASET_FOLDER, dataset_name)),
        'train': train_path,
        'val': val_path,
        'nc': num_classes,
        'names': class_names
    }
    
    config_path = os.path.join(DATASET_FOLDER, f'{dataset_name}.yaml')
    with open(config_path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    
    return config_path

@app.route("/api/train/detection", methods=['POST'])
@cross_origin()
def train_detection():
    # Get training parameters from form data
    data = request.form.to_dict()
    model_name = data.get('modelName', 'yolov5s')
    pretrained_model = data.get('pretrainedModel', 'yolov5s.pt')
    num_classes = 1
    class_names = ['character']
    
    # Additional optional training parameters
    epochs = int(data.get('epochs', 100))
    batch_size = int(data.get('batchSize', 16))

    # Get dataset file from request
    dataset = request.files['dataset']
    dataset_name = dataset.filename.split('.')[0]
    dataset_path = os.path.join(DATASET_FOLDER, dataset.filename)
    dataset.save(dataset_path)
    print("Received dataset ZIP file:", dataset.filename)

    extract_path = os.path.join(DATASET_FOLDER, dataset_name)
    temp_extract_path = os.path.join(DATASET_FOLDER, f"temp_{dataset_name}")
    
    os.makedirs(extract_path, exist_ok=True)
    os.makedirs(temp_extract_path, exist_ok=True)
    
    # Unzip the dataset
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_path)
    
    image_exts = {'jpg', 'jpeg', 'png'}
    image_files = []
    
    for root, _, files in os.walk(temp_extract_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = file.split('.')[-1].lower()
            if ext in image_exts:
                image_files.append(file_path)
    
    # Shuffle and split dataset into train and validation sets
    random.seed(42)
    random.shuffle(image_files)
    split_idx = int(0.9 * len(image_files))  # 90% train, 10% val
    train_images, val_images = image_files[:split_idx], image_files[split_idx:]
    
    # Create directories for train and val images and labels
    train_image_dir = os.path.join(extract_path, 'images', 'train')
    val_image_dir = os.path.join(extract_path, 'images', 'val')
    train_label_dir = os.path.join(extract_path, 'labels', 'train')
    val_label_dir = os.path.join(extract_path, 'labels', 'val')
    
    for folder in [train_image_dir, val_image_dir, train_label_dir, val_label_dir]:
        os.makedirs(folder, exist_ok=True)
    
    # Function to move files to the respective directories
    def move_files(image_list, image_dest, label_dest):
        for image in image_list:
            label = (os.path.splitext(image)[0] + '.txt').replace('images', 'labels')
            if os.path.exists(label):
                shutil.move(label, os.path.join(label_dest, os.path.basename(label)))
            shutil.move(image, os.path.join(image_dest, os.path.basename(image)))
    
    # Move files to the respective directories
    move_files(train_images, train_image_dir, train_label_dir)
    move_files(val_images, val_image_dir, val_label_dir)
    
    # Remove temporary files
    shutil.rmtree(temp_extract_path)
    os.remove(dataset_path)
    print(f"[*] Unzipped dataset to {extract_path}")

    train_path = os.path.join('images', 'train')
    val_path = os.path.join('images', 'val')

    # Create YAML configuration
    data_yaml_path = create_yaml_config(dataset_name, num_classes, class_names, train_path, val_path)
    
    try:
        # Prepare training options
        opt = parse_opt()
        opt.data = data_yaml_path
        opt.weights = 'yolov5/yolov5n.pt'  # default to small model, can be parameterized
        opt.cfg = ''  # use default model configuration
        opt.epochs = epochs
        opt.batch_size = batch_size
        opt.project = 'yolov5/runs/train'
                
        # Start training
        start_time = time.time()
        metrics, save_dir = main(opt)
        
        # Prepare response
        training_time = time.time() - start_time
        best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
        
        return jsonify({
            "status": "success",
            "message": f"Detection training completed successfully after {training_time:.2f} seconds.",
            "training_time": training_time,
            "results": metrics,
            "best_model_path": best_model_path
        }), 200
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 200

@app.route("/api/train/classification", methods=['POST'])
@cross_origin()
def train_classification():
    try:
        # Start time for training
        start_time = time.time()

        # Get training parameters from form data
        data = request.form.to_dict()
        model_name = data.get('modelName').replace(' ', '_').lower()
        pretrained_model = data.get('pretrainedModel', './weights/efficientnet_b7_last.pth')
        
        # Get dataset file from request
        dataset = request.files['dataset']
        dataset_name = data.get('datasetName', 'classify_dataset')
        dataset_path = os.path.join(DATASET_FOLDER, dataset.filename)
        dataset.save(dataset_path)
        print("Received dataset ZIP file:", dataset.filename)

        extract_path = os.path.join(DATASET_FOLDER)
        os.makedirs(extract_path, exist_ok=True)
        
        # Unzip the dataset
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        os.remove(dataset_path)
        print(f"[*] Unzipped dataset to folder {extract_path}")
                
        # Training parameters
        epochs = int(data.get('epochs', 10))
        batch_size = int(data.get('batchSize', 16))
        optimizer_name = data.get('optimizer', 'AdamW')
        
        # Load dataset
        list_path = os.path.join(DATASET_FOLDER, dataset_name)
        train_list = get_image_paths(list_path, phase='train')
        val_list = get_image_paths(list_path, phase='val')

        # Take all labels from the dataset
        all_labels = set()
        for path in train_list + val_list:
            label = os.path.basename(os.path.dirname(path))
            all_labels.add(label)

        # Find new labels
        new_labels = {label for label in all_labels if not label.isdigit() and label not in existing_labels}
        if new_labels:
            max_index = max(int(k) for k in class_index.keys())
            for i, label in enumerate(sorted(new_labels), start=max_index + 1):
                class_index[str(i)] = label
                label_to_index[label] = str(i)
            print(f'[+] Added new labels to mapping: {new_labels}')

            # Save updated mapping to JSON file
            with open(f'{HOME}/classify/mapping.json', mode='w', encoding='utf-8') as file:
                json.dump(class_index, file, ensure_ascii=False, indent=2)

        # Rename labels in the dataset
        def rename_label_folders_to_index(subset: Literal['train', 'val', 'test']):
            subset_path = os.path.join(DATASET_FOLDER, dataset_name, subset)

            for label in os.listdir(subset_path):
                label_folder = os.path.join(subset_path, label)
                if not os.path.isdir(label_folder):
                    continue

                class_idx = label_to_index.get(label)
                if class_idx is None:
                    continue

                new_folder = os.path.join(subset_path, class_idx)

                if label_folder != new_folder:
                    if os.path.exists(new_folder):
                        print(f"[!] Folder {new_folder} already exists, skipping {label}")
                        continue
                    os.rename(label_folder, new_folder)
                    print(f"[✓] Renamed {label} -> {class_idx}")
        
        rename_label_folders_to_index('train')
        rename_label_folders_to_index('val')

        # Reload the dataset paths after renaming
        train_list = get_image_paths(list_path, phase='train')
        val_list = get_image_paths(list_path, phase='val')
        
        # Create datasets
        train_dataset = SinoNomDataset(train_list, transform=transform, phase='train')
        val_dataset = SinoNomDataset(val_list, transform=transform, phase='val')
        
        # Create data loaders (you might want to modify this based on your existing code)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Select model
        num_classes = len(class_index)
        print(f"[*] Total number of classes: {num_classes}")
        model = get_model(num_classes)

        # if pretrained_model:
        #     print(f"[*] Loading pretrained weights from {pretrained_model}")
        #     model = load_model(model, pretrained_model)
        
        # Select optimizer
        if optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        elif optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        else:
            return jsonify({
                "status": "error",
                "message": f"Unsupported optimizer: {optimizer_name}"
            }), 200
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop (simplified, you'd want to add more comprehensive training logic)
        model.to(device)
        best_val_acc = 0.0
        train_histories = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            model.train()
            train_loss = 0.0
            train_acc = 0.0
            
            for inputs, labels in tqdm(train_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)

                # Calculate the loss
                loss = criterion(outputs, labels)
                train_loss += loss.item() * inputs.size(0)

                # Calculate the accuracy
                _, preds = torch.max(outputs.data, 1)
                train_acc += (preds == labels).sum().item()

                # Backpropagation
                loss.backward()

                # Update the weights
                optimizer.step()

            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader.dataset)
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')

            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                with torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs.data, 1)
                    val_acc += (preds == labels).sum().item()

            val_loss /= len(val_dataloader)
            val_acc /= len(val_dataloader.dataset)
            print(f'Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
            
            # Store training history
            train_histories['train_loss'].append(train_loss)
            train_histories['train_acc'].append(train_acc)
            train_histories['val_loss'].append(val_loss)
            train_histories['val_acc'].append(val_acc)

            # Save model weights
            save_model(model, f'{HOME}/weights/{model_name}/efficientnet_b7_last.pth')
            
            if best_val_acc <= val_acc:
                best_val_acc = val_acc
                save_model(model, f'{HOME}/weights/{model_name}/efficientnet_b7_best.pth')

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return jsonify({
            "status": "success",
            "message": f"Classification training completed successfully after {training_time:.2f} seconds.",
            "training_time": training_time,
            "best_validation_accuracy": best_val_acc,
            "training_histories": train_histories,
            "model_weights": f'{HOME}/models/{model_name}/efficientnet_b7_best.pth'
        }), 200
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": str(sys.exc_info())
        }), 200

ngrok.set_auth_token(os.environ.get('NGROK_AUTH_TOKEN'))
url = ngrok.connect(os.environ.get('PORT', 5000)).public_url
db.update({"server_url": url})
print('Global NGROK URL:', url)

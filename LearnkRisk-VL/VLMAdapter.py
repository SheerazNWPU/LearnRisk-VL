from __future__ import print_function
import os
from glob import glob
from os.path import join
import pickle
import numpy as np
import time
import sys
import math
import random
import seaborn as sns
import matplotlib.pyplot as plt
from  Folder import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
# from torch.nn.functional import softmax
from tqdm import tqdm
import torch, random
# import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
# import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
# import torch.nn.functional as func
from torch.utils.data import DataLoader, Dataset
# from torchvision.datasets import ImageFolder
import json
# HF transformers (needed for text encoder)
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from PIL import Image

from Densenet import densenet121
from Densenet import densenet169
from Densenet import densenet201
from Resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, wide_resnet50_2, wide_resnet101_2
from Efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from PIL import Image

import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
# import calibration as cb
from utils import *

from risk_one_rule import risk_dataset
from risk_one_rule import risk_torch_model
import risk_one_rule.risk_torch_model as risk_model
from common import config as config_risk

from scipy.special import softmax

import csv

cfg = config_risk.Configuration(config_risk.global_data_selection, config_risk.global_deep_learning_selection)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

"""Seed and GPU setting"""
seed = (int)(sys.argv[1])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.cuda.manual_seed(seed)

cudnn.benchmark = True
cudnn.deterministic = True

## Allow Large Images
Image.MAX_IMAGE_PIXELS = None

# -------------------------
# CIFAR-100 Dataset Class (from training code)
# ==================== DATASET CONFIGURATION ====================
def get_dataset_config(dataset_name):
    """Get dataset-specific configuration (from training code)"""
    config = {
        'normalize_mean': (0.5071, 0.4867, 0.4408),
        'normalize_std': (0.2675, 0.2565, 0.2761),
        'image_size': 32,
        'class_names': None,
        'is_office': False,
        'is_corrupted': False,
        'office_domain': None,
        'dataset_type': 'standard'
    }
    
    if dataset_name == 'CIFAR100':
        config['normalize_mean'] = (0.5071, 0.4867, 0.4408)
        config['normalize_std'] = (0.2675, 0.2565, 0.2761)
        config['image_size'] = 32
        config['class_names'] = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]
        
    elif dataset_name == 'CIFAR10':
        config['normalize_mean'] = (0.4914, 0.4822, 0.4465)
        config['normalize_std'] = (0.2470, 0.2435, 0.2616)
        config['image_size'] = 32
        config['class_names'] = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
    elif dataset_name == 'STL10':
        config['normalize_mean'] = (0.4914, 0.4822, 0.4465)
        config['normalize_std'] = (0.2470, 0.2435, 0.2616)
        config['image_size'] = 96
        config['class_names'] = [
            'airplane', 'bird', 'car', 'cat', 'deer',
            'dog', 'horse', 'monkey', 'ship', 'truck'
        ]
        
    elif dataset_name == 'TinyImageNet':
        config['normalize_mean'] = (0.4802, 0.4481, 0.3975)
        config['normalize_std'] = (0.2770, 0.2691, 0.2821)
        config['image_size'] = 64
        # Will load actual class names from dataset
        
    elif 'Office31_' in dataset_name:
        config['normalize_mean'] = (0.485, 0.456, 0.406)
        config['normalize_std'] = (0.229, 0.224, 0.225)
        config['image_size'] = 224
        config['is_office'] = True
        config['office_domain'] = dataset_name.split('_')[1]  # Amazon, Webcam, or DSLR
        config['class_names'] = [
            'backpack', 'bike', 'bike helmet', 'bookcase', 'bottle',
            'calculator', 'desk chair', 'desk lamp', 'desktop computer', 'file cabinet',
            'headphones', 'keyboard', 'laptop computer', 'letter tray', 'mobile phone',
            'monitor', 'mouse', 'mug', 'paper notebook', 'pen', 'phone', 'printer',
            'projector', 'punchers', 'ring binder', 'ruler', 'scissors', 'speaker',
            'stapler', 'tape dispenser', 'trash can'
        ]
        
    elif 'CIFAR100-C' in dataset_name:
        config['normalize_mean'] = (0.5071, 0.4867, 0.4408)
        config['normalize_std'] = (0.2675, 0.2565, 0.2761)
        config['image_size'] = 32
        config['is_corrupted'] = True
        config['class_names'] = get_dataset_config('CIFAR100')['class_names']
        
    elif 'CIFAR10-C' in dataset_name:
        config['normalize_mean'] = (0.4914, 0.4822, 0.4465)
        config['normalize_std'] = (0.2470, 0.2435, 0.2616)
        config['image_size'] = 32
        config['is_corrupted'] = True
        config['class_names'] = get_dataset_config('CIFAR10')['class_names']
        
    return config

# ==================== DATASET LOADERS (with domain adaptation support) ====================
# Dataset Loader Functions
def load_office31_dataset(domain, split='test', transform=None, val_split=0.1):
    """Load Office-31 dataset with class-balanced predefined splits"""
    # Use ALL 31 Office-31 classes
    office31_classes = [
        'back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle',
        'calculator', 'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet',
        'headphones', 'keyboard', 'laptop_computer', 'letter_tray', 'mobile_phone',
        'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer',
        'projector', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker',
        'stapler', 'tape_dispenser', 'trash_can'
    ]
    
    # Office-31 dataset statistics
    office31_stats = {
        'Amazon': {'train': 2000, 'val': 281, 'test': 536},
        'Webcam': {'train': 556, 'val': 80, 'test': 159},
        'DSLR': {'train': 348, 'val': 50, 'test': 100}
    }
    
    domain_lower = domain.lower()
    domain_key = domain  # 'Amazon', 'Webcam', or 'DSLR'
    data_path = f'/Datasets/Office-31/{domain_lower}/'
    
    if not os.path.exists(data_path):
        print(f"[WARNING] Office-31 {domain} not found at {data_path}")
        
        class DummyOfficeDataset(Dataset):
            def __init__(self, size=100):
                self.size = size
                self.classes = office31_classes
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                img = torch.randn(3, 224, 224)
                label = idx % len(self.classes)
                if transform:
                    img = transform(img)
                return img, label, f"{domain}_{idx:06d}.png"
        
        return DummyOfficeDataset(size=100), office31_classes
    
    from torchvision.datasets import ImageFolder
    
    # Load the full dataset
    print(f"[INFO] Loading Office-31 {domain} from {data_path}")
    full_dataset = ImageFolder(root=data_path, transform=None)
    print(f"[INFO] Found {len(full_dataset)} total images in {domain}")
    
    # Get actual classes
    actual_classes = full_dataset.classes
    print(f"[INFO] Found {len(actual_classes)} classes")
    
    # Verify all classes match
    if set(actual_classes) != set(office31_classes):
        print(f"[WARNING] Dataset classes don't match expected Office-31 classes")
        print(f"  Missing: {set(office31_classes) - set(actual_classes)}")
        print(f"  Extra: {set(actual_classes) - set(office31_classes)}")
    
    # Create class-to-indices mapping for balanced splitting
    class_indices = {i: [] for i in range(len(actual_classes))}
    
    # Group images by class
    for idx, (_, label) in enumerate(full_dataset):
        class_indices[label].append(idx)
    
    # Get predefined split sizes
    if domain_key in office31_stats:
        stats = office31_stats[domain_key]
        print(f"[INFO] Using predefined splits for {domain}: {stats}")
        
        # Calculate per-class split sizes
        total_train = stats['train']
        total_val = stats['val']
        total_test = stats['test']
        
        # Calculate approximate per-class sizes
        num_classes = len(actual_classes)
        train_per_class = total_train // num_classes
        val_per_class = total_val // num_classes
        test_per_class = total_test // num_classes
        
        print(f"[INFO] Approximate per-class: {train_per_class} train, {val_per_class} val, {test_per_class} test")
        
        # Collect indices for each split
        train_indices = []
        val_indices = []
        test_indices = []
        
        for class_idx in range(num_classes):
            indices = class_indices[class_idx]
            random.Random(42).shuffle(indices)  # Shuffle within each class
            
            # Take splits
            class_train = indices[:train_per_class]
            class_val = indices[train_per_class:train_per_class + val_per_class]
            class_test = indices[train_per_class + val_per_class:train_per_class + val_per_class + test_per_class]
            
            train_indices.extend(class_train)
            val_indices.extend(class_val)
            test_indices.extend(class_test)
        
        # If we have leftover images after even distribution, add them to train
        remaining = []
        for class_idx in range(num_classes):
            indices = class_indices[class_idx]
            used = set(train_indices + val_indices + test_indices)
            for idx in indices:
                if idx not in used:
                    remaining.append(idx)
        
        # Add remaining to train
        train_indices.extend(remaining)
        
        # Select indices based on split
        if split == 'train':
            subset_indices = train_indices
        elif split == 'val':
            subset_indices = val_indices
        elif split == 'test':
            subset_indices = test_indices
        else:  # 'all'
            subset_indices = list(range(len(full_dataset)))
        
        #print(f"[INFO] Created {split} split with {len(subset_indices)} images")
        #print(f"  Class distribution in {split}:")
        #for class_idx in range(num_classes):
            #class_count = sum(1 for idx in subset_indices if full_dataset[idx][1] == class_idx)
            #if class_count > 0:
            #    print(f"    {actual_classes[class_idx]}: {class_count}")
        
    else:
        print(f"[WARNING] No predefined splits for {domain}")
        num_samples = len(full_dataset)
        subset_indices = list(range(num_samples))
    
    # Create simple mapping
    class_mapping = {cls: cls for cls in actual_classes}
    
    # Create mapped dataset
    class MappedOfficeDataset(Dataset):
        def __init__(self, imagefolder_dataset, class_mapping, office31_classes, indices, transform=None):
            self.dataset = imagefolder_dataset
            self.class_mapping = class_mapping
            self.office31_classes = office31_classes
            self.indices = indices
            self.transform = transform
            
        def __len__(self):
            return len(self.indices)
            
        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            img, label = self.dataset[real_idx]
            actual_class = self.dataset.classes[label]
            
            # Map to index in our class list
            if actual_class in self.office31_classes:
                mapped_label = self.office31_classes.index(actual_class)
            else:
                mapped_label = 0
            
            if self.transform:
                img = self.transform(img)
                
            return img, mapped_label, f"{domain}_{split}_{real_idx:06d}.png"
    
    dataset = MappedOfficeDataset(full_dataset, class_mapping, office31_classes, subset_indices, transform)
    
    return dataset, office31_classes

def load_torchvision_dataset(dataset_name, split='test', transform=None, val_split=0.1):
    """Load standard torchvision datasets"""
    if split == 'test':
        if dataset_name == 'CIFAR100':
            dataset = datasets.CIFAR100(
                root='./data',
                train=False,
                download=True,
                transform=transform
            )
            class_names = dataset.classes
        elif dataset_name == 'CIFAR10':
            dataset = datasets.CIFAR10(
                root='./data',
                train=False,
                download=True,
                transform=transform
            )
            class_names = dataset.classes
        elif dataset_name == 'STL10':
            dataset = datasets.STL10(
                root='./data',
                split='test',
                download=True,
                transform=transform
            )
            class_names = dataset.classes
        elif dataset_name == 'TinyImageNet':
            from torchvision.datasets import ImageFolder
            dataset = ImageFolder(
                root='./data/tiny-imagenet-200/val',
                transform=transform
            )
            class_names = [name for name in os.listdir('./data/tiny-imagenet-200/train') 
                          if os.path.isdir(os.path.join('./data/tiny-imagenet-200/train', name))]
            class_names.sort()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    else:  # train or val
        if dataset_name == 'CIFAR100':
            full_dataset = datasets.CIFAR100(
                root='./data',
                train=True,
                download=True,
                transform=None
            )
            class_names = full_dataset.classes
        elif dataset_name == 'CIFAR10':
            full_dataset = datasets.CIFAR10(
                root='./data',
                train=True,
                download=True,
                transform=None
            )
            class_names = full_dataset.classes
        elif dataset_name == 'STL10':
            full_dataset = datasets.STL10(
                root='./data',
                split='train',
                download=True,
                transform=None
            )
            class_names = full_dataset.classes
        elif dataset_name == 'TinyImageNet':
            from torchvision.datasets import ImageFolder
            full_dataset = ImageFolder(
                root='./data/tiny-imagenet-200/train',
                transform=None
            )
            class_names = [name for name in os.listdir('./data/tiny-imagenet-200/train') 
                          if os.path.isdir(os.path.join('./data/tiny-imagenet-200/train', name))]
            class_names.sort()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Split into train and val
        num_samples = len(full_dataset)
        indices = list(range(num_samples))
        random.Random(42).shuffle(indices)
        
        split_idx = int(num_samples * (1 - val_split))
        
        if split == 'train':
            subset_indices = indices[:split_idx]
        else:  # val
            subset_indices = indices[split_idx:]
        
        # Apply transforms to subset - FIXED: return (img, label, path)
        class TransformedSubset(Dataset):
            def __init__(self, subset, indices, transform):
                self.subset = subset
                self.indices = indices
                self.transform = transform
                
            def __len__(self):
                return len(self.indices)
                
            def __getitem__(self, idx):
                real_idx = self.indices[idx]
                img, label = self.subset[real_idx]
                if self.transform:
                    img = self.transform(img)
                # Return path as third element
                return img, label, f"{split}_{real_idx:06d}.png"
        
        dataset = TransformedSubset(full_dataset, subset_indices, transform)
    
    # For test dataset, also wrap it to return path
    if split == 'test':
        class WrappedDataset(Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
                
            def __len__(self):
                return len(self.dataset)
                
            def __getitem__(self, idx):
                img, label = self.dataset[idx]
                return img, label, f"{split}_{idx:06d}.png"
        
        dataset = WrappedDataset(dataset)
    
    return dataset, class_names

def load_dataset(dataset_name, split='test', transform=None, val_split=0.1):
    """Universal dataset loader with domain adaptation support"""
    print(f"[INFO] Loading {dataset_name} {split} data")
    
    if 'Office31_' in dataset_name:
        domain = dataset_name.split('_')[1]
        return load_office31_dataset(domain, split, transform, val_split)
    
    elif 'CIFAR' in dataset_name and '-C' in dataset_name:
        return load_cifar_corrupted(dataset_name, split, transform, val_split)
    
    else:
        return load_torchvision_dataset(dataset_name, split, transform, val_split)
# -------------------------
# CIFAR-100 class names from your training code
class_names = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]
def unpickle(file):
    """Unpickle CIFAR-100 file"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class OriginalCIFAR100Dataset(Dataset):
    """Load CIFAR-100 from original pickle files"""
    def __init__(self, data_dir, split='train', transform=None, val_split=0.1):
        """
        Args:
            data_dir: Directory containing CIFAR-100 pickle files
            split: 'train', 'val', or 'test'
            transform: Optional transform to be applied
            val_split: Fraction of training data to use as validation
        """
        self.split = split
        self.transform = transform
        self.data_dir = data_dir
        self.class_names = class_names
        
        print(f"[INFO] Loading CIFAR-100 {split} data from {data_dir}")
        
        # Load data based on split
        if split == 'train' or split == 'val':
            # Load training data
            train_file = os.path.join(data_dir, 'train')
            if not os.path.exists(train_file):
                train_file = os.path.join(data_dir, 'train.bin')
            if not os.path.exists(train_file):
                train_file = os.path.join(data_dir, 'data_batch_1')
                
            if os.path.exists(train_file):
                print(f"[INFO] Loading training data from {train_file}")
                train_data = unpickle(train_file)
                
                # CIFAR-100 format
                if b'data' in train_data:
                    images = train_data[b'data']
                    fine_labels = train_data[b'fine_labels'] if b'fine_labels' in train_data else train_data.get(b'labels', [])
                else:
                    images = train_data.get('data', train_data.get('train_data', None))
                    fine_labels = train_data.get('fine_labels', train_data.get('labels', train_data.get('train_labels', [])))
                
                if images is None:
                    raise ValueError("Could not find image data in training file")
                
                # Convert to numpy arrays
                if isinstance(images, list):
                    images = np.array(images)
                if isinstance(fine_labels, list):
                    fine_labels = np.array(fine_labels)
                
                # Reshape images: CIFAR-100 images are 32x32 RGB (3, 32, 32)
                images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                
                print(f"[INFO] Loaded {len(images)} training images")
                
                # Split into train and val
                num_train = len(images)
                indices = list(range(num_train))
                random.Random(42).shuffle(indices)
                
                split_idx = int(num_train * (1 - val_split))
                
                if split == 'train':
                    train_indices = indices[:split_idx]
                    self.images = images[train_indices]
                    self.labels = fine_labels[train_indices] if len(fine_labels) > 0 else []
                    print(f"[INFO] Using {len(self.images)} images for training")
                else:  # val
                    val_indices = indices[split_idx:]
                    self.images = images[val_indices]
                    self.labels = fine_labels[val_indices] if len(fine_labels) > 0 else []
                    print(f"[INFO] Using {len(self.images)} images for validation")
            else:
                raise FileNotFoundError(f"Training file not found: {train_file}")
        
        elif split == 'test':
            # Load test data
            test_file = os.path.join(data_dir, 'test')
            if not os.path.exists(test_file):
                test_file = os.path.join(data_dir, 'test.bin')
            if not os.path.exists(test_file):
                test_file = os.path.join(data_dir, 'test_batch')
            
            if os.path.exists(test_file):
                print(f"[INFO] Loading test data from {test_file}")
                test_data = unpickle(test_file)
                
                # CIFAR-100 format
                if b'data' in test_data:
                    images = test_data[b'data']
                    fine_labels = test_data[b'fine_labels'] if b'fine_labels' in test_data else test_data.get(b'labels', [])
                else:
                    images = test_data.get('data', test_data.get('test_data', None))
                    fine_labels = test_data.get('fine_labels', test_data.get('labels', test_data.get('test_labels', [])))
                
                if images is None:
                    raise ValueError("Could not find image data in test file")
                
                # Convert to numpy arrays
                if isinstance(images, list):
                    images = np.array(images)
                if isinstance(fine_labels, list):
                    fine_labels = np.array(fine_labels)
                
                # Reshape images
                images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                
                self.images = images
                self.labels = fine_labels if len(fine_labels) > 0 else []
                print(f"[INFO] Loaded {len(self.images)} test images")
            else:
                raise FileNotFoundError(f"Test file not found: {test_file}")
        else:
            raise ValueError(f"Invalid split: {split}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx] if len(self.labels) > 0 else 0
        
        # Convert numpy array to PIL Image
        img = Image.fromarray(img.astype('uint8'))
        
        # Apply transform
        if self.transform:
            img = self.transform(img)
        
        # Create dummy path
        path = f"{self.split}_{idx:06d}.png"
        
        return img, label, path

# -------------------------
# Model classes (must match training code EXACTLY)
# -------------------------
class ImageEncoder(nn.Module):
    def __init__(self, backbone, cnn_name, proj_dim=512):
        super().__init__()
        self.backbone = backbone
        self.cnn_name = cnn_name
        self._in_features = None

        if cnn_name.startswith("r") or cnn_name.startswith("w"):
            if hasattr(self.backbone, "fc"):
                self._in_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
        elif cnn_name.startswith("v"):
            try:
                self._in_features = self.backbone.classifier[0].in_features
                self.backbone.classifier = nn.Identity()
            except Exception:
                self._in_features = None
        elif cnn_name.startswith("d"):
            try:
                self._in_features = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Identity()
            except Exception:
                self._in_features = None
        elif cnn_name.startswith("e"):
            try:
                last_linear = [m for m in self.backbone.classifier.modules() if isinstance(m, nn.Linear)][-1]
                self._in_features = last_linear.in_features
                self.backbone.classifier = nn.Identity()
            except Exception:
                self._in_features = None
        else:
            for _, m in reversed(list(self.backbone.named_modules())):
                if isinstance(m, nn.Linear):
                    self._in_features = m.out_features
                    break

        if self._in_features is None:
            self.backbone.eval()
            with torch.no_grad():
                x = torch.zeros(1, 3, 32, 32)
                out = self.backbone(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if out.dim() == 4:
                    out = F.adaptive_avg_pool2d(out, 1).reshape(1, -1)
                self._in_features = out.shape[1]
            self.backbone.train()

        self.proj = nn.Linear(self._in_features, proj_dim, bias=False)
        self.ln = nn.LayerNorm(proj_dim)

    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).reshape(feats.size(0), -1)
        out = self.proj(feats)
        out = self.ln(out)
        return F.normalize(out, dim=-1)

class ImprovedDeepSeekTextEncoder(nn.Module):
    def __init__(self, proj_dim=512, local_model_path=None):
        super().__init__()
        
        if local_model_path is None:
            raise RuntimeError("Please provide --local_model_path for DeepSeek")

        print(f"[INFO] Loading DeepSeek from: {local_model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            local_files_only=True,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config = AutoConfig.from_pretrained(local_model_path, local_files_only=True)
        self.hidden_size = getattr(config, "hidden_size", 4096)
        
        vocab_size = getattr(config, "vocab_size", 32000)
        self.embedding_layer = nn.Embedding(vocab_size, self.hidden_size)
        
        # Try to load pre-trained embeddings
        try:
            model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                local_files_only=True,
                trust_remote_code=True,
                output_hidden_states=True
            )
            for name, module in model.named_modules():
                if isinstance(module, nn.Embedding) and "embed" in name.lower():
                    if module.weight.shape == self.embedding_layer.weight.shape:
                        self.embedding_layer.weight.data.copy_(module.weight.data)
                        print(f"[INFO] Loaded embeddings from {name}")
                        break
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[INFO] Could not load embeddings: {e}")
            nn.init.normal_(self.embedding_layer.weight, mean=0.0, std=0.02)
        
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            batch_first=True
        )
        
        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size)
        
        self.proj = nn.Linear(self.hidden_size, proj_dim, bias=False)
        self.ln = nn.LayerNorm(proj_dim)
        
        print(f"[INFO] Improved DeepSeek encoder initialized")

    def forward(self, input_ids=None, attention_mask=None, texts=None):
        if texts is not None:
            tokens = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
        
        input_ids = input_ids.to(self.proj.weight.device)
        attention_mask = attention_mask.to(self.proj.weight.device)

        token_embeddings = self.embedding_layer(input_ids)
        
        batch_size, seq_len, hidden_dim = token_embeddings.shape
        query = token_embeddings.mean(dim=1, keepdim=True)
        
        attended_embeddings, attention_weights = self.attention_pool(
            query=query,
            key=token_embeddings,
            value=token_embeddings,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        
        embeddings = attended_embeddings.squeeze(1)
        embeddings = self.layer_norm1(embeddings)
        embeddings = self.ln(self.proj(embeddings))
        return F.normalize(embeddings, dim=-1)

# -------------------------
# Cross-Attention Fusion Module (NEW - from medical code)
# -------------------------
class CrossAttentionFusion(nn.Module):
    """
    Cross-attention based feature fusion between image and text embeddings
    """
    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Cross-attention layers
        self.img_to_txt_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.txt_to_img_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        # Feed-forward networks
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        self.output_proj = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, img_emb, txt_emb):
        """
        Args:
            img_emb: (batch_size, dim) - image embeddings
            txt_emb: (batch_size, num_classes, dim) - text embeddings for all classes
        Returns:
            fused_emb: (batch_size, dim) - fused embeddings
        """
        batch_size = img_emb.size(0)
        num_classes = txt_emb.size(0) if txt_emb.dim() == 2 else txt_emb.size(1)
        
        # Ensure proper shapes
        if txt_emb.dim() == 2:  # (num_classes, dim)
            txt_emb = txt_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_classes, dim)
        
        img_emb = img_emb.unsqueeze(1)  # (batch_size, 1, dim)
        
        # Image-to-text cross attention
        img_enhanced, _ = self.img_to_txt_attention(
            query=img_emb,  # (batch_size, 1, dim)
            key=txt_emb,    # (batch_size, num_classes, dim)
            value=txt_emb,  # (batch_size, num_classes, dim)
        )
        img_enhanced = self.norm1(img_emb + self.dropout(img_enhanced))
        
        # Text-to-image cross attention (using mean text embedding as query)
        mean_txt_emb = txt_emb.mean(dim=1, keepdim=True)  # (batch_size, 1, dim)
        txt_enhanced, _ = self.txt_to_img_attention(
            query=mean_txt_emb,  # (batch_size, 1, dim)
            key=img_emb,         # (batch_size, 1, dim)
            value=img_emb,       # (batch_size, 1, dim)
        )
        txt_enhanced = self.norm2(mean_txt_emb + self.dropout(txt_enhanced))
        
        # Concatenate and fuse
        combined = torch.cat([img_enhanced.squeeze(1), txt_enhanced.squeeze(1)], dim=-1)  # (batch_size, dim*2)
        fused_emb = self.output_proj(combined)
        fused_emb = self.norm3(fused_emb)
        
        return F.normalize(fused_emb, dim=-1)

class CLIPModel(nn.Module):
    def __init__(self, image_encoder: ImageEncoder, text_encoder: nn.Module, init_logit_scale=0.07):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        # Add cross-attention fusion module
        self.cross_attention_fusion = CrossAttentionFusion(dim=512, num_heads=8, dropout=0.1)
        # CRITICAL: This must match exactly with training code
        self.logit_scale = nn.Parameter(torch.tensor([math.log(1/init_logit_scale)]), requires_grad=True)

    def forward(self, images, input_ids, attention_mask):
        img_emb = self.image_encoder(images)
        txt_emb = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        if img_emb.dim() != 2:
            img_emb = img_emb.squeeze()
        if txt_emb.dim() != 2:
            txt_emb = txt_emb.squeeze()
            
        logit_scale = self.logit_scale.exp().clamp(max=100.0, min=0.01)
        logits_per_image = logit_scale * (img_emb @ txt_emb.t())
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text, logit_scale
# Dataset wrapper for CLIP
class CLIPWrappedDataset(Dataset):
    def __init__(self, dataset, prompts, tokenizer, max_len=32):
        super().__init__()
        self.dataset = dataset
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label, path = self.dataset[idx]
        prompt = self.prompts[label]
        enc = self.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].squeeze(0)
        attn_mask = enc['attention_mask'].squeeze(0)
        return img, input_ids, attn_mask, label, path

# Improved Loss with debugging
class ClipLoss(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.debug = debug

    def forward(self, logits_per_image, logits_per_text, temperature=None):
        device = logits_per_image.device
        batch_size = logits_per_image.size(0)
        targets = torch.arange(batch_size, device=device)
        
        # Debug information
        if self.debug:
            print(f"[DEBUG] Logit range: [{logits_per_image.min():.2f}, {logits_per_image.max():.2f}]")
            print(f"[DEBUG] Logit mean: {logits_per_image.mean():.2f}, std: {logits_per_image.std():.2f}")
            if temperature is not None:
                print(f"[DEBUG] Temperature: {temperature.item():.4f}")
            
            # Check for extreme values
            if torch.isnan(logits_per_image).any():
                print("[WARNING] NaN in logits_per_image!")
            if torch.isinf(logits_per_image).any():
                print("[WARNING] Inf in logits_per_image!")
        
        loss_i = self.ce(logits_per_image, targets)
        loss_t = self.ce(logits_per_text, targets)
        
        return (loss_i + loss_t) / 2.0

@torch.no_grad()
def retrieval_and_classify(img_embs, txt_embs, targets_txt=None, ks=(1,5,10)):
    device = img_embs.device
    sims = img_embs @ txt_embs.t()
    ranks = sims.argsort(dim=1, descending=True)
    targets_img = torch.arange(img_embs.size(0), device=device)

    itopk = {}
    if targets_txt is None:
        N_txt = txt_embs.size(0)
        N_img = img_embs.size(0)
        if N_img == 0:
            return {}, {}, {"acc":0.0, "f1":0.0, "prec":0.0, "rec":0.0}
        txt_per_img = max(1, N_txt // N_img)
        targets_txt = torch.arange(N_txt, device=device) // txt_per_img

    for k in ks:
        topk_txt = ranks[:, :k]
        hits = (topk_txt == targets_img.unsqueeze(1)).any(dim=1).float().mean().item()
        itopk[f"R@{k}"] = hits

    sims_t = sims.t()
    ranks_t = sims_t.argsort(dim=1, descending=True)
    ttopk = {}
    for k in ks:
        topk_img = ranks_t[:, :k]
        hits = (topk_img == targets_txt.unsqueeze(1)).any(dim=1).float().mean().item()
        ttopk[f"R@{k}"] = hits

    preds = sims.argmax(dim=1).cpu().numpy()
    labels = targets_img.cpu().numpy()
    acc = 100.0 * accuracy_score(labels, preds)
    f1 = 100.0 * f1_score(labels, preds, average='weighted', zero_division=0)
    prec = 100.0 * precision_score(labels, preds, average='weighted', zero_division=0)
    rec = 100.0 * recall_score(labels, preds, average='weighted', zero_division=0)

    return itopk, ttopk, {"acc": acc, "f1": f1, "prec": prec, "rec": rec}

# Gradient debugging function
def check_gradients(model, epoch, debug=False):
    """Check gradient statistics"""
    if not debug:
        return
    
    print(f"\n=== Gradient Analysis - Epoch {epoch} ===")
    total_norm = 0
    zero_grad_count = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            total_params += 1
            
            if param_norm == 0:
                zero_grad_count += 1
                if 'text_encoder' in name or 'logit_scale' in name:
                    print(f"  ZERO GRAD: {name}")
            
            if 'logit_scale' in name:
                print(f"  Temperature - grad: {param_norm:.6f}, value: {param.item():.4f}")
    
    if total_params > 0:
        total_norm = total_norm ** 0.5
        print(f"  Total gradient norm: {total_norm:.4f}")
        print(f"  Parameters with zero gradient: {zero_grad_count}/{total_params} ({zero_grad_count/total_params*100:.1f}%)")

# Debug embeddings function
def debug_embeddings(model, train_loader, prompts, tokenizer, max_seq_len, device, epoch):
    """Debug embedding statistics"""
    model.eval()
    
    with torch.no_grad():
        # Get some training data
        images, input_ids, attn_mask, targets, paths = next(iter(train_loader))
        images = images.to(device)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        
        # Get embeddings
        img_emb = model.image_encoder(images)
        txt_emb = model.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
        
        print(f"\n=== Embedding Debug - Epoch {epoch} ===")
        print(f"Image embeddings norm: {img_emb.norm(dim=1).mean():.4f}  {img_emb.norm(dim=1).std():.4f}")
        print(f"Text embeddings norm: {txt_emb.norm(dim=1).mean():.4f}  {txt_emb.norm(dim=1).std():.4f}")
        
        # Check similarity
        sims = img_emb @ txt_emb.t()
        print(f"Similarity diagonal (correct pairs): {sims.diag().mean():.4f}")
        print(f"Similarity off-diagonal: {sims[~torch.eye(sims.size(0), dtype=bool, device=device)].mean():.4f}")
        
        # Get text embeddings for all classes
        class_enc = tokenizer(prompts, padding='max_length', truncation=True, 
                             max_length=max_seq_len, return_tensors='pt')
        class_input_ids = class_enc['input_ids'].to(device)
        class_attns = class_enc['attention_mask'].to(device)
        class_txt_embs = model.text_encoder(input_ids=class_input_ids, attention_mask=class_attns)
        
        # Check class embedding norms
        print(f"Class embeddings norm: {class_txt_embs.norm(dim=1).mean():.4f}  {class_txt_embs.norm(dim=1).std():.4f}")
        
        # Temperature
        temp = model.logit_scale.exp().item()
        print(f"Temperature: {temp:.4f}")
    
    model.train()

# Custom scheduler with warmup
class WarmupCosineSchedule:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch):
        if epoch <= self.warmup_epochs:
            # Linear warmup
            lr_mult = epoch / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr_mult = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_mult
        
        return self.optimizer.param_groups[0]['lr']


class CustomModule(nn.Module):
    def __init__(self, base_model, class_num):
        super(CustomModule, self).__init__()
        self.base_model = base_model
        self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, class_num)  # Correcting the layer name for DenseNet
        self.alpha = nn.Parameter(torch.tensor(2.0))
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.alpha * x
        return x

def softmax_with_temperature(logits, temperature=1.0):
    """
    Computes softmax with temperature.
    Supports both PyTorch tensors and NumPy arrays.
    """

    # If PyTorch tensor, detach and convert to NumPy
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()

    return softmax(logits / temperature, axis=1)
        
        
          
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def output_risk_scores(file_path, id_2_scores, label_index, ground_truth_y, predict_y):
    op_file = open(file_path, 'w+', 1, encoding='utf-8')
    #print(op_file)
    for i in range(len(id_2_scores)):
        #print("CHECK")
        _id = id_2_scores[i][0]
        _risk = id_2_scores[i][1]
        _label_index = label_index.get(_id)
        _str = "{}, {}, {}, {}".format(ground_truth_y[_label_index],
                                       predict_y[_label_index],
                                       _risk,
                                       _id)
        op_file.write(_str + '\n')
    op_file.flush()
    op_file.close()
    return True

def prepare_data_4_risk_data():
    """
    first, generate , include all_info.csv, train.csv, val.csv, test.csv.
    second, use csvs to generate rules. one rule just judge one class
    :return:
    """
    train_data, validation_data, test_data = risk_dataset.load_data(cfg)
    return train_data, validation_data, test_data

def prepare_data_4_risk_model(train_data, validation_data, test_data):

    rm = risk_torch_model.RiskTorchModel()
    rm.train_data = train_data
    rm.validation_data = validation_data
    rm.test_data = test_data
    return rm
    
def adjust_state_dict(state_dict, model):
    if isinstance(model, nn.DataParallel):
        new_state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith('module.')}
    else:
        new_state_dict = state_dict
    return new_state_dict
# --------------------------------------------------------------------------------


class VLMadapter():



    def train(pathImgTrain, pathImgVal, pathImgTest, nnArchitecture, nnIsTrained, class_num, batch_size, nb_epoch,
          transResize, transCrop, launchTimestamp, val_num, store_name, model_path, start_epoch=0, resume=False):
    
        save_name = os.path.join('/risk_val_pmg_result/', str(val_num), store_name.split('/')[-1],
                                 str(seed))
        print(save_name)
        if (not os.path.exists(save_name)):
            os.makedirs(save_name)
    
        exp_dir = save_name
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            os.stat(exp_dir)
        except:
            os.makedirs(exp_dir)
    
        use_cuda = torch.cuda.is_available()
        print(use_cuda)
        print(nnArchitecture)
        
        print(f'===== Adaptive Training with cnn backbone =====')
        
        # Model zoo
        model_zoo = {
            'r18': resnet18, 'r34': resnet34, 'r50': resnet50, 'r101': resnet101, 'r152': resnet152,
            'wrn50': wide_resnet50_2, 'wrn101': wide_resnet101_2,
            'd121': densenet121, 'd169': densenet169, 'd201': densenet201,
            'eb4': efficientnet_b4, 'rx50': resnext50_32x4d
        }
        cnn = nnArchitecture
        # Build model EXACTLY as in training
        backbone = model_zoo[cnn](pretrained=True)  # Use pretrained=True as in training
        proj_dim = 512
        image_encoder = ImageEncoder(backbone, cnn, proj_dim)
        
        text_encoder = ImprovedDeepSeekTextEncoder(
            proj_dim=proj_dim,
            local_model_path='/.cache/huggingface/hub/models--deepseek-ai--deepseek-llm-7b-base/snapshots/main/'
        )
        
        model = CLIPModel(image_encoder, text_encoder, init_logit_scale=0.07)
        
        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # -------------------------
        # Load checkpoint with proper shape handling
        # -------------------------
        model_file_name = '/result_archive/Office31_Amazon_deepseek/max_f1_87.99_epoch29.pth'
        
        # Load checkpoint
        checkpoint = torch.load(model_file_name, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model', checkpoint)
        
        print(f"\n=== Loading checkpoint ===")
        print(f"Checkpoint keys: {list(state_dict.keys())[:10]}...")
        
        # Check if cross_attention_fusion exists in checkpoint
        if 'cross_attention_fusion' not in state_dict:
            print("Warning: cross_attention_fusion not found in checkpoint. Will use random initialization.")
        
        # Check logit_scale shape
        if 'logit_scale' in state_dict:
            saved_logit_scale = state_dict['logit_scale']
            print(f"Saved logit_scale shape: {saved_logit_scale.shape}, value: {saved_logit_scale}")
        
        # Check current model's logit_scale
        current_logit_scale = model.state_dict()['logit_scale']
        print(f"Current model logit_scale shape: {current_logit_scale.shape}, value: {current_logit_scale}")
        
        # Fix shape if necessary
        if 'logit_scale' in state_dict and state_dict['logit_scale'].shape != current_logit_scale.shape:
            print(f"\nFixing logit_scale shape mismatch:")
            print(f"  Saved shape: {state_dict['logit_scale'].shape}")
            print(f"  Current shape: {current_logit_scale.shape}")
            
            if state_dict['logit_scale'].dim() == 1 and state_dict['logit_scale'].shape[0] == 1:
                # Convert [1] to scalar [] if current expects scalar
                if current_logit_scale.dim() == 0:
                    state_dict['logit_scale'] = state_dict['logit_scale'].squeeze()
                    print(f"  Fixed: [1] -> []")
            elif state_dict['logit_scale'].dim() == 0 and current_logit_scale.dim() == 1:
                # Convert scalar to [1] if current expects [1]
                if current_logit_scale.shape[0] == 1:
                    state_dict['logit_scale'] = state_dict['logit_scale'].unsqueeze(0)
                    print(f"  Fixed: [] -> [1]")
        
        # Now try loading
        print("\nAttempting to load state_dict...")
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded checkpoint!")
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")
            
            if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                print("All parameters loaded perfectly!")
            else:
                print("Some parameters didn't match, but continuing...")
                
        except Exception as e:
            print(f"Error loading state_dict: {e}")
            print("Trying to load compatible keys only...")
            
            # Get current state dict
            current_state_dict = model.state_dict()
            
            # Filter compatible keys
            compatible_state_dict = {}
            for key in state_dict.keys():
                if key in current_state_dict:
                    if state_dict[key].shape == current_state_dict[key].shape:
                        compatible_state_dict[key] = state_dict[key]
                    else:
                        print(f"Shape mismatch for {key}: {state_dict[key].shape} vs {current_state_dict[key].shape}")
                else:
                    print(f"Key not in current model: {key}")
            
            model.load_state_dict(compatible_state_dict, strict=False)
            print(f"Loaded {len(compatible_state_dict)}/{len(state_dict)} parameters")
        
        model.eval()
        print("Model loaded and ready.")
        
       

        # -------------------- SETTINGS: DATA TRANSFORMS
       
        transform_test = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
          ])

        # -------------------- SETTINGS: OPTIMIZER & SCHEDULER
       
        lr_begin = 0.001
        
        param_groups = [
            # Image encoder - higher learning rate
            {'params': model.image_encoder.parameters(), 'lr': lr_begin, 'name': 'image_encoder'},
            # Text encoder embedding layer - lower learning rate
            {'params': text_encoder.embedding_layer.parameters(), 'lr': lr_begin * 0.01, 'name': 'text_embedding'},
            # Text encoder attention and projection - medium learning rate
            {'params': text_encoder.attention_pool.parameters(), 'lr': lr_begin * 0.1, 'name': 'text_attention'},
            {'params': text_encoder.proj.parameters(), 'lr': lr_begin * 0.1, 'name': 'text_proj'},
            {'params': text_encoder.ln.parameters(), 'lr': lr_begin * 0.1, 'name': 'text_ln'},
            {'params': text_encoder.layer_norm1.parameters(), 'lr': lr_begin * 0.1, 'name': 'text_ln1'},
            {'params': text_encoder.layer_norm2.parameters(), 'lr': lr_begin * 0.1, 'name': 'text_ln2'},
            # Temperature parameter - higher learning rate
            {'params': [model.logit_scale], 'lr': lr_begin * 10, 'name': 'temperature'},
        ]
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
        
        # Use custom scheduler with warmup
        scheduler = WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_epochs=10,
            total_epochs=nb_epoch,
            min_lr_ratio=0.01
        )
        # ---- TRAIN THE NETWORK
        train_data, val_data, test_data = prepare_data_4_risk_data()
        
        risk_data = [train_data, val_data, test_data]

       
        testset, class_names = load_office31_dataset('Amazon', 'test', transform_test, 0.1)
        
        TestDataLoader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=0)
        LSLLoss = LabelSmoothingLoss(class_num, 0.1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        max_test_acc=0

        for epochID in range(0, nb_epoch):
            #Use different models for different epochs
            print('/Datasets/{}/train'.format(store_name))
            _,train_pre=VLMAdapter.test('/Datasets/{}'.format(store_name),model, 'RESNET-101', class_num, False, 1,
                                                        256,244, nb_epoch, 'train')
            _,val_pre = VLMAdapter.Valtest(
                '/Datasets/{}'.format(store_name),  # pathImgTest
                model,                                            # pathModel
                'RESNET-101',                                     # nnArchitecture
                None,                                             # testdataloader (correct!)
                False,                                            # nnIsTrained
                1,                                                # batch_size
                256,                                              # transResize
                244,                                              # transCrop
                'val',                                            # split
                False                                             # ckpt
            )
            _,test_pre=VLMAdapter.test('/Datasets/{}'.format(store_name),model, 'RESNET-101',class_num, False, 1,
                                                        256,244, nb_epoch, 'test')

            my_risk_model = prepare_data_4_risk_model(risk_data[0], risk_data[1], risk_data[2])
            train_one_pre = torch.empty((0, 1), dtype=torch.float64)
            val_one_pre = torch.empty((0, 1), dtype=torch.float64)
            test_one_pre = torch.empty((0, 1), dtype=torch.float64)

            a, _ = torch.max(train_pre, 1)
            b, _ = torch.max(val_pre, 1)
            c, _ = torch.max(test_pre, 1)

            train_one_pre = torch.cat((train_one_pre.cpu(), torch.reshape(a, (-1, 1))), dim=0).cpu().numpy()
            val_one_pre = torch.cat((val_one_pre.cpu(), torch.reshape(b, (-1, 1))), dim=0).cpu().numpy()
            test_one_pre = torch.cat((test_one_pre.cpu(), torch.reshape(c, (-1, 1))), dim=0).cpu().numpy()
            train_labels = torch.argmax(train_pre, 1).cpu().numpy()
            print(train_labels)
            val_labels = torch.argmax(val_pre, 1).cpu().numpy()
            test_labels = torch.argmax(test_pre, 1).cpu().numpy()

            my_risk_model.train(train_one_pre, val_one_pre, test_one_pre, train_pre.cpu().numpy(),
                                     val_pre.cpu().numpy(),
                                     test_pre.cpu().numpy(), train_labels, val_labels, test_labels, epochID)
            my_risk_model.predict(test_one_pre, test_pre.cpu().numpy(), )

            test_num = my_risk_model.test_data.data_len
            test_ids = my_risk_model.test_data.data_ids
            test_pred_y = test_labels
            test_true_y = my_risk_model.test_data.true_labels
            risk_scores = my_risk_model.test_data.risk_values

            id_2_label_index = dict()
            id_2_VaR_risk = []
            for i in range(test_num):
                id_2_VaR_risk.append([test_ids[i], risk_scores[i]])
                id_2_label_index[test_ids[i]] = i
            id_2_VaR_risk = sorted(id_2_VaR_risk, key=lambda item: item[1], reverse=True)
            #if epochID == 0:
            #    output_risk_scores(exp_dir + '/risk_score.txt', id_2_VaR_risk, id_2_label_index, test_true_y,
            #                       test_pred_y)
            print('this is epoch: {epochID}')
            output_risk_scores(exp_dir + f'/risk_score_epoch{epochID}.txt', id_2_VaR_risk, id_2_label_index, test_true_y, test_pred_y)
            id_2_risk = []
            for i in range(test_num):
                test_pred = test_one_pre[i]
                m_label = test_pred_y[i]
                t_label = test_true_y[i]
                if m_label == t_label:
                    label_value = 0.0
                else:
                    label_value = 1.0
                id_2_risk.append([test_ids[i], 1 - test_pred])
            id_2_risk_desc = sorted(id_2_risk, key=lambda item: item[1], reverse=True)
            #if epochID == 0:
            #output_risk_scores(exp_dir + '/base_score.txt', id_2_risk_desc, id_2_label_index, test_true_y,
            #                       test_pred_y)
            output_risk_scores(exp_dir + f'/base_score_epoch{epochID}.txt', id_2_risk_desc, id_2_label_index, test_true_y, test_pred_y)
            budgets = [10, 20, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
            risk_correct = [0] * len(budgets)
            base_correct = [0] * len(budgets)
            for i in range(test_num):
                for budget in range(len(budgets)):
                    if i < budgets[budget]:
                        pair_id = id_2_VaR_risk[i][0]
                        _index = id_2_label_index.get(pair_id)
                        if test_true_y[_index] != test_pred_y[_index]:
                            risk_correct[budget] += 1
                        pair_id = id_2_risk_desc[i][0]
                        _index = id_2_label_index.get(pair_id)
                        if test_true_y[_index] != test_pred_y[_index]:
                            base_correct[budget] += 1


            risk_loss_criterion = risk_model.RiskLoss(my_risk_model)
            risk_loss_criterion = risk_loss_criterion.cuda()

            rule_mus = torch.tensor(my_risk_model.test_data.get_risk_mean_X_discrete(), dtype=torch.float64).cuda()
            machine_mus = torch.tensor(my_risk_model.test_data.get_risk_mean_X_continue(), dtype=torch.float64).cuda()
            rule_activate = torch.tensor(my_risk_model.test_data.get_rule_activation_matrix(),
                                         dtype=torch.float64).cuda()
            machine_activate = torch.tensor(my_risk_model.test_data.get_prob_activation_matrix(),
                                            dtype=torch.float64).cuda()
            machine_one = torch.tensor(my_risk_model.test_data.machine_label_2_one, dtype=torch.float64).cuda()
            risk_y = torch.tensor(my_risk_model.test_data.risk_labels, dtype=torch.float64).cuda()

            test_ids = my_risk_model.test_data.data_ids
            test_ids_dict = dict()
            for ids_i in range(len(test_ids)):
                test_ids[ids_i] = os.path.basename(
                    test_ids[ids_i])
                test_ids_dict[test_ids[ids_i]] = ids_i

            del my_risk_model

            data_len = len(risk_y)

            model.train()

            # ---------------------------
            # Preparation (run once before batch loop)
            # ---------------------------
            
            prompt_template = "a photo of {}"
            prompts = [prompt_template.format(name.replace("_", " ")) for name in class_names]
            tokenizer = AutoTokenizer.from_pretrained(
                '/huggingface/hub/models--deepseek-ai--deepseek-llm-7b-base/snapshots/main/'
            )
            class_enc = tokenizer(
                prompts,
                padding='max_length',
                truncation=True,
                max_length=16,
                return_tensors='pt'
            )
            class_input_ids = class_enc['input_ids'].to(device)
            class_attention_mask = class_enc['attention_mask'].to(device)
            
            # Risk embedding layer
            risk_embed_layer = torch.nn.Linear(1, model.text_encoder.proj.out_features).to(device)  # proj_dim=512
            
            # Optimizers
            optimizer_img = optimizer  # your existing optimizer for image encoder
            optimizer_txt = torch.optim.AdamW(
                list(model.text_encoder.parameters()) + list(risk_embed_layer.parameters()),
                lr=1e-5,
                weight_decay=1e-6
            )
            
            # ---------------------------
            # Main batch loop
            # ---------------------------
            for batch_idx, (inputs, targets, paths) in enumerate(TestDataLoader):
            
                optimizer_img.zero_grad()
                optimizer_txt.zero_grad()
            
                if inputs.shape[0] < batch_size:
                    continue
            
                # Handle crops
                chex = 0
                if chex == 1:
                    bs, n_crops, c, h, w = inputs.size()
                    inputs = inputs.view(-1, c, h, w)
            
                inputs, targets = inputs.to(device), targets.to(device)
                inputs, targets = Variable(inputs), Variable(targets)
            
                # Map paths to indices for risk calculation
                index = [test_ids_dict[os.path.basename(p)] for p in paths]
                test_pre_batch = test_pre[index]
                rule_mus_batch = rule_mus[index]
                machine_mus_batch = machine_mus[index]
                rule_activate_batch = rule_activate[index]
                machine_activate_batch = machine_activate[index]
                machine_one_batch = machine_one[index]
            
                # === Image embeddings ===
                img_emb = model.image_encoder(inputs)  # (batch_size, proj_dim=512)
                img_emb = F.normalize(img_emb, dim=-1)
            
                # === Compute preliminary logits for softmax / y_score (needed for risk) ===
                with torch.no_grad():
                    # Token embeddings
                    token_embeds = model.text_encoder.embedding_layer(class_input_ids)
                    # Attention pooling
                    query = token_embeds.mean(dim=1, keepdim=True)
                    attended, _ = model.text_encoder.attention_pool(
                        query=query,
                        key=token_embeds,
                        value=token_embeds
                    )
                    attended = attended.squeeze(1)
                    attended = model.text_encoder.layer_norm1(attended)
                    # Projection
                    base_txt_embeds = model.text_encoder.ln(model.text_encoder.proj(attended))
                    base_txt_embeds = F.normalize(base_txt_embeds, dim=-1)  # shape: num_classes x proj_dim
                    xc_temp = img_emb @ base_txt_embeds.t()
                y_score = softmax_with_temperature(xc_temp.cpu(), 2)
            
                out_2 = 1 - xc_temp
                out_temp = xc_temp.view(-1, 1)
                out_2 = out_2.view(-1, 1)
                out_2D = torch.cat((out_temp, out_2), dim=1)
            
                # === Compute risk labels ===
                risk_labels = risk_loss_criterion(
                    test_pre_batch,
                    rule_mus_batch,
                    machine_mus_batch,
                    rule_activate_batch,
                    machine_activate_batch,
                    machine_one_batch,
                    y_score,
                    labels=None
                ).to(device)
            
                batch_risk_scalar = risk_labels.float().mean().detach().unsqueeze(-1)
            
                # === Compute class text embeddings with dynamic risk injection (AFTER projection) ===
                # Token embeddings
                token_embeds = model.text_encoder.embedding_layer(class_input_ids)
                # Attention pooling
                query = token_embeds.mean(dim=1, keepdim=True)
                attended, _ = model.text_encoder.attention_pool(
                    query=query,
                    key=token_embeds,
                    value=token_embeds
                )
                attended = attended.squeeze(1)
                attended = model.text_encoder.layer_norm1(attended)
                
                # Project to 512-dim
                proj_embeds = model.text_encoder.ln(model.text_encoder.proj(attended))  # (num_classes, 512)
                
                # Add risk bias in projected space
                risk_bias = risk_embed_layer(batch_risk_scalar.to(device)).squeeze(0)  # (512,)
                proj_embeds = proj_embeds + risk_bias  # broadcasting fine now
                
                class_txt_embs = F.normalize(proj_embeds, dim=-1)

            
                # === Cross-modal logits ===
                xc = img_emb @ class_txt_embs.t()  # shape: batch_size x num_classes
                out = xc
            
                # === Save batch info for inspection ===
                with open('/PMG/risk_lable.txt', 'a') as file:
                    file.write(f'{batch_idx}\n')
                    np.savetxt(file, out.detach().cpu().numpy(), delimiter=',')
                    np.savetxt(file, risk_labels.detach().cpu().numpy().astype(float), delimiter=',')
                    np.savetxt(file, targets.detach().cpu().numpy(), delimiter=',')
                    file.write('\n')

            
                # === Compute LSLLoss and backprop ===
                Loss = LSLLoss(out, risk_labels)
                Loss.backward(retain_graph=True)
            
                # === Dynamic risk scaling for text encoder update ===
                dynamic_scale = torch.clamp(batch_risk_scalar, 0.1, 1.0).item()
            
                # Update image encoder
                optimizer_img.step()
            
                # Update text encoder proportionally to batch risk
                for g in optimizer_txt.param_groups:
                    base_lr = g.get('base_lr', g['lr'])
                    g['lr'] = base_lr * dynamic_scale
                optimizer_txt.step()
            
                # Restore base LR
                for g in optimizer_txt.param_groups:
                    if 'base_lr' in g:
                        g['lr'] = g['base_lr']
                    else:
                        g['base_lr'] = g['lr']

            # === Run evaluation on test set ===
            test_acc, test_pre = VLMAdapter.test(
                f'/Datasets/{store_name}/',
                model,
                'CLIP',  
                class_num,
                False,
                1,
                256,
                244,
                nb_epoch, 
                'test'
            )
        print(max_test_acc)





    # --------------------------------------------------------------------------------



    def test(pathImgTest, pathModel, nnArchitecture, class_num, nnIsTrained, batch_size,
         transResize, transCrop, nb_epoch, split, ckpt=False):

        # -------------------- Load model --------------------
        model = pathModel
        model.eval()
        model.cuda()
    
        # -------------------- Dataset & transforms --------------------
        chex = 1
        # Use CIFAR-100 normalization stats instead of ImageNet
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    
        
        if split == 'train':
            testset, class_names  = load_office31_dataset('Amazon', split, transform_test, 0.1)
        else:
            testset, class_names  = load_office31_dataset('Amazon', split, transform_test, 0.1) 
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=4
        )
    
        # -------------------- Prompt preparation --------------------
        prompt_template = "a photo of {}"
        prompts = [prompt_template.format(name.replace("_", " ")) for name in class_names]
        tokenizer = AutoTokenizer.from_pretrained(
            '/.cache/huggingface/hub/models--deepseek-ai--deepseek-llm-7b-base/snapshots/main/'
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # -------------------- Tracking --------------------
        all_preds, all_targets, y_score, paths = [], [], [], []
        
        with torch.no_grad():
            # === Encode class prompts (text side) ===
            class_enc = tokenizer(
                prompts,
                padding='max_length',
                truncation=True,
                max_length=16,
                return_tensors='pt'
            )
            class_input_ids = class_enc['input_ids'].to(device)
            class_attns = class_enc['attention_mask'].to(device)
    
            # Get class text embeddings (C, D)
            class_txt_embs = model.text_encoder(class_input_ids, class_attns)
            class_txt_embs = F.normalize(class_txt_embs, dim=-1)
    
            # === Iterate over test data ===
            for images, targets, _ in tqdm(testloader, ncols=80):
                if chex == 1 and images.dim() == 5:
                    bs, n_crops, c, h, w = images.size()
                    images = images.view(-1, c, h, w).to(device)
                    img_emb = model.image_encoder(images)              # (bs*n_crops, D)
                    img_emb = img_emb.view(bs, n_crops, -1).mean(1)    # average crops
                else:
                    images = images.to(device)
                    img_emb = model.image_encoder(images)              # (bs, D)
    
                # Normalize embeddings
                img_emb = F.normalize(img_emb, dim=-1)
    
                # Similarity scores (bs, C)
                sims = img_emb @ class_txt_embs.t()
    
                # Predictions + calibrated softmax scores
                preds = sims.argmax(dim=1).cpu().tolist()
                scores = softmax(sims.cpu().numpy(), axis=1).tolist()
    
                # Collect results
                all_preds.extend(preds)
                y_score.extend(scores)
                all_targets.extend(targets.cpu().tolist())
    
            
            y_score = np.array(y_score)
    
            # === Metrics ===
            test_acc = 100.0 * accuracy_score(all_targets, all_preds)
            per_class_f1 = 100.0 * f1_score(all_targets, all_preds, average=None)
            test_f1_mean = np.mean(per_class_f1)
            test_recall = 100.0 * recall_score(all_targets, all_preds, average='weighted')
            test_precision = 100.0 * precision_score(all_targets, all_preds, average='weighted')
            )
    
            # Print summary
            print(f"\n{'='*60}")
            print(f"CIFAR-100 Test Results")
            print(f"{'='*60}")
            print(f"Accuracy:       {test_acc:.2f}%")
            print(f"F1 Score (mean): {test_f1_mean:.2f}%")
            print(f"Precision:      {test_precision:.2f}%")
            print(f"Recall:         {test_recall:.2f}%")
            #print(f"AUC:            {test_auc:.2f}%")
            print(f"{'='*60}")
    
            # Per-class F1 (show first 10 classes only for brevity)
            print(f"\nPer-class F1 Scores (first 10 of {len(class_names)}):")
            for i, f1 in enumerate(per_class_f1[:10]):
                print(f"  Class {class_names[i]}: {f1:.2f}%")
            if len(class_names) > 10:
                print(f"  ... and {len(class_names)-10} more classes")
    
            # Confusion Matrix
            cm = confusion_matrix(all_targets, all_preds)
            print(f"\nConfusion Matrix shape: {cm.shape} (100x100)")
            
            # Print diagonal of confusion matrix (correct predictions per class)
            print(f"\nCorrect predictions per class (diagonal):")
            for i in range(min(10, len(class_names))):  # Show first 10
                print(f"  {class_names[i]}: {cm[i,i]}")
    
           
    
        return test_acc, torch.tensor(y_score)

    def Valtest(pathImgTest, pathModel, nnArchitecture, testdataloader, nnIsTrained, batch_size, transResize, transCrop, split, ckpt=False):
        """
        Validation / Test function for CLIP-like models with image_encoder + text_encoder.
        """
        model = pathModel
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        model.to(device)
    
        # -------------------- Dataset & Transforms --------------------
        
        normalize = transforms.Normalize([0.5071, 0.4867, 0.4408], 
                                         [0.2675, 0.2565, 0.2761])
       
        #])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        testset, class_names   = load_office31_dataset('Amazon', split, transform_test, 0.1)
        
        # Use provided testdataloader if available, otherwise create one
        if testdataloader is not None:
            testloader = testdataloader
        else:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False, num_workers=4
            )
    
        # -------------------- Prepare Class Prompts --------------------
        prompt_template = "a photo of {}"
        prompts = [prompt_template.format(name.replace("_", " ")) for name in class_names]
        tokenizer = AutoTokenizer.from_pretrained(
            '/.cache/huggingface/hub/models--deepseek-ai--deepseek-llm-7b-base/snapshots/main/'
        )
        
        # Encode class prompts using model's text_encoder
        with torch.no_grad():
            class_enc = tokenizer(prompts, padding='max_length', truncation=True,
                                  max_length=16, return_tensors='pt')
            class_input_ids = class_enc['input_ids'].to(device)
            class_attns = class_enc['attention_mask'].to(device)
            class_txt_embs = model.text_encoder(class_input_ids, class_attns)  # (C, D)
            class_txt_embs = F.normalize(class_txt_embs, dim=-1)
    
        # -------------------- Evaluation --------------------
        all_preds, all_targets, y_score, paths = [], [], [], []
    
        with torch.no_grad():
            for batch in tqdm(testloader, ncols=80):
                # Unpack batch - CIFAR-100 dataset returns (image, label, path)
                if len(batch) == 3:
                    images, targets, batch_paths = batch
                elif len(batch) == 2:
                    images, targets = batch
                    batch_paths = [f"img_{i}" for i in range(len(targets))]  # dummy paths
                else:
                    raise ValueError(f"Unexpected batch format: {batch}")
    
                images, targets = images.to(device), targets.to(device)
    
                # Multi-crop handling
                if images.dim() == 5:  # bs, n_crops, c, h, w
                    bs, n_crops, c, h, w = images.size()
                    images = images.view(-1, c, h, w)
                    img_emb = model.image_encoder(images)  # (bs*n_crops, D)
                    img_emb = img_emb.view(bs, n_crops, -1).mean(1)  # average over crops
                else:
                    img_emb = model.image_encoder(images)  # (bs, D)
    
                img_emb = F.normalize(img_emb, dim=-1)
    
                # Similarity between image embeddings and class text embeddings
                sims = img_emb @ class_txt_embs.t()  # (bs, C)
                preds = sims.argmax(dim=1).cpu().tolist()
                scores = softmax(sims.cpu().numpy(), axis=1).tolist()
    
                # Track results
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().tolist())
                y_score.extend(scores)
                paths.extend(batch_paths)
    
        # -------------------- Metrics --------------------
        y_score_arr = np.array(y_score)
        test_acc = 100.0 * accuracy_score(all_targets, all_preds)
        per_class_f1 = 100.0 * f1_score(all_targets, all_preds, average=None)
        test_f1_mean = np.mean(per_class_f1)
        test_recall = 100.0 * recall_score(all_targets, all_preds, average='weighted')
        test_precision = 100.0 * precision_score(all_targets, all_preds, average='weighted')
        #test_auc = 100.0 * roc_auc_score(all_targets, y_score_arr, multi_class="ovr", average="weighted")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"CIFAR-100 Test Results")
        print(f"{'='*60}")
        print(f"Accuracy:       {test_acc:.2f}%")
        print(f"F1 Score (mean): {test_f1_mean:.2f}%")
        print(f"Precision:      {test_precision:.2f}%")
        print(f"Recall:         {test_recall:.2f}%")
        #print(f"AUC:            {test_auc:.2f}%")
        print(f"{'='*60}")
        # KEEPING THE EXACT SAME PRINT FORMAT AS ORIGINAL CODE
        for i, f1 in enumerate(per_class_f1):
            print(f"Class {class_names[i]}: F1 Score = {f1:.4f}")
    
        print("Dataset \t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(
            test_acc, test_f1_mean, test_precision, test_recall
        ))
    
        return test_acc, torch.tensor(y_score_arr)


    
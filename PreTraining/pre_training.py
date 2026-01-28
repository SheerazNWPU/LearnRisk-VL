# train_clip_deepseek_domain_adaptation.py
import os
os.environ['TORCH_HOME'] = '/.torch_cache'
import pickle
import argparse
import random
import shutil
from os.path import join
from glob import glob
from collections import Counter
import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from text_encoder import SimpleDeepSeekTextEncoder, SimpleBERTTextEncoder, SimpleDeBERTaTextEncoder, SimpleFlanT5TextEncoder, SimpleQwenTextEncoder, SimpleLLaVATextEncoder, SimpleGPTTextEncoder
from torchvision import transforms, datasets
from PIL import Image
import torchvision

# HF transformers (needed for text encoder)
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig

# your model zoos and dataset loader imports (assumed to be in same folder)
from Densenet import densenet121, densenet161, densenet169, densenet201
from Resnet import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    resnext50_32x4d, wide_resnet50_2, wide_resnet101_2
)
from Efficientnet import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
)
from Vgg import vgg11, vgg13, vgg16, vgg19

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from scipy.special import softmax

# -------------------------
# Arg parsing
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-m','--multiple', default=2, type=int, help='multiple of input size')
parser.add_argument('--ckpt', default='', help='path of check_point model')
parser.add_argument('--requires_grad', default='all', help='the layers need finetune (fc/all)')
parser.add_argument('--source_dataset', default='Office31_Amazon', 
                    choices=['CIFAR100', 'CIFAR10', 'CIFAR10', 'TinyImageNet', 'Office31_Amazon', 'Office31_Webcam', 'Office31_DSLR'],
                    help='source dataset for training')
parser.add_argument('--target_dataset', default='Office31_Webcam', 
                    choices=['', 'Office31_Webcam', 'CIFAR10', 'STL10', 'TinyImageNet', 'Office31_Amazon', 'Office31_Webcam', 'Office31_DSLR', 'CIFAR100-C', 'CIFAR10-C'],
                    help='target dataset for domain adaptation (if empty, use source for test)')
parser.add_argument('--domain_adaptation', default=1, type=int, help='1 for domain adaptation, 0 for standard training')
parser.add_argument('-c','--cnn', default='r50', help='CNN model')
parser.add_argument('-b','--batch_size', default=32, type=int, help='batch_size')
parser.add_argument('--wt', default=0, type=int, help='weight loss')
parser.add_argument('-g','--gpu', default='1', help='gpu id')
parser.add_argument('--train_set', default='train', help='name of training set')
parser.add_argument('--test_set', default='test', help='name of testing set')
parser.add_argument('-w','--num_workers', default=4, type=int, help='num_workers')
parser.add_argument('-e','--epoch', default=100, type=int, help='epoch')
parser.add_argument('--chex', default=0, type=int, help='use chexnet setting or not (set to 0 for CIFAR)')
parser.add_argument('-r','--random_seed', default=0, type=int, help='random seed')
parser.add_argument('-s','--save_dir', default='Office31_AmazonWEB', help='save_dir')
parser.add_argument('-l','--label_smooth', default=0, type=float, help='label_smooth')
parser.add_argument('--learning_rate', default=0.001, type=float, help='learning_rate')
parser.add_argument('--scheduler', default=1, type=int, help='use scheduler')
parser.add_argument('-v','--evaluate', default=1, type=int, help='test every epoch')
parser.add_argument('-a', '--amp', default=2, type=int, help='0: off, 1: apex.amp, 2: torch.cuda.amp')
# Add text_model choices
parser.add_argument(
    '--text_model',
    default='gpt2',
    choices=['deepseek', 'bert', 'deberta', 'flan_t5', 'qwen', 'llava', 'gpt2'],
    help='Text encoder model to use'
)
parser.add_argument('--max_len', type=int, default=32, help='max text length')
parser.add_argument('--grad_clip', type=float, default=1.0, help='max grad norm (0 to disable)')
# Local model path (required for offline)
parser.add_argument('--local_model_path', type=str, default='/.cache/huggingface/hub/models--deepseek-ai--deepseek-llm-7b-base/snapshots/main/', help='Path to locally cached HF model/tokenizer')
parser.add_argument('--cifar_style', type=int, default=1, help='Use CIFAR-style transforms (1) or medical-style (0)')
parser.add_argument('--val_split', type=float, default=0.1, help='Fraction of training data to use as validation')
parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs')
parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
parser.add_argument('--debug_gradients', type=int, default=0, help='Print gradient information')
# NEW: Add temperature parameter
parser.add_argument('--init_temperature', type=float, default=0.07, help='Initial temperature value (logit scale) - CLIP default is 0.07')
parser.add_argument('--temperature_clip_min', type=float, default=0.01, help='Minimum temperature (maximum logit scale)')
parser.add_argument('--temperature_clip_max', type=float, default=100.0, help='Maximum temperature (minimum logit scale)')
parser.add_argument('--temperature_reg_weight', type=float, default=0.001, help='Weight for temperature regularization')
parser.add_argument('--visualize_similarity', type=int, default=0, help='Visualize similarity matrix every N epochs (0 to disable)')
args = parser.parse_args()

# -------------------------
# Basic settings
# -------------------------
torch.multiprocessing.set_sharing_strategy('file_system')
chex = args.chex
num_epoch = args.epoch
begin_epoch = 1
seed = args.random_seed
cnn = args.cnn
batch_size = args.batch_size
accumulation_steps = args.accumulation_steps
domain_adaptation = args.domain_adaptation
source_dataset = args.source_dataset
target_dataset = args.target_dataset if args.target_dataset else source_dataset

if args.learning_rate:
    lr_begin = args.learning_rate
else:
    lr_begin = (batch_size / 256) * 0.1
use_amp = args.amp
test_every_epoch = args.evaluate
cifar_style = args.cifar_style
val_split = args.val_split

Image.MAX_IMAGE_PIXELS = None

exp_dir = "/result_archive/{}_{}".format(args.save_dir, args.text_model)
os.makedirs(exp_dir, exist_ok=True)
os.makedirs(exp_dir, exist_ok=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# seeds
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# -------------------------
# Dataset-specific configurations
# -------------------------
def get_dataset_config(dataset_name):
    """Get dataset-specific configuration"""
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
        config['dataset_type'] = 'cifar'
        
    elif dataset_name == 'CIFAR10':
        config['normalize_mean'] = (0.4914, 0.4822, 0.4465)
        config['normalize_std'] = (0.2470, 0.2435, 0.2616)
        config['image_size'] = 32
        config['class_names'] = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        config['dataset_type'] = 'cifar'
        
    elif dataset_name == 'STL10':
        config['normalize_mean'] = (0.4914, 0.4822, 0.4465)
        config['normalize_std'] = (0.2470, 0.2435, 0.2616)
        config['image_size'] = 96
        config['class_names'] = [
            'airplane', 'bird', 'car', 'cat', 'deer',
            'dog', 'horse', 'monkey', 'ship', 'truck'
        ]
        config['dataset_type'] = 'stl10'
        
    elif dataset_name == 'TinyImageNet':
        config['normalize_mean'] = (0.4802, 0.4481, 0.3975)
        config['normalize_std'] = (0.2770, 0.2691, 0.2821)
        config['image_size'] = 64
        # TinyImageNet has 200 classes - we'll load actual class names from dataset
        config['dataset_type'] = 'tinyimagenet'
        
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
        config['dataset_type'] = 'office'
        # Add path info for debugging
        config['expected_path'] = f'/Datasets/Office-31/{dataset_name.split("_")[1].lower()}/'
        
    elif 'CIFAR100-C' in dataset_name:
        config['normalize_mean'] = (0.5071, 0.4867, 0.4408)
        config['normalize_std'] = (0.2675, 0.2565, 0.2761)
        config['image_size'] = 32
        config['is_corrupted'] = True
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
        config['dataset_type'] = 'cifar_corrupted'
        
    elif 'CIFAR10-C' in dataset_name:
        config['normalize_mean'] = (0.4914, 0.4822, 0.4465)
        config['normalize_std'] = (0.2470, 0.2435, 0.2616)
        config['image_size'] = 32
        config['is_corrupted'] = True
        config['class_names'] = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        config['dataset_type'] = 'cifar_corrupted'
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return config

# Get dataset configs
source_config = get_dataset_config(source_dataset)
target_config = get_dataset_config(target_dataset)

# Check if class names match for domain adaptation
if domain_adaptation and source_config['class_names'] and target_config['class_names']:
    if source_config['class_names'] != target_config['class_names']:
        print(f"[WARNING] Source and target datasets have different class names!")
        print(f"  Source classes: {len(source_config['class_names'])}")
        print(f"  Target classes: {len(target_config['class_names'])}")
        # For domain adaptation, we'll use source class names for prompts
        print(f"  Using source class names for prompts")

# Use source class names for prompts
class_names = source_config['class_names'] if source_config['class_names'] else []

if cifar_style:
    # Standard transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(source_config['image_size'], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(source_config['normalize_mean'], source_config['normalize_std']),
    ])
    
    # Test transform for source dataset - ADD RESIZE
    transform_test_source = transforms.Compose([
        transforms.Resize((source_config['image_size'], source_config['image_size'])),  # ADD THIS
        transforms.ToTensor(),
        transforms.Normalize(source_config['normalize_mean'], source_config['normalize_std']),
    ])
    
    # Test transform for target dataset - ADD RESIZE
    transform_test_target = transforms.Compose([
        transforms.Resize((target_config['image_size'], target_config['image_size'])),  # ADD THIS
        transforms.ToTensor(),
        transforms.Normalize(target_config['normalize_mean'], target_config['normalize_std']),
    ])
else:
    # Medical-style transforms (if needed)
    if chex == 0:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])
        
        transform_test_source = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])
        transform_test_target = transform_test_source
    else:
        normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        
        transformList = []
        transformList.append(transforms.Resize(256))
        transformList.append(transforms.FiveCrop(224))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transform_test_source = transforms.Compose(transformList)
        transform_test_target = transform_test_source

# -------------------------
# Model zoo
# -------------------------
model_zoo = {
    'r18':resnet18, 'r34':resnet34, 'r50':resnet50, 'r101':resnet101, 'r152':resnet152,
    'd121':densenet121, 'd161':densenet161, 'd169':densenet169, 'd201':densenet201,
    'v11':vgg11, 'v13':vgg13, 'v16':vgg16, 'v19':vgg19,
    'eb0':efficientnet_b0, 'eb1':efficientnet_b1, 'eb2':efficientnet_b2, 'eb3':efficientnet_b3,
    'eb4':efficientnet_b4, 'eb5':efficientnet_b5, 'eb6':efficientnet_b6,  'eb7':efficientnet_b7,
    'rx50':resnext50_32x4d, 'wrn50':wide_resnet50_2, 'wrn101':wide_resnet101_2
}

# -------------------------
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

def load_cifar_corrupted(dataset_name, split='test', transform=None, corruption_type='gaussian_noise'):
    """Load CIFAR-C corrupted datasets - CORRECTED VERSION for 10,000 images"""
    
    if '100' in dataset_name:
        base_name = 'CIFAR100'
        num_classes = 100
        data_path = '/Datasets/CIFAR-100-C/'
    else:
        base_name = 'CIFAR10'
        num_classes = 10
        data_path = '/Datasets/CIFAR-10-C/'
    
    # Verify path exists
    if not os.path.exists(data_path):
        print(f"[ERROR] CIFAR-C dataset not found at {data_path}")
        raise FileNotFoundError(f"CIFAR-C dataset not found at {data_path}")
    
    print(f"[INFO] Loading actual CIFAR-C from: {data_path}")
    
    # Available corruptions
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    # Use specified corruption or default
    if corruption_type not in corruptions:
        print(f"[WARNING] Corruption '{corruption_type}' not found. Using '{corruptions[0]}'")
        corruption_type = corruptions[0]
    
    # Load data and labels
    data_file = os.path.join(data_path, f'{corruption_type}.npy')
    labels_file = os.path.join(data_path, 'labels.npy')
    
    if not os.path.exists(data_file):
        print(f"[ERROR] Corruption file not found: {data_file}")
        raise FileNotFoundError(f"Corruption file not found: {data_file}")
    
    print(f"[INFO] Loading corruption: {corruption_type}")
    
    # ========== KEY FIX: Handle 50,000 images (5 severity levels) ==========
    images = np.load(data_file)  # Shape: (50000, 3072) for 5 severity levels
    labels = np.load(labels_file)  # Shape: (10000,)
    
    print(f"[DEBUG] Raw images shape: {images.shape}, Labels shape: {labels.shape}")
    
    # CIFAR-C files contain 5 severity levels  10,000 images = 50,000 total
    # We want ONLY severity level 5 (hardest corruption) - LAST 10,000 images
    if images.shape[0] == 50000:
        print(f"[INFO] Detected 5 severity levels. Taking severity level 5 (last 10,000 images)")
        images = images[40000:50000]  # Last severity level (indices 40000-49999)
        
        # Labels are only 10,000 for the original test set
        # They get repeated in the images array, but labels.npy is only 10,000
        # So we use the original 10,000 labels
        if labels.shape[0] != 10000:
            print(f"[WARNING] Unexpected labels shape: {labels.shape}, using original labels")
            labels = np.load(labels_file)  # Reload to ensure 10,000
    elif images.shape[0] == 10000:
        print(f"[INFO] Already 10,000 images (single severity level)")
    else:
        print(f"[WARNING] Unexpected image count: {images.shape[0]}, taking first 10,000")
        images = images[:10000]
    
    # Ensure we have exactly 10,000 images
    if images.shape[0] > 10000:
        images = images[:10000]
        print(f"[INFO] Truncated to 10,000 images")
    
    # Reshape from (10000, 3072) to (10000, 32, 32, 3)
    if images.shape[1] == 3072:
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        print(f"[INFO] Reshaped images to: {images.shape}")
    
    # Ensure labels are 10,000
    if labels.shape[0] != 10000:
        if labels.shape[0] == 50000:
            labels = labels[:10000]
        else:
            print(f"[WARNING] Labels shape mismatch: {labels.shape}, repeating to match")
            labels = np.tile(labels, (10000 // labels.shape[0] + 1))[:10000]
    
    print(f"[INFO] Final: {images.shape[0]} images, {labels.shape[0]} labels")
    
    # For validation split, take subset (1,000 images)
    if split == 'val':
        # Take first 1,000 images for validation
        images = images[:1000]
        labels = labels[:1000]
        print(f"[INFO] Using {len(images)} images for validation")
    elif split == 'train':
        # CIFAR-C doesn't have train split, use small subset for few-shot
        images = images[:1000]
        labels = labels[:1000]
        print(f"[INFO] Using {len(images)} images for training (small subset)")
    
    class CIFARCorruptedDataset(Dataset):
        def __init__(self, images, labels, transform=None, split='test'):
            self.images = images
            self.labels = labels.astype(np.int64)
            self.transform = transform
            self.split = split
            
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            # Convert numpy array to PIL Image
            img_array = self.images[idx]
            
            # Ensure uint8 dtype (CIFAR-C might be float 0-255)
            if img_array.dtype != np.uint8:
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
            img = Image.fromarray(img_array)
            label = self.labels[idx]
            
            if self.transform:
                img = self.transform(img)
            
            return img, label, f"cifar-c_{self.split}_{idx:06d}.png"
    
    # Create dataset
    dataset = CIFARCorruptedDataset(images, labels, transform, split)
    
    # Get class names from config
    source_config = get_dataset_config(base_name)
    class_names = source_config['class_names']
    
    print(f"[INFO] Successfully loaded CIFAR-100-C: {len(dataset)} {split} images")
    print(f"[INFO] Corruption: {corruption_type}, Severity: 5 (hardest)")
    
    return dataset, class_names
    

def load_dataset(dataset_name, split='train', transform=None, val_split=0.1):
    """Load dataset from various sources"""
    
    print(f"[INFO] Loading {dataset_name} {split} data")
    
    # Handle Office-31 datasets
    if 'Office31_' in dataset_name:
        domain = dataset_name.split('_')[1]
        return load_office31_dataset(domain, split, transform, val_split)
    
    # Handle CIFAR-C corrupted datasets
    # In load_dataset function, around the CIFAR-C handling:
    elif 'CIFAR' in dataset_name and '-C' in dataset_name:
        if split == 'train':
            print(f"[INFO] CIFAR-C {split}: Using clean CIFAR for training")
            # Load clean CIFAR for training
            if '100' in dataset_name:
                clean_name = 'CIFAR100'
            else:
                clean_name = 'CIFAR10'
            return load_dataset(clean_name, split, transform, val_split)
        else:
            # For validation/test, load actual corrupted data
            print(f"[INFO] CIFAR-C {split}: Loading actual corrupted data")
            return load_cifar_corrupted(dataset_name, split, transform)
    
    # Handle standard torchvision datasets
    else:
        if split == 'train' or split == 'val':
            # Load full training dataset
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
                try:
                    from torchvision.datasets import ImageFolder
                    full_dataset = ImageFolder(
                        root='./data/tiny-imagenet-200/train',
                        transform=None
                    )
                    # Extract class names from folder names
                    class_names = [name for name in os.listdir('./data/tiny-imagenet-200/train') 
                                  if os.path.isdir(os.path.join('./data/tiny-imagenet-200/train', name))]
                    class_names.sort()
                except:
                    raise ValueError(f"TinyImageNet dataset not found. Please download it first.")
            
            # Split into train and val
            num_samples = len(full_dataset)
            indices = list(range(num_samples))
            random.Random(42).shuffle(indices)
            
            split_idx = int(num_samples * (1 - val_split))
            
            if split == 'train':
                subset_indices = indices[:split_idx]
            else:  # val
                subset_indices = indices[split_idx:]
            
            # Apply transforms to subset
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
                    return img, label, f"{split}_{idx:06d}.png"
            
            return TransformedSubset(full_dataset, subset_indices, transform), class_names
            
        elif split == 'test':
            # Load test dataset
            if dataset_name == 'CIFAR100':
                test_dataset = datasets.CIFAR100(
                    root='./data',
                    train=False,
                    download=True,
                    transform=transform
                )
                class_names = test_dataset.classes
            elif dataset_name == 'CIFAR10':
                test_dataset = datasets.CIFAR10(
                    root='./data',
                    train=False,
                    download=True,
                    transform=transform
                )
                class_names = test_dataset.classes
            elif dataset_name == 'STL10':
                test_dataset = datasets.STL10(
                    root='./data',
                    split='test',
                    download=True,
                    transform=transform
                )
                class_names = test_dataset.classes
            elif dataset_name == 'TinyImageNet':
                try:
                    from torchvision.datasets import ImageFolder
                    test_dataset = ImageFolder(
                        root='./data/tiny-imagenet-200/val',
                        transform=transform
                    )
                    # Extract class names from folder names
                    class_names = [name for name in os.listdir('./data/tiny-imagenet-200/train') 
                                  if os.path.isdir(os.path.join('./data/tiny-imagenet-200/train', name))]
                    class_names.sort()
                except:
                    raise ValueError("TinyImageNet test dataset not found.")
            
            # Wrap to match our interface
            class WrappedTestDataset(Dataset):
                def __init__(self, dataset):
                    self.dataset = dataset
                    
                def __len__(self):
                    return len(self.dataset)
                    
                def __getitem__(self, idx):
                    img, label = self.dataset[idx]
                    return img, label, f"test_{idx:06d}.png"
            
            return WrappedTestDataset(test_dataset), class_names
        
        else:
            raise ValueError(f"Invalid split: {split}")

# -------------------------
# Image encoder
# -------------------------
class ImageEncoder(nn.Module):
    def __init__(self, backbone, cnn_name, proj_dim=512, image_size=224):
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
                x = torch.zeros(1, 3, image_size, image_size)
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

# -------------------------
# Text Encoder Factory
# -------------------------
class TextEncoderFactory:
    @staticmethod
    def create_text_encoder(model_name, proj_dim=512, local_model_path=None):
        """
        Create text encoder for different LLM architectures
        """
        MODEL_PATHS = {
            "deepseek": "/.cache/huggingface/hub/models--deepseek-ai--deepseek-llm-7b-base/snapshots/main",
            "bert": "/.cache/huggingface/hub/models--google-bert--bert-base-uncased/snapshots/main", 
            "flan_t5": "/.cache/huggingface/hub/models--google--flan-t5-large/snapshots/main",
            "deberta": "/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/main",
            "qwen": "/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/main",
            "llava": "/.cache/huggingface/hub/models--llava--llava-1.5-7b-hf/snapshots/main",
            "gpt2" : "/.cache/huggingface/hub/models--openai--gpt-2/snapshots/vlmgpt2"
        }
        
        if model_name == "deepseek":
            return SimpleDeepSeekTextEncoder(
                proj_dim=proj_dim,
                local_model_path=MODEL_PATHS["deepseek"]
            )
        elif model_name == "bert":
            return SimpleBERTTextEncoder(
                proj_dim=proj_dim,
                model_path=MODEL_PATHS["bert"]
            )
        elif model_name == "gpt2":
            return SimpleGPTTextEncoder(
                model_name="gpt2",
                proj_dim=proj_dim,
                model_path=MODEL_PATHS["gpt2"]
            )
        elif model_name == "deberta":
            return SimpleDeBERTaTextEncoder(
                proj_dim=proj_dim,
                model_path=MODEL_PATHS["deberta"]
            )
        elif model_name == "flan_t5":
            return SimpleFlanT5TextEncoder(
                proj_dim=proj_dim,
                model_path=MODEL_PATHS["flan_t5"]
            )
        elif model_name == "qwen":
            return SimpleQwenTextEncoder(
                proj_dim=proj_dim,
                model_path=MODEL_PATHS["qwen"]
            )
        elif model_name == "llava":
            return SimpleLLaVATextEncoder(
                proj_dim=proj_dim,
                model_path=MODEL_PATHS["llava"]
            )
        else:
            raise ValueError(f"Unknown text model: {model_name}")
    
    @staticmethod
    def create_tokenizer(model_name):
        """Create tokenizer for the model"""
        MODEL_PATHS = {
            "deepseek": "/.cache/huggingface/hub/models--deepseek-ai--deepseek-llm-7b-base/snapshots/main",
            "bert": "/.cache/huggingface/hub/models--google-bert--bert-base-uncased/snapshots/main", 
            "flan_t5": "/.cache/huggingface/hub/models--google--flan-t5-large/snapshots/main",
            "deberta": "/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/main",
            "qwen": "/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/main",
            "llava": "/.cache/huggingface/hub/models--llava--llava-1.5-7b-hf/snapshots/main",
            "gpt2" : "/.cache/huggingface/hub/models--openai--gpt-2/snapshots/vlmgpt2"
        }
        
        if model_name == "deepseek":
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATHS["deepseek"],
                trust_remote_code=True,
                local_files_only=True
            )
        elif model_name == "bert":
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained(
                MODEL_PATHS["bert"],
                local_files_only=True
            )
        elif model_name == "deberta":
            from transformers import DebertaV2Tokenizer
            tokenizer = DebertaV2Tokenizer.from_pretrained(
                MODEL_PATHS["deberta"],
                local_files_only=True
            )
        elif model_name == "flan_t5":
            from transformers import T5Tokenizer
            tokenizer = T5Tokenizer.from_pretrained(
                MODEL_PATHS["flan_t5"],
                local_files_only=True
            )
        elif model_name == "gpt2":
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATHS["gpt2"],
                local_files_only=True
            )
        elif model_name == "qwen":
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATHS["qwen"],
                trust_remote_code=True,
                local_files_only=True
            )
        elif model_name == "llava":
            from transformers import LlavaNextProcessor
            processor = LlavaNextProcessor.from_pretrained(
                MODEL_PATHS["llava"],
                local_files_only=True
            )
            tokenizer = processor.tokenizer
        else:
            raise ValueError(f"Unknown text model: {model_name}")
        
        # Set pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        
        return tokenizer

# -------------------------
# FIXED CLIP model with temperature control
# -------------------------
class CLIPModel(nn.Module):
    def __init__(self, image_encoder: ImageEncoder, text_encoder: nn.Module, 
                 init_logit_scale=0.07, temp_min=0.01, temp_max=100.0):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        
        # Proper initialization: log(1/temperature) where temperature=0.07 for CLIP
        # So log(1/0.07) = log(14.2857)  2.66
        if init_logit_scale <= 0:
            init_value = math.log(1.0 / 0.07)  # CLIP default
        else:
            init_value = math.log(1.0 / init_logit_scale) if init_logit_scale < 1.0 else math.log(init_logit_scale)
        
        self.logit_scale = nn.Parameter(torch.tensor([init_value]), requires_grad=True)
        
        # Temperature bounds
        self.logit_scale_min = math.log(1.0 / temp_max)  # Minimum logit scale (maximum temperature)
        self.logit_scale_max = math.log(1.0 / temp_min)  # Maximum logit scale (minimum temperature)
        
        print(f"[INFO] Initialized logit scale to: {init_value:.4f} (temperature: {1.0/torch.exp(torch.tensor([init_value])).item():.4f})")
        print(f"[INFO] Temperature bounds: [{temp_min:.4f}, {temp_max:.4f}]")

    def forward(self, images, input_ids, attention_mask):
        img_emb = self.image_encoder(images)
        txt_emb = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Clamp logit scale to prevent numerical instability
        logit_scale = torch.clamp(self.logit_scale, 
                                  min=self.logit_scale_min, 
                                  max=self.logit_scale_max).exp()
        
        # Temperature = 1 / logit_scale
        temperature = 1.0 / logit_scale
        
        logits_per_image = logit_scale * (img_emb @ txt_emb.t())
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text, temperature

# -------------------------
# FIXED Loss function with temperature regularization
# -------------------------
class StableClipLoss(nn.Module):
    def __init__(self, debug=False, temperature_reg_weight=0.001, target_temp=0.07):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.debug = debug
        self.temperature_reg_weight = temperature_reg_weight
        self.target_temp = target_temp

    def forward(self, logits_per_image, logits_per_text, temperature=None):
        batch_size = logits_per_image.size(0)
        device = logits_per_image.device
        
        if self.debug:
            print(f"[DEBUG] Temperature: {temperature.item() if temperature is not None else 'N/A'}")
            print(f"[DEBUG] Logits range: [{logits_per_image.min():.2f}, {logits_per_image.max():.2f}]")
            print(f"[DEBUG] Logits mean: {logits_per_image.mean():.2f} +- {logits_per_image.std():.2f}")
        
        # Create targets (diagonal should be matching pairs)
        targets = torch.arange(batch_size, device=device)
        
        # Normalize logits to prevent numerical issues
        logits_per_image = logits_per_image - logits_per_image.max(dim=1, keepdim=True).values
        logits_per_text = logits_per_text - logits_per_text.max(dim=1, keepdim=True).values
        
        loss_i = self.cross_entropy(logits_per_image, targets)
        loss_t = self.cross_entropy(logits_per_text, targets)
        
        contrastive_loss = (loss_i + loss_t) / 2.0
        
        # Temperature regularization to prevent explosion
        temp_reg_loss = 0.0
        if temperature is not None and self.temperature_reg_weight > 0:
            temp_reg_loss = self.temperature_reg_weight * F.mse_loss(temperature, 
                                                                     torch.tensor([self.target_temp], device=device))
        
        total_loss = contrastive_loss + temp_reg_loss
        
        if self.debug:
            # Check if diagonal has highest values (should be for correct learning)
            diag_vals = logits_per_image.diag()
            off_diag_mask = ~torch.eye(batch_size, dtype=bool, device=device)
            off_diag_vals = logits_per_image[off_diag_mask].view(batch_size, batch_size-1)
            
            print(f"[DEBUG] Diagonal mean: {diag_vals.mean():.4f}")
            print(f"[DEBUG] Off-diagonal mean: {off_diag_vals.mean():.4f}")
            is_diag_higher = (diag_vals > off_diag_vals.max(dim=1).values).float().mean()
            print(f"[DEBUG] Is diagonal highest? {is_diag_higher.item():.1%}")
            print(f"[DEBUG] Contrastive loss: {contrastive_loss.item():.4f}")
            print(f"[DEBUG] Temp reg loss: {temp_reg_loss.item():.4f}")
            print(f"[DEBUG] Total loss: {total_loss.item():.4f}")
        
        return total_loss

# -------------------------
# FIXED Retrieval function
# -------------------------
@torch.no_grad()
def proper_retrieval(img_embs, txt_embs, targets_img, ks=(1,5,10)):
    """
    Proper retrieval evaluation for CLIP-style models
    img_embs: [N_img, D] normalized image embeddings
    txt_embs: [N_txt, D] normalized text embeddings
    targets_img: ground truth text indices for each image
    ks: tuple of k values for R@k
    """
    # Compute similarity matrix
    sim_matrix = img_embs @ txt_embs.t()  # [N_img, N_txt]
    
    # Image->Text retrieval
    itopk = {}
    for k in ks:
        if k > txt_embs.size(0):
            continue
        _, topk_indices = sim_matrix.topk(k=k, dim=1)  # [N_img, k]
        
        # Check if correct text is in top-k
        correct = (topk_indices == targets_img.unsqueeze(1)).any(dim=1)
        recall = correct.float().mean().item()
        itopk[f"R@{k}"] = recall
    
    # Text->Image retrieval (symmetric)
    ttopk = {}
    sim_matrix_t = sim_matrix.t()  # [N_txt, N_img]
    for k in ks:
        if k > img_embs.size(0):
            continue
        _, topk_indices = sim_matrix_t.topk(k=k, dim=1)
        
        # For text->image, we need to find which image corresponds to each text
        # In a balanced batch, each text should retrieve its corresponding image
        targets_txt = torch.arange(txt_embs.size(0), device=img_embs.device)
        if txt_embs.size(0) == img_embs.size(0):
            # Perfect 1:1 mapping (batch training case)
            correct = (topk_indices == targets_txt.unsqueeze(1)).any(dim=1)
        else:
            # For class embeddings case, we need to map text index to correct images
            # Each text (class) should retrieve all images of that class
            correct = []
            for txt_idx in range(txt_embs.size(0)):
                img_indices = (targets_img == txt_idx).nonzero(as_tuple=True)[0]
                if len(img_indices) > 0:
                    hits = (topk_indices[txt_idx].unsqueeze(0) == img_indices.unsqueeze(1)).any()
                    correct.append(hits)
                else:
                    correct.append(torch.tensor(False, device=img_embs.device))
            correct = torch.stack(correct) if correct else torch.tensor([], device=img_embs.device)
        
        recall = correct.float().mean().item() if len(correct) > 0 else 0.0
        ttopk[f"R@{k}"] = recall
    
    return itopk, ttopk, sim_matrix

# -------------------------
# Dataset wrapper for CLIP
# -------------------------
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

# -------------------------
# Debug functions
# -------------------------
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
                print(f"  Temperature param - grad: {param_norm:.6f}, value: {param.item():.4f}")
        elif param.requires_grad:
            zero_grad_count += 1
            if 'logit_scale' in name:
                print(f"  NO GRADIENT but requires_grad=True: {name}")
    
    if total_params > 0:
        total_norm = total_norm ** 0.5
        print(f"  Total gradient norm: {total_norm:.4f}")
        print(f"  Parameters with zero gradient: {zero_grad_count}/{total_params} ({zero_grad_count/total_params*100:.1f}%)")

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
        print(f"Image embeddings shape: {img_emb.shape}")
        print(f"Text embeddings shape: {txt_emb.shape}")
        print(f"Image embeddings norm: mean={img_emb.norm(dim=1).mean():.4f}, std={img_emb.norm(dim=1).std():.4f}")
        print(f"Text embeddings norm: mean={txt_emb.norm(dim=1).mean():.4f}, std={txt_emb.norm(dim=1).std():.4f}")
        
        # Check similarity
        sims = img_emb @ txt_emb.t()
        print(f"Similarity diagonal (correct pairs): mean={sims.diag().mean():.4f}, min={sims.diag().min():.4f}, max={sims.diag().max():.4f}")
        print(f"Similarity off-diagonal: mean={sims[~torch.eye(sims.size(0), dtype=bool, device=device)].mean():.4f}")
        
        # Check if diagonal is higher (it should be for contrastive learning to work)
        diag_mean = sims.diag().mean().item()
        off_diag_mean = sims[~torch.eye(sims.size(0), dtype=bool, device=device)].mean().item()
        print(f"Diagonal vs Off-diagonal difference: {diag_mean - off_diag_mean:.4f} (should be positive)")
        
        # Get text embeddings for all classes
        class_enc = tokenizer(prompts, padding='max_length', truncation=True, 
                             max_length=max_seq_len, return_tensors='pt')
        class_input_ids = class_enc['input_ids'].to(device)
        class_attns = class_enc['attention_mask'].to(device)
        class_txt_embs = model.text_encoder(input_ids=class_input_ids, attention_mask=class_attns)
        
        # Check class embedding norms
        print(f"Class embeddings norm: mean={class_txt_embs.norm(dim=1).mean():.4f}, std={class_txt_embs.norm(dim=1).std():.4f}")
        
        # Temperature
        logits_i, logits_t, temperature = model(images[:2], input_ids[:2], attn_mask[:2])
        print(f"Temperature: {temperature.item():.4f}")
        print(f"Logit scale parameter: {model.logit_scale.item():.4f}")
    
    model.train()

# -------------------------
# Visualization function for similarity matrix
# -------------------------
def visualize_similarity_matrix(sim_matrix, epoch, save_dir, title="Similarity Matrix"):
    """Visualize the similarity matrix to see structure"""
    try:
        import matplotlib.pyplot as plt
        
        sim_np = sim_matrix.cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(sim_np, cmap='viridis', aspect='auto', vmin=sim_np.min(), vmax=sim_np.max())
        plt.colorbar(label='Similarity')
        plt.title(f'{title} (Epoch {epoch})')
        plt.xlabel('Text Embeddings')
        plt.ylabel('Image Embeddings')
        
        # Add diagonal line
        min_dim = min(sim_np.shape[0], sim_np.shape[1])
        plt.plot([0, min_dim-1], [0, min_dim-1], 'r--', alpha=0.5, label='Diagonal')
        plt.legend()
        
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'similarity_epoch_{epoch:03d}.png'), dpi=150)
        plt.close()
        
        # Also check diagonal vs off-diagonal stats
        diag = np.diag(sim_np[:min_dim, :min_dim])
        mask = ~np.eye(min_dim, dtype=bool)
        off_diag = sim_np[:min_dim, :min_dim][mask]
        
        print(f"[VIS] {title}:")
        print(f"      Diagonal mean: {diag.mean():.4f} +- {diag.std():.4f}")
        print(f"      Off-diagonal mean: {off_diag.mean():.4f} +- {off_diag.std():.4f}")
        print(f"      Diagonal > Off-diagonal: {diag.mean() > off_diag.mean()}")
        print(f"      Diagonal min/max: {diag.min():.4f}/{diag.max():.4f}")
        
    except ImportError:
        print("[WARNING] matplotlib not installed, skipping visualization")

# -------------------------
# Custom scheduler with warmup
# -------------------------
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

# -------------------------
# Main execution
# -------------------------

print(f"[INFO] Using text model: {args.text_model}")

# Create tokenizer using factory
tokenizer = TextEncoderFactory.create_tokenizer(args.text_model)
print(f"[INFO] Loaded {args.text_model} tokenizer")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

max_seq_len = args.max_len

# -------------------------
# Load datasets from torchvision
# -------------------------
dataset_name = args.source_dataset
dataset_name1 = args.target_dataset
print(f"\n[INFO] Loading {dataset_name} dataset from torchvision")

# Get dataset configuration FIRST
source_config = get_dataset_config(source_dataset)
target_config = get_dataset_config(target_dataset)

# Load all splits
# For standard training (source == target)
if not domain_adaptation or source_dataset == target_dataset:
    # Use source transforms for everything
    train_dataset_raw, loaded_class_names = load_dataset(source_dataset, split='train', transform=train_transform, val_split=val_split)
    val_dataset_raw, _ = load_dataset(source_dataset, split='val', transform=transform_test_source, val_split=val_split)
    test_dataset_raw, _ = load_dataset(source_dataset, split='test', transform=transform_test_source, val_split=val_split)
else:
    # Domain adaptation: train on source, validate/test on target
    train_dataset_raw, loaded_class_names = load_dataset(source_dataset, split='train', transform=train_transform, val_split=val_split)
    val_dataset_raw, _ = load_dataset(target_dataset, split='val', transform=transform_test_source, val_split=val_split)  # Validate on source val
    test_dataset_raw, _ = load_dataset(target_dataset, split='test', transform=transform_test_target, val_split=val_split)  # Test on target

# Use dataset_config class names if available, otherwise use loaded names
if source_config['class_names'] is not None:
    class_names = source_config['class_names']
else:
    class_names = loaded_class_names

nb_class = len(class_names)

print(f"[INFO] {dataset_name} has {nb_class} classes:")
print(f"  Classes (first 10): {class_names[:10]}...")

# -------------------------
# Create prompts for the dataset
# -------------------------
prompt_templates = [
    "a photo of {}",
    "a picture of {}",
    "an image of {}",
    "{} in a photo",
    "a photograph of {}",
    "{}",
    "a high quality image of {}",
    "a clear photo of {}",
]

# Create prompts for all classes
prompts = [prompt_templates[0].format(name.replace("_", " ").replace("-", " ")) for name in class_names]
print(f"[INFO] Created {len(prompts)} prompts for {dataset_name}")
print(f"  Example prompts: {prompts[:5]}")

print(f"\n[INFO] Dataset sizes:")
print(f"  Train: {len(train_dataset_raw)} images")
print(f"  Val:   {len(val_dataset_raw)} images")
print(f"  Test:  {len(test_dataset_raw)} images")

# Wrap datasets with CLIP
train_ds = CLIPWrappedDataset(train_dataset_raw, prompts, tokenizer, max_len=max_seq_len)
val_ds = CLIPWrappedDataset(val_dataset_raw, prompts, tokenizer, max_len=max_seq_len)
test_ds = CLIPWrappedDataset(test_dataset_raw, prompts, tokenizer, max_len=max_seq_len)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# Build model with selected text encoder
backbone = model_zoo[cnn](pretrained=True)
proj_dim = 512
image_encoder = ImageEncoder(backbone, cnn, proj_dim, image_size=source_config['image_size'])

# Use Text Encoder Factory to create the text encoder
text_encoder = TextEncoderFactory.create_text_encoder(
    model_name=args.text_model,
    proj_dim=proj_dim
)

# Use FIXED CLIP model with proper temperature control
model = CLIPModel(image_encoder, text_encoder, 
                  init_logit_scale=args.init_temperature,
                  temp_min=args.temperature_clip_min,
                  temp_max=args.temperature_clip_max)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create separate parameter groups with different learning rates
# Create learning rate groups based on model type
if args.text_model in ["deberta", "flan_t5"]:
    # These models need lower learning rates for the LLM part
    param_groups = [
        {'params': model.image_encoder.parameters(), 'lr': lr_begin, 'name': 'image_encoder'},
        {'params': [p for n, p in model.text_encoder.named_parameters() if 'proj' in n or 'ln' in n], 'lr': lr_begin, 'name': 'text_proj'},
        {'params': [p for n, p in model.text_encoder.named_parameters() if 'proj' not in n and 'ln' not in n], 'lr': lr_begin * 0.01, 'name': 'text_backbone'},
        {'params': [model.logit_scale], 'lr': lr_begin * 0.1, 'name': 'temperature'},
    ]
else:
    # Original for other models
    param_groups = [
        {'params': model.image_encoder.parameters(), 'lr': lr_begin, 'name': 'image_encoder'},
        {'params': text_encoder.parameters(), 'lr': lr_begin * 0.1, 'name': 'text_encoder'},
        {'params': [model.logit_scale], 'lr': lr_begin * 0.1, 'name': 'temperature'},
        ]
optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

# Use custom scheduler with warmup
scheduler = WarmupCosineSchedule(
    optimizer=optimizer,
    warmup_epochs=args.warmup_epochs,
    total_epochs=args.epoch,
    min_lr_ratio=0.01
)

if args.ckpt:
    print("Loading checkpoint:", args.ckpt)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck['model'])
    if 'optimizer' in ck:
        optimizer.load_state_dict(ck['optimizer'])
    # Check if model matches
    if 'text_model' in ck and ck['text_model'] != args.text_model:
        print(f"[WARNING] Checkpoint was trained with {ck['text_model']}, but current model is {args.text_model}")

if args.requires_grad == 'fc':
    for name, p in model.named_parameters():
        if ("proj" in name) or ("text_encoder" in name) or ("ln" in name):
            p.requires_grad = True
        else:
            p.requires_grad = False
elif args.requires_grad == 'all':
    for p in model.parameters():
        p.requires_grad = True

if use_amp == 2:
    print("Using torch.cuda.amp")
    from torch.amp import GradScaler, autocast
    scaler = GradScaler('cuda')
else:
    scaler = None

if len(args.gpu) > 1:
    model = torch.nn.DataParallel(model)

# Use FIXED loss function
criterion = StableClipLoss(
    debug=args.debug_gradients,
    temperature_reg_weight=args.temperature_reg_weight,
    target_temp=args.init_temperature
)

# Archive scripts
if os.path.exists("train_clip_deepseek_cifar100_original.py"):
    shutil.copyfile("train_clip_deepseek_cifar100_original.py", os.path.join(exp_dir, "train_clip_deepseek_cifar100_original.py"))

# Create log file
log_file = open(os.path.join(exp_dir, "training_log.csv"), "w")
log_file.write("epoch,train_loss,train_acc,train_f1,val_acc,val_f1,temperature,lr,image_r1,text_r1\n")

min_train_loss = float('inf')
max_val_score = 0.0

print("\n" + "="*60)
print(f"Starting training on {dataset_name} Dataset")
print(f"Using CNN: {cnn}, Batch size: {batch_size}, Epochs: {num_epoch}")
print(f"Learning rate: {lr_begin}, Warmup epochs: {args.warmup_epochs}")
print(f"Accumulation steps: {accumulation_steps}")
print(f"Validation split: {val_split}")
print(f"Initial temperature: {args.init_temperature}")
print(f"Temperature bounds: [{args.temperature_clip_min}, {args.temperature_clip_max}]")
print("="*60 + "\n")

# Pre-compute class text embeddings for evaluation
print("[INFO] Pre-computing class text embeddings...")
with torch.no_grad():
    class_enc = tokenizer(
        prompts,
        padding='max_length',
        truncation=True,
        max_length=max_seq_len,
        return_tensors='pt'
    )
    class_input_ids = class_enc['input_ids'].to(device)
    class_attns = class_enc['attention_mask'].to(device)

# -------------------------
# Training loop with metrics (IMPROVED)
# -------------------------
for epoch in range(begin_epoch, num_epoch+1):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch:03d}/{num_epoch:03d}")
    print(f"{'='*60}")
    
    # Debug embeddings every 5 epochs
    if epoch % 5 == 1 or epoch == begin_epoch:
        debug_embeddings(model, train_loader, prompts, tokenizer, max_seq_len, device, epoch)
    
    # Training phase
    model.train()
    running_loss = 0.0
    n_samples = 0
    
    # Lists to store predictions during training
    train_preds = []
    train_targets = []
    
    optimizer.zero_grad()
    
    for batch_idx, (images, input_ids, attn_mask, targets, paths) in enumerate(tqdm(train_loader, ncols=80, desc="Training")):
        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        
        if use_amp == 2:
            with autocast('cuda'):
                logits_i, logits_t, temperature = model(images, input_ids, attn_mask)
                loss = criterion(logits_i, logits_t, temperature) / accumulation_steps
            scaler.scale(loss).backward()
        else:
            logits_i, logits_t, temperature = model(images, input_ids, attn_mask)
            loss = criterion(logits_i, logits_t, temperature) / accumulation_steps
            loss.backward()
        
        running_loss += loss.item() * images.size(0) * accumulation_steps
        n_samples += images.size(0)
        
        # Calculate training accuracy
        with torch.no_grad():
            # Get image embeddings
            img_emb = model.image_encoder(images)
            img_emb = F.normalize(img_emb, dim=-1)
            
            # Get text embeddings for all classes
            class_txt_embs = model.text_encoder(input_ids=class_input_ids, attention_mask=class_attns)
            class_txt_embs = F.normalize(class_txt_embs, dim=-1)
            
            # Calculate similarity and predictions
            sims = img_emb @ class_txt_embs.t()
            preds = sims.argmax(dim=1).cpu().numpy()
            
            train_preds.extend(preds.tolist())
            train_targets.extend(targets.numpy().tolist())
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp == 2:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Print progress every 50 batches
        if batch_idx % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            # Calculate batch accuracy
            batch_acc = 100.0 * accuracy_score(targets.numpy(), preds)
            print(f"  Batch {batch_idx:04d}/{len(train_loader):04d} | Loss: {loss.item()*accumulation_steps:.4f} | Acc: {batch_acc:.2f}% | Temp: {temperature.item():.4f} | LR: {current_lr:.6f}")
            
            # Visualize similarity matrix for debugging
            if args.visualize_similarity and batch_idx == 0:
                with torch.no_grad():
                    # Get image and text embeddings for this batch
                    img_emb_batch = model.image_encoder(images)
                    txt_emb_batch = model.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
                    sim_matrix_batch = img_emb_batch @ txt_emb_batch.t()
                    visualize_similarity_matrix(sim_matrix_batch, epoch, exp_dir, title="Batch Similarity")

    # Step scheduler
    scheduler.step(epoch)
    
    # Check gradients
    check_gradients(model, epoch, debug=args.debug_gradients)

    # Calculate training metrics
    train_loss = running_loss / max(1, n_samples)
    train_acc = 100.0 * accuracy_score(train_targets, train_preds)
    train_f1 = 100.0 * f1_score(train_targets, train_preds, average='weighted', zero_division=0)
    
    print(f"\nTraining Results:")
    print(f"  Loss:       {train_loss:.4f}")
    print(f"  Accuracy:   {train_acc:.2f}%")
    print(f"  F1 Score:   {train_f1:.2f}%")
    print(f"  Temperature: {temperature.item():.4f}")
    
    # Validation phase
    if (epoch % args.evaluate) == 0 or epoch == num_epoch:
        model.eval()
        with torch.no_grad():
            class_txt_embs = model.text_encoder(input_ids=class_input_ids, attention_mask=class_attns)
            class_txt_embs = F.normalize(class_txt_embs, dim=-1)

            val_imgemb_list = []
            val_targets = []
            val_preds = []
            
            for images, input_ids, attn_mask, targets, paths in tqdm(val_loader, ncols=80, desc="Validation"):
                if chex == 1 and images.dim() == 5:
                    bs, n_crops, c, h, w = images.size()
                    images = images.view(-1, c, h, w).to(device)
                    img_emb = model.image_encoder(images)
                    img_emb = img_emb.view(bs, n_crops, -1).mean(dim=1)
                else:
                    images = images.to(device)
                    img_emb = model.image_encoder(images)
                
                img_emb = F.normalize(img_emb, dim=-1)
                sims = img_emb @ class_txt_embs.t()
                preds = sims.argmax(dim=1).cpu().numpy()
                
                val_imgemb_list.append(img_emb)
                val_targets.extend(targets.numpy().tolist())
                val_preds.extend(preds.tolist())

            all_img_embs = torch.cat(val_imgemb_list, dim=0)
            val_targets_tensor = torch.tensor(val_targets, device=device)
            
            # Use FIXED retrieval function
            if all_img_embs.size(0) > 0 and class_txt_embs.size(0) > 0:
                itopk, ttopk, sim_matrix = proper_retrieval(
                    all_img_embs, class_txt_embs, val_targets_tensor, ks=(1,5,10)
                )
                
                # Calculate validation metrics
                val_acc = 100.0 * accuracy_score(val_targets, val_preds)
                val_f1 = 100.0 * f1_score(val_targets, val_preds, average='weighted', zero_division=0)
                val_prec = 100.0 * precision_score(val_targets, val_preds, average='weighted', zero_division=0)
                val_rec = 100.0 * recall_score(val_targets, val_preds, average='weighted', zero_division=0)
                
                print(f"\nValidation Results:")
                print(f"  Accuracy:  {val_acc:.2f}%")
                print(f"  F1 Score:  {val_f1:.2f}%")
                print(f"  Precision: {val_prec:.2f}%")
                print(f"  Recall:    {val_rec:.2f}%")
                print(f"  Image->Text R@1: {itopk.get('R@1', 0):.4f}")
                print(f"  Image->Text R@5: {itopk.get('R@5', 0):.4f}")  # ADD THIS LINE
                print(f"  Text->Image R@1: {ttopk.get('R@1', 0):.4f}")
                print(f"  Text->Image R@5: {ttopk.get('R@5', 0):.4f}")  # ADD THIS LINE
                
                # Visualize similarity matrix
                if args.visualize_similarity and (epoch % 10 == 0 or epoch == num_epoch):
                    visualize_similarity_matrix(sim_matrix, epoch, exp_dir, title="Validation Similarity")

                # Save best model based on validation F1 score
                if val_f1 > max_val_score and epoch >= args.warmup_epochs:  # Wait at least warmup epochs
                    max_val_score = val_f1
                    for old_ckpt in glob(os.path.join(exp_dir, "max_f1_*.pth")):
                        try:
                            os.remove(old_ckpt)
                        except OSError:
                            pass
                    ckpt_path = os.path.join(exp_dir, f"max_f1_{max_val_score:.2f}_epoch{epoch}.pth")
                    
                    # Save model state
                    if isinstance(model, torch.nn.DataParallel):
                        model_state = model.module.state_dict()
                    else:
                        model_state = model.state_dict()
                    
                    torch.save({
                        'epoch': epoch,
                        'model': model_state,
                        'optimizer': optimizer.state_dict(),
                        'val_f1': max_val_score,
                        'val_acc': val_acc,
                        'train_acc': train_acc,
                        'train_f1': train_f1,
                        'temperature': temperature.item(),
                        'text_model': args.text_model,  # Save which model was used
                        'args': vars(args)
                    }, ckpt_path)
                    print(f"  ? Saved best checkpoint to {ckpt_path}")

    # Log to file
    current_lr = optimizer.param_groups[0]['lr']
    image_r1 = itopk.get('R@1', 0) if 'itopk' in locals() else 0.0
    text_r1 = ttopk.get('R@1', 0) if 'ttopk' in locals() else 0.0
    log_file.write(f"{epoch},{train_loss:.4f},{train_acc:.2f},{train_f1:.2f},{val_acc:.2f},{val_f1:.2f},{temperature.item():.4f},{current_lr:.6f},{image_r1:.4f},{text_r1:.4f}\n")
    log_file.flush()

    # Print epoch summary
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%, F1={train_f1:.2f}%")
    print(f"  Val:   Acc={val_acc:.2f}%, F1={val_f1:.2f}%")
    print(f"  Retr:  Image->Text R@1={image_r1:.4f}, Text->Image R@1={text_r1:.4f}")
    print(f"  Temp:  {temperature.item():.4f}")
    print(f"  LR:    {current_lr:.6f}")

log_file.close()

print("\n" + "="*60)
print("Training completed!")
print(f"Best validation F1: {max_val_score:.2f}%")
print("="*60 + "\n")

# -------------------------
# After training - TESTING (with proper retrieval)
# -------------------------
print("\n\n===== FINAL TESTING FOR PAPER RESULTS =====")
best_ckpts = sorted(glob(join(exp_dir, "max_f1_*.pth")))
if len(best_ckpts) > 0:
    ck = torch.load(best_ckpts[-1], map_location=device)
    model.load_state_dict(ck['model'])
    print(f"Loaded best ckpt: {best_ckpts[-1]} (Val F1: {ck.get('val_f1', 0):.2f}%)")
else:
    print("No checkpoint found, using final model")
    ck = None  # Initialize ck variable

model.eval()

with torch.no_grad():
    # Pre-compute class text embeddings
    class_txt_embs = model.text_encoder(input_ids=class_input_ids, attention_mask=class_attns)
    class_txt_embs = F.normalize(class_txt_embs, dim=-1)

    # Test on validation set (for reference only)
    print("\n=== VALIDATION SET (Model Selection) ===")
    val_targets, val_preds, val_imgembs = [], [], []
    for batch in tqdm(val_loader, ncols=80, desc="Validation"):
        try:
            images, input_ids, attn_mask, targets, paths = batch
        except ValueError:
            if len(batch) == 3:
                images, targets, paths = batch
            else:
                raise
        
        images = images.to(device)
        
        # Handle different image dimensions
        if images.dim() == 5 and chex == 1:
            bs, n_crops, c, h, w = images.size()
            images = images.view(-1, c, h, w)
            img_emb = model.image_encoder(images)
            img_emb = img_emb.view(bs, n_crops, -1).mean(dim=1)
        else:
            # Resize if needed
            if images.size(2) != source_config['image_size'] or images.size(3) != source_config['image_size']:
                images = F.interpolate(images, size=(source_config['image_size'], source_config['image_size']), 
                                      mode='bilinear', align_corners=False)
            img_emb = model.image_encoder(images)
        
        img_emb = F.normalize(img_emb, dim=-1)
        sims = img_emb @ class_txt_embs.t()
        preds = sims.argmax(dim=1).cpu().numpy()
        
        val_imgembs.append(img_emb)
        val_targets.extend(targets.numpy().tolist())
        val_preds.extend(preds.tolist())

    if len(val_imgembs) > 0:
        all_val_imgembs = torch.cat(val_imgembs, dim=0)
        val_targets_tensor = torch.tensor(val_targets, device=device)
        itopk_val, ttopk_val, _ = proper_retrieval(
            all_val_imgembs, class_txt_embs, val_targets_tensor, ks=(1,5)
        )
        
        val_acc = 100.0 * accuracy_score(val_targets, val_preds)
        val_f1 = 100.0 * f1_score(val_targets, val_preds, average='weighted', zero_division=0)
        print(f"Val Accuracy: {val_acc:.2f}%")
        print(f"Val F1: {val_f1:.2f}%")
        print(f"Val R@1: {itopk_val.get('R@1', 0):.4f}, R@5: {itopk_val.get('R@5', 0):.4f}")

    # Test on test set (FOR PAPER RESULTS)
    print("\n" + "="*60)
    print("TEST SET (FINAL PAPER RESULTS)")
    print("="*60)
    print(f"\nDATASET CONFIGURATION:")
    print(f"  Source Dataset: {source_dataset}")
    print(f"  Target Dataset: {target_dataset}")
    print(f"  Domain Adaptation: {'ENABLED' if domain_adaptation else 'DISABLED'}")
    print(f"  Number of Classes: {nb_class}")
    print("-" * 60)
    test_targets, test_preds, test_imgembs = [], [], []
    test_paths, test_scores = [], []
    
    for batch in tqdm(test_loader, ncols=80, desc="Test Set"):
        try:
            images, input_ids, attn_mask, targets, paths = batch
        except ValueError:
            if len(batch) == 3:
                images, targets, paths = batch
            else:
                raise
        
        images = images.to(device)
        
        # Handle different image dimensions
        if images.dim() == 5 and chex == 1:
            bs, n_crops, c, h, w = images.size()
            images = images.view(-1, c, h, w)
            img_emb = model.image_encoder(images)
            img_emb = img_emb.view(bs, n_crops, -1).mean(dim=1)
        else:
            # Resize if needed
            test_image_size = target_config['image_size'] if domain_adaptation else source_config['image_size']
            if images.size(2) != test_image_size or images.size(3) != test_image_size:
                images = F.interpolate(images, size=(test_image_size, test_image_size), 
                                      mode='bilinear', align_corners=False)
            img_emb = model.image_encoder(images)
        
        img_emb = F.normalize(img_emb, dim=-1)
        sims = img_emb @ class_txt_embs.t()
        preds = sims.argmax(dim=1).cpu().numpy()
        scores = softmax(sims.cpu().numpy(), axis=1).tolist()
        
        test_imgembs.append(img_emb)
        test_targets.extend(targets.numpy().tolist())
        test_preds.extend(preds.tolist())
        test_scores.extend(scores)
        test_paths.extend(paths)

    if len(test_imgembs) > 0:
        all_test_imgembs = torch.cat(test_imgembs, dim=0)
        test_targets_tensor = torch.tensor(test_targets, device=device)
        itopk_test, ttopk_test, test_sim_matrix = proper_retrieval(
            all_test_imgembs, class_txt_embs, test_targets_tensor, ks=(1,5)
        )
        
        test_acc = 100.0 * accuracy_score(test_targets, test_preds)
        test_f1 = 100.0 * f1_score(test_targets, test_preds, average='weighted', zero_division=0)
        test_prec = 100.0 * precision_score(test_targets, test_preds, average='weighted', zero_division=0)
        test_rec = 100.0 * recall_score(test_targets, test_preds, average='weighted', zero_division=0)
        
        print(f"\n" + "="*60)
        print("FINAL RESULTS FOR PAPER")
        print("="*60)
        
        if domain_adaptation:
            print(f"Domain Adaptation: {source_dataset} ? {target_dataset}")
        else:
            print(f"Standard Training on: {source_dataset}")
        
        print(f"Model: {cnn} with DeepSeek text encoder")
        print(f"\nCLASSIFICATION METRICS:")
        print(f"  Accuracy:  {test_acc:.2f}%")
        print(f"  F1 Score:  {test_f1:.2f}%")
        print(f"  Precision: {test_prec:.2f}%")
        print(f"  Recall:    {test_rec:.2f}%")
        
        print(f"\nRETRIEVAL METRICS:")
        print(f"  Image?Text R@1: {itopk_test.get('R@1', 0):.4f}")
        print(f"  Image?Text R@5: {itopk_test.get('R@5', 0):.4f}")
        print(f"  Text?Image R@1: {ttopk_test.get('R@1', 0):.4f}")
        print(f"  Text?Image R@5: {ttopk_test.get('R@5', 0):.4f}")
        
        # Confusion matrix info
        cm = confusion_matrix(test_targets, test_preds)
        print(f"\nCONFUSION MATRIX:")
        print(f"  Shape: {cm.shape}")
        print(f"  Total test samples: {len(test_targets)}")
        
        # Visualize similarity matrix for test set
        if args.visualize_similarity:
            visualize_similarity_matrix(test_sim_matrix, "final_test", exp_dir, title="Test Set Similarity Matrix")

# Save detailed results for paper
results = {
    'paper_results': {
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'image_to_text_r1': itopk_test.get('R@1', 0),
        'image_to_text_r5': itopk_test.get('R@5', 0),
        'text_to_image_r1': ttopk_test.get('R@1', 0),
        'text_to_image_r5': ttopk_test.get('R@5', 0),
    },
    'validation_results': {
        'val_accuracy': val_acc if 'val_acc' in locals() else 0,
        'val_f1': val_f1 if 'val_f1' in locals() else 0,
    },
    'metadata': {
        'source_dataset': source_dataset,
        'target_dataset': target_dataset,
        'domain_adaptation': domain_adaptation,
        'cnn_backbone': cnn,
        'text_encoder': 'DeepSeek-LLM',
        'num_classes': nb_class,
        'best_epoch': ck.get('epoch', 0) if ck is not None else 0,
        'temperature': ck.get('temperature', 0) if ck is not None else 0,
    }
}

pickle.dump(results, open(join(exp_dir, "paper_results.pkl"), "wb"))

# Create a clean text file with just the paper results
with open(join(exp_dir, "paper_results.txt"), "w") as f:
    f.write("="*60 + "\n")
    f.write("RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Experiment: {args.save_dir}\n")
    f.write(f"Source Dataset: {source_dataset}\n")
    f.write(f"Target Dataset: {target_dataset}\n")
    f.write(f"Domain Adaptation: {'Yes' if domain_adaptation else 'No'}\n")
    f.write(f"CNN Backbone: {cnn}\n")
    f.write(f"Text Encoder: DeepSeek-LLM\n")
    f.write(f"Number of Classes: {nb_class}\n\n")
    
    f.write("TEST SET RESULTS:\n")
    f.write("-" * 40 + "\n")
    f.write(f"Accuracy:  {test_acc:.2f}%\n")
    f.write(f"F1 Score:  {test_f1:.2f}%\n")
    f.write(f"Precision: {test_prec:.2f}%\n")
    f.write(f"Recall:    {test_rec:.2f}%\n\n")
    
    f.write("RETRIEVAL METRICS:\n")
    f.write("-" * 40 + "\n")
    f.write(f"Image?Text R@1: {itopk_test.get('R@1', 0):.4f}\n")
    f.write(f"Image?Text R@5: {itopk_test.get('R@5', 0):.4f}\n")
    f.write(f"Text?Image R@1: {ttopk_test.get('R@1', 0):.4f}\n")
    f.write(f"Text?Image R@5: {ttopk_test.get('R@5', 0):.4f}\n\n")
    
    f.write("METADATA:\n")
    f.write("-" * 40 + "\n")
    
    # Fixed: Separate conditional from format specifier
    best_epoch = ck.get('epoch', 0) if ck is not None else 0
    f.write(f"Best Epoch: {best_epoch}\n")
    
    temp_value = ck.get('temperature', 0) if ck is not None else 0
    f.write(f"Final Temperature: {temp_value:.4f}\n")
    
    param_count = sum(p.numel() for p in model.parameters())
    f.write(f"Total Parameters: {param_count:,}\n")

print(f"\n? Paper results saved to {exp_dir}/paper_results.txt")
print(f"? Detailed results saved to {exp_dir}/paper_results.pkl")
print("\n" + "="*60)
print("EXPERIMENT COMPLETE")
print("="*60)
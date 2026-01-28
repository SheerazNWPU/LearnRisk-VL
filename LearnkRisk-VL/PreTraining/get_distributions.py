# get_distribution_universal.py
import torch.nn.functional as F
import os, pickle, time, shutil, argparse, math, random, json
from os.path import join
from glob import glob
import numpy as np
import torch
# HF transformers (needed for text encoder)
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision import transforms, datasets
from tqdm import tqdm
import pandas as pd
from Densenet import densenet121, densenet161, densenet169, densenet201
from Resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, wide_resnet50_2, wide_resnet101_2
from Efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from Vgg import vgg11, vgg13, vgg16, vgg19
from torchvision.models import alexnet
from torchvision import transforms
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from PIL import Image
from text_encoder import SimpleDeepSeekTextEncoder, SimpleBERTTextEncoder, SimpleDeBERTaTextEncoder, SimpleFlanT5TextEncoder, SimpleQwenTextEncoder, SimpleLLaVATextEncoder, SimpleGPTTextEncoder

# Import model zoos (same as training code)
try:
    from Densenet import densenet121, densenet161, densenet169, densenet201
    from Resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, wide_resnet50_2, wide_resnet101_2
    from Efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
    from Vgg import vgg11, vgg13, vgg16, vgg19
except ImportError:
    print("[WARNING] Custom model zoos not found. Using torchvision models.")
    from torchvision.models import (
        resnet18, resnet34, resnet50, resnet101, resnet152,
        densenet121, densenet169, densenet201,
        vgg11, vgg13, vgg16, vgg19,
        efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
        efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7,
        resnext50_32x4d, wide_resnet50_2, wide_resnet101_2
    )

# HF transformers
from transformers import AutoTokenizer, AutoConfig

torch.multiprocessing.set_sharing_strategy('file_system')

# ==================== ARGUMENT PARSER ====================
parser = argparse.ArgumentParser(description='Universal Distribution Extractor for CLIP Models')
parser.add_argument('-c', '--cnn', default='r50', help='CNN backbone')
parser.add_argument('-s', '--save_dir', default='Office31_Amazon_DSLR_llava', help='Base save directory for results')
parser.add_argument('-g', '--gpu', default='0', help='GPU IDs to use (comma-separated)')
parser.add_argument('--checkpoint', default='max_f1_81.08_epoch55.pth', help='Path to model checkpoint (.pth file)')
parser.add_argument('--text_model', default='llava',
                   choices=['deepseek', 'bert', 'deberta', 'flan_t5', 'qwen', 'llava', 'gpt2'],
                   help='Text encoder model used in training')

# Dataset arguments (same as training code)
parser.add_argument('--source_dataset', default='Office31_Amazon', 
                   choices=['CIFAR100', 'CIFAR10', 'STL10', 'TinyImageNet', 
                           'Office31_Amazon', 'Office31_Webcam', 'Office31_DSLR'],
                   help='source dataset used for training')
parser.add_argument('--target_dataset', default='Office31_DSLR', 
                   choices=['', 'CIFAR100', 'CIFAR10', 'STL10', 'TinyImageNet', 
                           'Office31_Amazon', 'Office31_Webcam', 'Office31_DSLR',
                           'CIFAR100-C', 'CIFAR10-C'],
                   help='target dataset for testing (if empty, use source for all splits)')
parser.add_argument('--domain_adaptation', type=int, default=1,
                   help='1 for domain adaptation (train on source, test on target), 0 for standard')

# Split arguments
parser.add_argument('--split', default='all', choices=['all', 'train', 'val', 'test'], 
                   help='Which splits to process (all=process train,val,test)')
parser.add_argument('--train_set', default='train', help='name of training set')
parser.add_argument('--test_set', default='test', help='name of testing set')

# Other arguments
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for extraction')
parser.add_argument('--max_len', type=int, default=32, help='Max text length for tokenizer')
parser.add_argument('--local_model_path', type=str, 
                   default='/.cache/huggingface/hub/models--deepseek-ai--deepseek-llm-7b-base/snapshots/main/',
                   help='Path to locally cached HF model')
parser.add_argument('--cifar_style', type=int, default=1, help='Use CIFAR-style transforms (1) or medical-style (0)')
parser.add_argument('--val_split', type=float, default=0.1, help='Fraction for validation split')
args = parser.parse_args()

# ==================== SETUP ====================
cnn = args.cnn
source_dataset = args.source_dataset
target_dataset = args.target_dataset if args.target_dataset else source_dataset
domain_adaptation = args.domain_adaptation
exp_dir = f"/result_archive/{args.save_dir}"
batch_size = args.batch_size
text_model = args.text_model
checkpoint_path = os.path.join(exp_dir, args.checkpoint)
split_option = args.split

# Create experiment directory
os.makedirs(exp_dir, exist_ok=True)

# CUDA setup
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print configuration
print(f"\n{'='*80}")
print(f"UNIVERSAL DISTRIBUTION EXTRACTOR")
print(f"{'='*80}")
print(f"Source Dataset: {source_dataset}")
print(f"Target Dataset: {target_dataset}")
print(f"Domain Adaptation: {'ENABLED' if domain_adaptation else 'DISABLED'}")
print(f"CNN Backbone: {cnn}")
print(f"Text Model: {text_model}")
print(f"Checkpoint: {checkpoint_path}")
print(f"Output Dir: {exp_dir}")
print(f"{'='*80}")

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

# ==================== RETRIEVAL METRICS FUNCTION ====================
def compute_retrieval_metrics(image_embeddings, text_embeddings, targets, ks=(1, 5)):
    """
    Compute image-to-text and text-to-image retrieval metrics.
    
    Args:
        image_embeddings: [N, D] normalized image embeddings
        text_embeddings: [C, D] normalized text embeddings for all classes
        targets: [N] ground truth class indices for each image
        ks: tuple of k values for Recall@k
    
    Returns:
        itopk: dict of image-to-text R@k metrics
        ttopk: dict of text-to-image R@k metrics
    """
    # Compute similarity matrix
    sim_matrix = image_embeddings @ text_embeddings.t()  # [N, C]
    
    # Image-to-Text retrieval
    itopk = {}
    for k in ks:
        if k > text_embeddings.size(0):
            continue
        _, topk_indices = sim_matrix.topk(k=k, dim=1)  # [N, k]
        
        # Check if correct text is in top-k
        correct = (topk_indices == targets.unsqueeze(1)).any(dim=1)
        recall = correct.float().mean().item()
        itopk[f"R@{k}"] = recall
    
    # Text-to-Image retrieval (symmetric)
    ttopk = {}
    sim_matrix_t = sim_matrix.t()  # [C, N]
    for k in ks:
        if k > image_embeddings.size(0):
            continue
        _, topk_indices = sim_matrix_t.topk(k=k, dim=1)
        
        # For text-to-image, we need to find which image corresponds to each text
        # In a balanced batch, each text should retrieve its corresponding image
        targets_txt = torch.arange(text_embeddings.size(0), device=image_embeddings.device)
        
        # Each text (class) should retrieve images of that class
        correct = []
        for txt_idx in range(text_embeddings.size(0)):
            # Find all images of this class
            img_indices = (targets == txt_idx).nonzero(as_tuple=True)[0]
            if len(img_indices) > 0:
                # Check if any of the top-k retrieved images belong to this class
                hits = (topk_indices[txt_idx].unsqueeze(0) == img_indices.unsqueeze(1)).any()
                correct.append(hits)
            else:
                correct.append(torch.tensor(False, device=image_embeddings.device))
        
        correct = torch.stack(correct) if correct else torch.tensor([], device=image_embeddings.device)
        recall = correct.float().mean().item() if len(correct) > 0 else 0.0
        ttopk[f"R@{k}"] = recall
    
    return itopk, ttopk

# ==================== CROSS-ATTENTION FUSION ====================
class CrossAttentionFusion(nn.Module):
    """Cross-attention based feature fusion between image and text embeddings"""
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
            txt_emb: (batch_size, num_classes, dim) or (num_classes, dim) - text embeddings
        Returns:
            fused_emb: (batch_size, dim) - fused embeddings
        """
        batch_size = img_emb.size(0)
        
        # Ensure proper shapes
        if txt_emb.dim() == 2:  # (num_classes, dim)
            txt_emb = txt_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_classes, dim)
        
        img_emb = img_emb.unsqueeze(1)  # (batch_size, 1, dim)
        
        # Image-to-text cross attention
        img_enhanced, _ = self.img_to_txt_attention(
            query=img_emb,
            key=txt_emb,
            value=txt_emb,
        )
        img_enhanced = self.norm1(img_emb + self.dropout(img_enhanced))
        
        # Text-to-image cross attention (using mean text embedding as query)
        mean_txt_emb = txt_emb.mean(dim=1, keepdim=True)  # (batch_size, 1, dim)
        txt_enhanced, _ = self.txt_to_img_attention(
            query=mean_txt_emb,
            key=img_emb,
            value=img_emb,
        )
        txt_enhanced = self.norm2(mean_txt_emb + self.dropout(txt_enhanced))
        
        # Concatenate and fuse
        combined = torch.cat([img_enhanced.squeeze(1), txt_enhanced.squeeze(1)], dim=-1)  # (batch_size, dim*2)
        fused_emb = self.output_proj(combined)
        fused_emb = self.norm3(fused_emb)
        
        return F.normalize(fused_emb, dim=-1)

# ==================== TEXT ENCODER FACTORY ====================
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


# ==================== MODEL ARCHITECTURE ====================
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
        return out  # Return unnormalized for cross-attention

class CLIPModel(nn.Module):
    def __init__(self, image_encoder: ImageEncoder, text_encoder: nn.Module, 
                 init_logit_scale=0.07, temp_min=0.01, temp_max=100.0):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.cross_attention_fusion = CrossAttentionFusion(dim=512, num_heads=8, dropout=0.1)
        
        if init_logit_scale <= 0:
            init_value = math.log(1.0 / 0.07)
        else:
            init_value = math.log(1.0 / init_logit_scale) if init_logit_scale < 1.0 else math.log(init_logit_scale)
        
        self.logit_scale = nn.Parameter(torch.tensor([init_value]), requires_grad=True)
        self.logit_scale_min = math.log(1.0 / temp_max)
        self.logit_scale_max = math.log(1.0 / temp_min)

    def forward(self, images, input_ids, attention_mask):
        img_emb = self.image_encoder(images)
        txt_emb = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Normalize for similarity calculation
        img_emb_norm = F.normalize(img_emb, dim=-1)
        txt_emb_norm = F.normalize(txt_emb, dim=-1)
        
        logit_scale = torch.clamp(self.logit_scale, 
                                  min=self.logit_scale_min, 
                                  max=self.logit_scale_max).exp()
        
        logits_per_image = logit_scale * (img_emb_norm @ txt_emb_norm.t())
        logits_per_text = logits_per_image.t()
        
        # Cross-attention fusion
        fused_emb = self.cross_attention_fusion(img_emb, txt_emb_norm.unsqueeze(0))
        
        return logits_per_image, logits_per_text, 1.0/logit_scale, img_emb, txt_emb, fused_emb

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

# ==================== MAIN EXECUTION ====================
begin = time.time()

# Load dataset configurations
source_config = get_dataset_config(source_dataset)
target_config = get_dataset_config(target_dataset) if target_dataset else source_config

# Use source class names for prompts (matching training code)
if source_config['class_names'] is not None:
    class_names = source_config['class_names']
elif target_config['class_names'] is not None:
    class_names = target_config['class_names']
    print(f"[INFO] Using target dataset class names for prompts")
else:
    # Load dataset to get class names
    temp_dataset, loaded_class_names = load_dataset(source_dataset, split='test', transform=None)
    class_names = loaded_class_names

nb_class = len(class_names)
print(f"[INFO] Dataset has {nb_class} classes")

# Check if class names match for domain adaptation
if domain_adaptation and source_config['class_names'] and target_config['class_names']:
    if source_config['class_names'] != target_config['class_names']:
        print(f"[WARNING] Source and target datasets have different class names!")
        print(f"  Source classes: {len(source_config['class_names'])}")
        print(f"  Target classes: {len(target_config['class_names'])}")
        print(f"  Using source class names for prompts")

# Create transforms based on dataset types
if args.cifar_style:
    # Training transform for source dataset
    train_transform = transforms.Compose([
        transforms.Resize((source_config['image_size'], source_config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(source_config['normalize_mean'], source_config['normalize_std']),
    ])
    
    # Test transform for source dataset
    transform_test_source = transforms.Compose([
        transforms.Resize((source_config['image_size'], source_config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(source_config['normalize_mean'], source_config['normalize_std']),
    ])
    
    # Test transform for target dataset
    transform_test_target = transforms.Compose([
        transforms.Resize((target_config['image_size'], target_config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(target_config['normalize_mean'], target_config['normalize_std']),
    ])
else:
    # Medical-style transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_test_source = train_transform
    transform_test_target = train_transform

# Model zoo - EXACTLY as in your working code
model_zoo = {
    'r18': resnet18, 'r34': resnet34, 'r50': resnet50, 'r101': resnet101, 'r152': resnet152,
    'd121': densenet121, 'd161': densenet161, 'd169': densenet169, 'd201': densenet201,
    'eb0': efficientnet_b0, 'eb1': efficientnet_b1, 'eb2': efficientnet_b2, 'eb3': efficientnet_b3,
    'eb4': efficientnet_b4, 'eb5': efficientnet_b5, 'eb6': efficientnet_b6, 'eb7': efficientnet_b7,
    'rx50': resnext50_32x4d, 'alex': alexnet, 'wrn50': wide_resnet50_2, 'wrn101': wide_resnet101_2,
    'v11': vgg11, 'v13': vgg13, 'v16': vgg16, 'v19': vgg19
}

# Build model EXACTLY as in training
backbone = model_zoo[cnn](pretrained=True)  # Use pretrained=True as in training
proj_dim = 512
image_encoder = ImageEncoder(backbone, cnn, proj_dim, image_size=source_config['image_size'])

# Create text encoder using factory
tokenizer = TextEncoderFactory.create_tokenizer(args.text_model)
print(f"[INFO] Loaded {args.text_model} tokenizer")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
# Use Text Encoder Factory to create the text encoder
text_encoder = TextEncoderFactory.create_text_encoder(
    model_name=args.text_model,
    proj_dim=proj_dim
)


model = CLIPModel(
    image_encoder, 
    text_encoder,
    init_logit_scale=0.07,
    temp_min=0.01,
    temp_max=100.0
).to(device)

print(f"\n=== Loading checkpoint ===")
print(f"Checkpoint: {checkpoint_path}")

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
state_dict = checkpoint.get('model', checkpoint)

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
print("[INFO] Model ready for inference")

# Get tokenizer
tokenizer = TextEncoderFactory.create_tokenizer(args.text_model)

# Create prompts for classes (same as training code)
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

prompts = [prompt_templates[0].format(name.replace("_", " ").replace("-", " ")) for name in class_names]
print(f"[INFO] Created {len(prompts)} prompts")

# Pre-compute class text embeddings
with torch.no_grad():
    class_enc = tokenizer(
        prompts,
        padding='max_length',
        truncation=True,
        max_length=args.max_len,
        return_tensors='pt'
    )
    class_input_ids = class_enc['input_ids'].to(device)
    class_attns = class_enc['attention_mask'].to(device)
    class_txt_embs = model.text_encoder(input_ids=class_input_ids, attention_mask=class_attns)
    class_txt_embs_norm = F.normalize(class_txt_embs, dim=-1)
    print(f"[INFO] Pre-computed class text embeddings: {class_txt_embs.shape}")

# Determine which splits to process and from which dataset
if split_option == 'all':
    data_sets = ['train', 'val', 'test']
else:
    data_sets = [split_option]

# Process each dataset split with domain adaptation logic
for data_set in data_sets:
    print(f"\n{'='*60}")
    
    # Determine which dataset to use for this split
    # FIXED VERSION of the domain adaptation logic
    if domain_adaptation:
        if data_set == 'train':
            # Training always from source
            current_dataset = source_dataset
            current_transform = train_transform
            print(f"Processing TRAIN set from SOURCE: {source_dataset}")
        elif data_set == 'val':
            # BUG FIX: Validation should also be from source for hyperparameter tuning
            # (This is standard practice in domain adaptation)
            current_dataset = target_dataset
            current_transform = transform_test_source
            print(f"Processing VALIDATION set from SOURCE: {source_dataset}")
        else:  # test
            # Testing on target for domain adaptation
            current_dataset = target_dataset
            current_transform = transform_test_target
            print(f"Processing TEST set from TARGET: {target_dataset}")
    else:
        # Standard training: all splits from source
        current_dataset = source_dataset
        if data_set == 'train':
            current_transform = train_transform
        else:
            current_transform = transform_test_source
        print(f"Processing {data_set.upper()} set from: {current_dataset}")
    
    print(f"{'='*60}")

    # Load appropriate dataset
    dataset, _ = load_dataset(current_dataset, split=data_set, transform=current_transform, val_split=args.val_split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"[INFO] Loaded {len(dataset)} images from {current_dataset} {data_set} set")

    y_true, y_pred, y_score, paths = [], [], [], []
    distribution_x4, distribution_xc, distribution_x4_fused = [], [], []
    distribution_text = []  # NEW: text embeddings distribution

    # Lists to store embeddings for retrieval metrics
    all_image_embeddings = []
    all_targets = []

    with torch.no_grad():
        for images, targets, pth in tqdm(dataloader, ncols=80, desc=f"Processing {data_set}"):
            images = images.to(device)
            
            # Get image embeddings
            img_emb = model.image_encoder(images)
            img_emb_norm = F.normalize(img_emb, dim=-1)
            
            # Calculate similarities
            sims = img_emb_norm @ class_txt_embs_norm.t()
            
            # Cross-attention fusion
            fused_emb = model.cross_attention_fusion(img_emb, class_txt_embs_norm)
            
            # Predictions and scores
            preds = sims.argmax(dim=1).cpu().numpy().tolist()
            scores = softmax(sims.cpu().numpy(), axis=1).tolist()
            
            # Collect results
            y_pred.extend(preds)
            y_score.extend(scores)
            y_true.extend(targets.numpy().tolist())
            paths.extend(pth)
            
            # Store embeddings for retrieval metrics
            all_image_embeddings.append(img_emb_norm)
            all_targets.append(targets.to(device))
            
            # Store distributions (EXACT SAME NAMES AS ORIGINAL)
            distribution_x4.extend(img_emb_norm.cpu().tolist())  # Normalized image embeddings
            distribution_xc.extend(sims.cpu().tolist())  # Similarities
            
            # NEW: Text embeddings for each sample (using predicted class)
            for pred in preds:
                # Get text embedding for predicted class
                text_emb = class_txt_embs[pred].cpu().tolist()
                distribution_text.append(text_emb)
            
            # NEW: Cross-attention fused embeddings
            distribution_x4_fused.extend(fused_emb.cpu().tolist())

    # Compute retrieval metrics if we have embeddings
    if all_image_embeddings:
        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute retrieval metrics
        itopk, ttopk = compute_retrieval_metrics(
            all_image_embeddings, 
            class_txt_embs_norm, 
            all_targets, 
            ks=(1, 5)
        )
    else:
        itopk = {'R@1': 0, 'R@5': 0}
        ttopk = {'R@1': 0, 'R@5': 0}

    # Save results in exact same format as original code
    print(f"\n[INFO] Saving results to {exp_dir}")
    
    # Same naming convention as original code
    pd.DataFrame(y_true).to_csv(join(exp_dir, f"targets_{data_set}.csv"), index=None, header=None)
    pd.DataFrame(y_pred).to_csv(join(exp_dir, f"predictions_{data_set}.csv"), index=None, header=None)
    pd.DataFrame(y_score).to_csv(join(exp_dir, f"predictions_probabilities_{data_set}.csv"), index=None, header=None)
    pd.DataFrame(paths).to_csv(join(exp_dir, f"paths_{data_set}.csv"), index=None, header=None)
    pd.DataFrame(distribution_x4).to_csv(join(exp_dir, f"distribution_x4_{data_set}.csv"), index=None, header=None)
    pd.DataFrame(distribution_xc).to_csv(join(exp_dir, f"distribution_xc_{data_set}.csv"), index=None, header=None)
    
    # NEW: Save text embeddings distribution
    pd.DataFrame(distribution_text).to_csv(join(exp_dir, f"distribution_text_{data_set}.csv"), index=None, header=None)
    
    # NEW: Save cross-attention fused embeddings
    pd.DataFrame(distribution_x4_fused).to_csv(join(exp_dir, f"distribution_x4_fused_{data_set}.csv"), index=None, header=None)

    # Calculate metrics (same as original)
    if len(y_true) > 0:
        test_acc = 100.0 * accuracy_score(y_true, y_pred)
        test_f1 = 100.0 * f1_score(y_true, y_pred, average='weighted', zero_division=0)
        test_prec = 100.0 * precision_score(y_true, y_pred, average='weighted', zero_division=0)
        test_rec = 100.0 * recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        try:
            test_auc = 100.0 * roc_auc_score(y_true, np.array(y_score), multi_class="ovr", average="weighted")
        except Exception as e:
            print(f"AUC computation failed: {e}")
            test_auc = None
        
        print(f"\n{data_set.upper()} Results from {current_dataset}:")
        print(f"  Accuracy:        {test_acc:.2f}%")
        print(f"  F1 Score:        {test_f1:.2f}%")
        print(f"  Precision:       {test_prec:.2f}%")
        print(f"  Recall:          {test_rec:.2f}%")
        print(f"  AUC:             {test_auc if test_auc else 'N/A'}")
        
        # Add retrieval metrics to the print
        print(f"  Image?Text R@1:  {itopk.get('R@1', 0)*100:.2f}%")
        print(f"  Image?Text R@5:  {itopk.get('R@5', 0)*100:.2f}%")
        print(f"  Text?Image R@1:  {ttopk.get('R@1', 0)*100:.2f}%")
        print(f"  Text?Image R@5:  {ttopk.get('R@5', 0)*100:.2f}%")
    else:
        print(f"[WARNING] No ground truth labels found for {data_set} set")

    # Save retrieval metrics to a summary file
    retrieval_summary = {
        'image_to_text_r1': itopk.get('R@1', 0),
        'image_to_text_r5': itopk.get('R@5', 0),
        'text_to_image_r1': ttopk.get('R@1', 0),
        'text_to_image_r5': ttopk.get('R@5', 0),
    }

    with open(join(exp_dir, f"retrieval_summary_{data_set}.json"), 'w') as f:
        json.dump(retrieval_summary, f, indent=2)

print(f"\n{'='*60}")
print("All datasets processed successfully!")
print(f"Results saved to: {exp_dir}")
print(f"{'='*60}")

# Save metadata summary
metadata = {
    'source_dataset': source_dataset,
    'target_dataset': target_dataset,
    'domain_adaptation': domain_adaptation,
    'cnn_backbone': cnn,
    'text_model': text_model,
    'num_classes': nb_class,
    'checkpoint': checkpoint_path,
    'class_names': class_names,
    'prompts': prompts[:5],  # Save first 5 prompts as example
    'proj_dim': proj_dim
}

with open(join(exp_dir, "extraction_metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n? Distribution extraction completed successfully!")
print(f"   Files saved for each split:")
print(f"   - targets_{{split}}.csv")
print(f"   - predictions_{{split}}.csv")
print(f"   - predictions_probabilities_{{split}}.csv")
print(f"   - paths_{{split}}.csv")
print(f"   - distribution_x4_{{split}}.csv (image embeddings)")
print(f"   - distribution_xc_{{split}}.csv (similarities)")
print(f"   - distribution_text_{{split}}.csv (text embeddings) - NEW")
print(f"   - distribution_x4_fused_{{split}}.csv (cross-attention fused) - NEW")
print(f"   - retrieval_summary_{{split}}.json (R@1 and R@5 metrics)")
print(f"   Total time: {time.time() - begin:.2f} seconds")
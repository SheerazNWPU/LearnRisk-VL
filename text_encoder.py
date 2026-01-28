import torch.nn.functional as F
import os, pickle, time, shutil, argparse, math, random, json
from os.path import join
from glob import glob
# HF transformers (needed for text encoder)
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision import transforms, datasets
from tqdm import tqdm
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from PIL import Image
# ==================== SIMPLIFIED TEXT ENCODERS ====================
# ==================== SIMPLIFIED TEXT ENCODERS ====================
class SimpleDeepSeekTextEncoder(nn.Module):
    def __init__(self, proj_dim=512, local_model_path=None):
        super().__init__()
        print(f"[INFO] Loading DeepSeek tokenizer from: {local_model_path}")
        
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
        nn.init.normal_(self.embedding_layer.weight, mean=0.0, std=0.02)
        
        self.proj = nn.Linear(self.hidden_size, proj_dim, bias=False)
        self.ln = nn.LayerNorm(proj_dim)
        self.pooling = 'mean'

    def forward(self, input_ids=None, attention_mask=None, texts=None):
        # Tokenize if texts provided
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
        
        # Ensure tensors are on correct device
        device = self.proj.weight.device
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        if attention_mask is not None and attention_mask.device != device:
            attention_mask = attention_mask.to(device)
        
        # Get token embeddings
        token_embeddings = self.embedding_layer(input_ids)
        
        # Pooling
        if self.pooling == 'mean' and attention_mask is not None:
            attention_mask_expanded = attention_mask.unsqueeze(-1).float()
            token_embeddings = token_embeddings * attention_mask_expanded
            sum_embeddings = torch.sum(token_embeddings, dim=1)
            sum_mask = torch.sum(attention_mask_expanded, dim=1)
            embeddings = sum_embeddings / (sum_mask + 1e-8)
        else:
            embeddings = torch.mean(token_embeddings, dim=1)
        
        embeddings = self.ln(self.proj(embeddings))
        return embeddings

class SimpleBERTTextEncoder(nn.Module):
    def __init__(self, proj_dim=512, model_path=None):
        super().__init__()
        from transformers import BertTokenizer, BertModel
        
        print(f"[BERT] Loading from {model_path}")
        
        self.tokenizer = BertTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        self.bert = BertModel.from_pretrained(
            model_path,
            local_files_only=True,
            output_hidden_states=True  # IMPORTANT: Get all hidden states
        )
        self.hidden_size = self.bert.config.hidden_size
        self.proj = nn.Linear(self.hidden_size, proj_dim, bias=False)
        self.ln = nn.LayerNorm(proj_dim)
        self.pooling = 'mean'  # Change to mean pooling for better gradients
        
        # Initialize projection with small weights
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        
        print(f"[BERT] Hidden size: {self.hidden_size}, Projection to: {proj_dim}")
    
    def forward(self, input_ids=None, attention_mask=None, texts=None):
        # Tokenize if texts provided
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
        
        # Move to correct device
        device = next(self.bert.parameters()).device
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        if attention_mask is not None and attention_mask.device != device:
            attention_mask = attention_mask.to(device)
        
        # Get ALL hidden states
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, 
                           output_hidden_states=True)
        
        # Use last 4 layers average for better gradients
        if hasattr(outputs, 'hidden_states'):
            # Take average of last 4 layers
            last_four_layers = outputs.hidden_states[-4:]
            hidden_states = torch.stack(last_four_layers, dim=0).mean(dim=0)
        else:
            hidden_states = outputs.last_hidden_state
        
        # Pooling
        if self.pooling == 'mean' and attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            embeddings = hidden_states * mask
            pooled = torch.sum(embeddings, dim=1) / (torch.sum(mask, dim=1) + 1e-8)
        else:  # cls pooling
            pooled = hidden_states[:, 0, :]
        
        # Project and normalize
        projected = self.proj(pooled)
        embeddings = self.ln(projected)
        
        return embeddings

# ==================== FIXED DeBERTa ENCODER ====================
# ==================== FIXED DeBERTa ENCODER ====================
class SimpleDeBERTaTextEncoder(nn.Module):
    def __init__(self, proj_dim=512, model_path=None):
        super().__init__()
        print(f"[DeBERTa-FIXED] Loading from: {model_path}")
        
        # Load tokenizer from YOUR local path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        # CRITICAL: Add pad token if missing
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print(f"[DeBERTa] Added pad_token: {self.tokenizer.pad_token}")
        
        # Load model from YOUR local path
        self.deberta = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get hidden size
        self.hidden_size = self.deberta.config.hidden_size
        
        # Projection layer - ALWAYS TRAINABLE
        self.proj = nn.Linear(self.hidden_size, proj_dim, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        
        # LayerNorm - ALWAYS TRAINABLE
        self.ln = nn.LayerNorm(proj_dim)
        
        print(f"[DeBERTa] Hidden size: {self.hidden_size}, Projection: {self.hidden_size} -> {proj_dim}")
        
        # Set up proper gradient flow - AFTER proj and ln are initialized
        self._setup_gradients()
        
    def _setup_gradients(self):
        """Proper gradient setup for DeBERTa"""
        total_layers = self.deberta.config.num_hidden_layers
        print(f"[DeBERTa] Total layers: {total_layers}")
        
        # Freeze most layers, train last 2 layers + pooler
        layer_names = []
        for name, param in self.deberta.named_parameters():
            # Freeze embeddings
            if 'embeddings' in name:
                param.requires_grad = False
            # Train last 2 transformer layers
            elif f'encoder.layer.{total_layers-2}' in name or f'encoder.layer.{total_layers-1}' in name:
                param.requires_grad = True
                layer_names.append(name.split('.')[-2] if '.' in name else name)
            # Train pooler
            elif 'pooler' in name:
                param.requires_grad = True
                layer_names.append('pooler')
            # Freeze everything else
            else:
                param.requires_grad = False
        
        # Always train projection and LayerNorm
        for param in self.proj.parameters():
            param.requires_grad = True
        for param in self.ln.parameters():
            param.requires_grad = True
        
        print(f"[DeBERTa] Training layers: {set(layer_names)}")
        print("[DeBERTa] Gradient setup complete: last 2 layers + pooler + projection trainable")
    
    def forward(self, input_ids=None, attention_mask=None, texts=None):
        # Tokenize if texts provided
        if texts is not None:
            tokens = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            input_ids = tokens["input_ids"]
            attention_mask = tokens.get("attention_mask", None)
        
        # Move to correct device
        device = next(self.deberta.parameters()).device
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        if attention_mask is not None and attention_mask.device != device:
            attention_mask = attention_mask.to(device)
        
        # CRITICAL: Use return_dict=True
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Use pooler_output if available (DeBERTa has it), otherwise use CLS token
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # Fallback: use mean pooling of last hidden state
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                hidden_states = outputs.last_hidden_state * mask_expanded
                pooled = torch.sum(hidden_states, dim=1) / (torch.sum(mask_expanded, dim=1) + 1e-8)
            else:
                pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Project and normalize
        projected = self.proj(pooled)
        normalized = self.ln(projected)
        
        return normalized

# ==================== FIXED Flan-T5 ENCODER ====================
# ==================== FIXED Flan-T5 ENCODER ====================
class SimpleFlanT5TextEncoder(nn.Module):
    def __init__(self, proj_dim=512, model_path=None):
        super().__init__()
        print(f"[Flan-T5-FIXED] Loading from {model_path}")
        
        # Load tokenizer from YOUR local path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        # Use AutoModel instead of T5Model for flexibility
        # This will load T5Model from your local path
        self.t5 = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            output_hidden_states=True
        )
        
        self.hidden_size = self.t5.config.d_model
        
        # Initialize projection and LayerNorm FIRST
        self.proj = nn.Linear(self.hidden_size, proj_dim, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        
        self.ln = nn.LayerNorm(proj_dim)
        
        print(f"[Flan-T5] Hidden size: {self.hidden_size}, Projection to: {proj_dim}")
        
        # CRITICAL: Setup gradients AFTER initializing proj and ln
        self._setup_gradients()
    
    def _setup_gradients(self):
        """Setup proper gradient flow for T5"""
        # Freeze all T5 parameters initially
        for param in self.t5.parameters():
            param.requires_grad = False
        
        # Check if it has decoder (T5 has encoder-decoder architecture)
        if hasattr(self.t5, 'decoder') and hasattr(self.t5.decoder, 'block'):
            decoder_layers = self.t5.decoder.block
            total_decoder_layers = len(decoder_layers)
            
            # Unfreeze last 2 decoder layers
            for i in range(max(0, total_decoder_layers - 2), total_decoder_layers):
                for param in decoder_layers[i].parameters():
                    param.requires_grad = True
                print(f"[Flan-T5] Training decoder layer {i}")
        else:
            # If no decoder, unfreeze last 2 encoder layers
            print("[Flan-T5] No decoder found, unfreezing last 2 encoder layers")
            if hasattr(self.t5, 'encoder') and hasattr(self.t5.encoder, 'block'):
                encoder_layers = self.t5.encoder.block
                total_encoder_layers = len(encoder_layers)
                
                for i in range(max(0, total_encoder_layers - 2), total_encoder_layers):
                    for param in encoder_layers[i].parameters():
                        param.requires_grad = True
                    print(f"[Flan-T5] Training encoder layer {i}")
            else:
                # For models without block structure, unfreeze last layers differently
                print("[Flan-T5] No block structure found, training last 2 layers by name")
                layer_names = []
                for name, param in self.t5.named_parameters():
                    if any([f'layer.{i}' in name for i in [self.t5.config.num_hidden_layers-2, 
                                                           self.t5.config.num_hidden_layers-1]]):
                        param.requires_grad = True
                        layer_names.append(name.split('.')[-2] if '.' in name else name)
                print(f"[Flan-T5] Training layers: {set(layer_names)}")
        
        # Always train projection and layernorm
        for param in self.proj.parameters():
            param.requires_grad = True
        for param in self.ln.parameters():
            param.requires_grad = True
        
        print("[Flan-T5] Gradient setup complete")
    
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
            attention_mask = tokens.get("attention_mask", None)
        
        device = next(self.t5.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # T5 forward pass - check if it has encoder attribute
        if hasattr(self.t5, 'encoder'):
            # Use encoder for T5 models
            outputs = self.t5.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            last_hidden_state = outputs.last_hidden_state
        else:
            # For encoder-only models
            outputs = self.t5(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            last_hidden_state = outputs.last_hidden_state
        
        # Use mean pooling of last hidden state
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            hidden_states = last_hidden_state * mask_expanded
            pooled = torch.sum(hidden_states, dim=1) / (torch.sum(mask_expanded, dim=1) + 1e-8)
        else:
            pooled = torch.mean(last_hidden_state, dim=1)
        
        projected = self.proj(pooled)
        embeddings = self.ln(projected)
        
        return embeddings

class SimpleQwenTextEncoder(nn.Module):
    def __init__(self, proj_dim=512, model_path=None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        self.hidden_size = getattr(config, "hidden_size", 4096)
        vocab_size = getattr(config, "vocab_size", 151936)
        self.embedding_layer = nn.Embedding(vocab_size, self.hidden_size)
        nn.init.normal_(self.embedding_layer.weight, mean=0.0, std=0.02)
        self.proj = nn.Linear(self.hidden_size, proj_dim, bias=False)
        self.ln = nn.LayerNorm(proj_dim)
        self.pooling = 'mean'
    
    def forward(self, input_ids=None, attention_mask=None, texts=None):
        # Tokenize if texts provided
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
        
        # Move to correct device
        device = self.proj.weight.device
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        if attention_mask is not None and attention_mask.device != device:
            attention_mask = attention_mask.to(device)
        
        # Get token embeddings
        token_embeddings = self.embedding_layer(input_ids)
        
        # Pooling
        if self.pooling == 'mean' and attention_mask is not None:
            attention_mask_expanded = attention_mask.unsqueeze(-1).float()
            token_embeddings = token_embeddings * attention_mask_expanded
            sum_embeddings = torch.sum(token_embeddings, dim=1)
            sum_mask = torch.sum(attention_mask_expanded, dim=1)
            embeddings = sum_embeddings / (sum_mask + 1e-8)
        else:
            embeddings = torch.mean(token_embeddings, dim=1)
        
        embeddings = self.ln(self.proj(embeddings))
        return embeddings
class SimpleGPTTextEncoder(nn.Module):
    """
    HuggingFace encoder backbone (e.g., GPT-2, DistilBERT).
    Uses mean pooling for encoder-only models, and last-token embedding for GPT-like models.
    Supports local model loading from cache.
    """
    def __init__(self, model_name="gpt2", proj_dim=512, model_path=None):
        super().__init__()
        self.model_name = model_name

        try:
            if model_path:
                print(f"[INFO] Loading HF model from local path: {model_path}")
                if not os.path.exists(model_path):
                    raise ValueError(f"Model path does not exist: {model_path}")
                    
                self.backbone = AutoModel.from_pretrained(
                    model_path,
                    local_files_only=True
                )
            else:
                print(f"[INFO] Loading HF model from Hugging Face Hub: {model_name}")
                self.backbone = AutoModel.from_pretrained(model_name)
                
        except Exception as e:
            print(f"[ERROR] Failed to load model {model_name} from path {model_path}: {e}")
            raise

        hidden_size = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden_size, proj_dim, bias=False)
        self.ln = nn.LayerNorm(proj_dim)

    def forward(self, input_ids, attention_mask=None):  # Changed from attn_mask to attention_mask
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state  # (B, L, H)

        if "gpt" in self.model_name.lower() or getattr(self.backbone.config, "is_decoder", False):
            # GPT-style causal LMs: take last non-padded token
            if attention_mask is not None:  # Changed from attn_mask
                last_indices = attention_mask.sum(dim=1) - 1  # index of last valid token
                x = x[torch.arange(x.size(0)), last_indices]  # (B, H)
            else:
                x = x[:, -1, :]  # fallback to last token
        else:
            # Encoder-only models (BERT, DistilBERT, RoBERTa): mean pooling
            if attention_mask is not None:  # Changed from attn_mask
                denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1).float()
                x = (x * attention_mask.unsqueeze(-1)).sum(dim=1) / denom.squeeze(-1)
            else:
                x = x.mean(dim=1)

        x = self.ln(self.proj(x))
        return F.normalize(x, dim=-1)
class SimpleLLaVATextEncoder(nn.Module):
    def __init__(self, proj_dim=512, model_path=None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        self.hidden_size = getattr(config, "hidden_size", 4096)
        vocab_size = getattr(config, "vocab_size", 32000)
        self.embedding_layer = nn.Embedding(vocab_size, self.hidden_size)
        nn.init.normal_(self.embedding_layer.weight, mean=0.0, std=0.02)
        self.proj = nn.Linear(self.hidden_size, proj_dim, bias=False)
        self.ln = nn.LayerNorm(proj_dim)
        self.pooling = 'mean'
    
    def forward(self, input_ids=None, attention_mask=None, texts=None):
        # Tokenize if texts provided
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
        
        # Move to correct device
        device = self.proj.weight.device
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        if attention_mask is not None and attention_mask.device != device:
            attention_mask = attention_mask.to(device)
        
        # Get token embeddings
        token_embeddings = self.embedding_layer(input_ids)
        
        # Pooling
        if self.pooling == 'mean' and attention_mask is not None:
            attention_mask_expanded = attention_mask.unsqueeze(-1).float()
            token_embeddings = token_embeddings * attention_mask_expanded
            sum_embeddings = torch.sum(token_embeddings, dim=1)
            sum_mask = torch.sum(attention_mask_expanded, dim=1)
            embeddings = sum_embeddings / (sum_mask + 1e-8)
        else:
            embeddings = torch.mean(token_embeddings, dim=1)
        
        embeddings = self.ln(self.proj(embeddings))
        return embeddings
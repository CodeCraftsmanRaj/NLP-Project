"""Training pipeline."""

import torch
from torch.optim import AdamW
from config import LR, EPOCHS
from tqdm import tqdm
import json
from pathlib import Path

def train(model, loader, device, epochs=EPOCHS, lr=LR, verbose=True):
    """Train model on dataset.
    
    Args:
        model: PyTorch model to train
        loader: DataLoader for training data
        device: Device to train on (cpu/cuda)
        epochs: Number of training epochs
        lr: Learning rate
        verbose: Print progress
    
    Returns:
        tuple: (losses, model)
    """
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    losses = []
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", disable=not verbose)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lex = batch["lex"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, lex)
            loss = loss_fn(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
    
    return losses, model

def save_checkpoint(model, optimizer, epoch, loss, save_path="models_checkpoint/latest.pt"):
    """Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save checkpoint
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    print(f"✓ Checkpoint saved: {save_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint
    
    Returns:
        tuple: (epoch, loss)
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"✓ Checkpoint loaded from: {checkpoint_path}")
    return epoch, loss
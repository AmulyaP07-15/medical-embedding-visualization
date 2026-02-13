"""
Triplet Margin Loss Training for BioBERT Fine-tuning

Fine-tunes BioBERT on medical specialties using triplet loss.
Improves embedding quality for specialty clustering.

Architecture:
- Base: BioBERT (dmis-lab/biobert-v1.1)
- Projection head: 768
- Loss: Triplet margin loss (margin=0.5)
- Pooling: Mean pooling

OPTIMIZED FOR MAC MPS - Faster training with reduced parameters
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from triplet_dataset import load_triplet_data
from tqdm import tqdm
import numpy as np
import os

class TripletBERT(nn.Module):
    """
    BioBERT with projection head for triplet loss fine-tuning
    """
    def __init__(self, model_name="dmis-lab/biobert-v1.1", projection_dim=128):
        super().__init__()

        print(f"Loading base model: {model_name}")

        # Load pre-trained BioBERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Projection head: reduces dimensionality and learns task-specific features
        # 768 (BioBERT) -> projection_dim
        self.projection = nn.Sequential(
            nn.Linear(768, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, projection_dim)
        )

        print(f"✓ Model loaded with {projection_dim}-dim projection head")

    def mean_pooling(self, token_embeddings, attention_mask):
        """
        Mean pooling: average token embeddings, weighted by attention mask

        Args:
            token_embeddings: (batch_size, seq_len, 768)
            attention_mask: (batch_size, seq_len)

        Returns:
            pooled: (batch_size, 768)
        """
        # Expand attention mask to match embedding dimensions
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum embeddings weighted by mask
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)

        # Sum mask values (number of real tokens)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

        # Average
        return sum_embeddings / sum_mask

    def encode(self, texts, device):
        """
        Encode texts to embeddings

        Args:
            texts: List of strings
            device: torch device

        Returns:
            embeddings: (batch_size, projection_dim) - L2 normalized
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        # Move to device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Mean pooling
        token_embeddings = outputs.last_hidden_state
        pooled = self.mean_pooling(token_embeddings, attention_mask)

        # Project to lower dimension
        projected = self.projection(pooled)

        # L2 normalize (important for cosine-based triplet loss)
        normalized = nn.functional.normalize(projected, p=2, dim=1)

        return normalized

    def forward(self, anchor, positive, negative, device):
        """
        Forward pass for triplet

        Args:
            anchor: List of anchor texts
            positive: List of positive texts (same specialty)
            negative: List of negative texts (different specialty)
            device: torch device

        Returns:
            anchor_emb, positive_emb, negative_emb (all normalized)
        """
        anchor_emb = self.encode(anchor, device)
        positive_emb = self.encode(positive, device)
        negative_emb = self.encode(negative, device)

        return anchor_emb, positive_emb, negative_emb


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
    Train for one epoch

    Returns:
        average_loss: float
    """
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        optimizer.zero_grad()

        # Get triplets from batch
        anchors = batch['anchor']
        positives = batch['positive']
        negatives = batch['negative']

        # Forward pass
        anchor_emb, positive_emb, negative_emb = model(anchors, positives, negatives, device)

        # Compute triplet loss
        loss = criterion(anchor_emb, positive_emb, negative_emb)

        # Backward pass
        loss.backward()

        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/num_batches:.4f}'
        })

    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """
    Validate the model

    Returns:
        avg_loss: float
        accuracy: float (% where d(anchor,pos) < d(anchor,neg))
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            anchors = batch['anchor']
            positives = batch['positive']
            negatives = batch['negative']

            # Forward pass
            anchor_emb, positive_emb, negative_emb = model(anchors, positives, negatives, device)

            # Compute loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            total_loss += loss.item()
            num_batches += 1

            # Compute accuracy: positive should be closer than negative
            pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb)
            neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb)

            correct += (pos_dist < neg_dist).sum().item()
            total += len(anchors)

    avg_loss = total_loss / num_batches
    accuracy = correct / total

    return avg_loss, accuracy


def main():
    """
    Main training loop
    """
    # =========================================================================
    # Hyperparameters - OPTIMIZED FOR FAST TRAINING ON MAC
    # =========================================================================
    BATCH_SIZE = 8        # Reduced from 16 (faster per batch)
    EPOCHS = 3              # Reduced from 10 (faster overall)
    LEARNING_RATE = 2e-5
    MARGIN = 0.5
    PROJECTION_DIM = 768

    # =========================================================================
    # Setup
    # =========================================================================

    # Device selection
    # FORCE CPU - MPS runs out of memory on 8GB Mac
    device = torch.device('cpu')
    print("✓ Using CPU (slower but won't crash)")
    print("⚠ Training will take ~5-6 hours total")

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # =========================================================================
    # Load Data
    # =========================================================================

    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    train_dataset, val_dataset = load_triplet_data()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Set to 0 for Mac MPS compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")

    # =========================================================================
    # Initialize Model
    # =========================================================================

    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)

    model = TripletBERT(
        model_name='dmis-lab/biobert-v1.1',
        projection_dim=PROJECTION_DIM
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # =========================================================================
    # Loss and Optimizer
    # =========================================================================

    # Triplet Margin Loss
    # L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    criterion = nn.TripletMarginLoss(
        margin=MARGIN,
        p=2,  # L2 distance (Euclidean)
        reduction='mean'
    )

    # AdamW optimizer (better than Adam for transformers)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01  # L2 regularization
    )

    # Learning rate scheduler (optional but recommended)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )

    # =========================================================================
    # Training Loop
    # =========================================================================

    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Margin: {MARGIN}")
    print(f"Projection dim: {PROJECTION_DIM}")
    print(f"Device: {device}")
    print("="*80 + "\n")

    best_val_loss = float('inf')
    best_val_accuracy = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{EPOCHS}")
        print(f"{'='*80}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        print(f"\n✓ Train Loss: {train_loss:.4f}")

        # Validate
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        print(f"✓ Val Loss: {val_loss:.4f}")
        print(f"✓ Val Accuracy: {val_accuracy:.2%}")

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"✓ Learning Rate: {current_lr:.2e}")

        # Save best model (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy

            print(f"\n🌟 New best model! Saving...")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'hyperparameters': {
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE,
                    'margin': MARGIN,
                    'projection_dim': PROJECTION_DIM
                }
            }, 'models/triplet_biobert_best.pt')

            print(f"✓ Saved to: models/triplet_biobert_best.pt")

    # =========================================================================
    # Training Complete
    # =========================================================================

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2%}")
    print(f"Model saved to: models/triplet_biobert_best.pt")
    print("="*80 + "\n")

    print("Next steps:")
    print("1. Generate embeddings: python generate_finetuned_embeddings.py")
    print("2. Evaluate improvement: python evaluate_improvement.py")
    print("3. Add to dashboard")


if __name__ == "__main__":
    main()
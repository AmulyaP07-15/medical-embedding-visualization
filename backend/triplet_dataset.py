"""
Triplet Dataset for Medical Note Fine-tuning

Creates (anchor, positive, negative) triplets where:
- Anchor: Random medical note
- Positive: Different note from SAME specialty
- Negative: Note from DIFFERENT specialty
"""

import numpy as np
import pickle
import random
from torch.utils.data import Dataset


class MedicalTripletDataset(Dataset):
    """
    PyTorch Dataset that generates triplets on-the-fly
    """
    
    def __init__(self, texts, specialties, num_triplets=5000, seed=42):
        """
        Args:
            texts: List of medical note texts (900 notes)
            specialties: List of specialty labels (900 labels)
            num_triplets: How many triplets to generate
            seed: Random seed for reproducibility
        """
        self.texts = texts
        self.specialties = specialties
        self.num_triplets = num_triplets
        
        # Set seed for reproducibility
        random.seed(seed)
        
        # Create index mapping: specialty -> list of indices
        # This makes sampling efficient
        self.specialty_to_indices = {}
        
        for idx, specialty in enumerate(specialties):
            if specialty not in self.specialty_to_indices:
                self.specialty_to_indices[specialty] = []
            self.specialty_to_indices[specialty].append(idx)
        
        # List of all specialties
        self.specialty_list = list(self.specialty_to_indices.keys())
        
        print(f"Created dataset with {len(self.specialty_list)} specialties:")
        for specialty, indices in self.specialty_to_indices.items():
            print(f"  {specialty}: {len(indices)} samples")
    
    def __len__(self):
        """Number of triplets in dataset"""
        return self.num_triplets
    
    def __getitem__(self, idx):
        """
        Generate one triplet
        
        Returns:
            dict with keys: 'anchor', 'positive', 'negative'
        """
        # Step 1: Pick random anchor
        anchor_idx = random.randint(0, len(self.texts) - 1)
        anchor_text = self.texts[anchor_idx]
        anchor_specialty = self.specialties[anchor_idx]
        
        # Step 2: Pick positive (same specialty, different text)
        # Get all indices for this specialty EXCEPT the anchor
        positive_candidates = [i for i in self.specialty_to_indices[anchor_specialty] 
                              if i != anchor_idx]
        
        # Safety check (should always have at least 1 other sample per specialty)
        if len(positive_candidates) == 0:
            # Fallback: use anchor itself (rare edge case)
            positive_idx = anchor_idx
        else:
            positive_idx = random.choice(positive_candidates)
        
        positive_text = self.texts[positive_idx]
        
        # Step 3: Pick negative (different specialty)
        # Get all specialties except anchor's specialty
        negative_specialties = [s for s in self.specialty_list 
                               if s != anchor_specialty]
        
        # Pick random specialty
        negative_specialty = random.choice(negative_specialties)
        
        # Pick random sample from that specialty
        negative_idx = random.choice(self.specialty_to_indices[negative_specialty])
        negative_text = self.texts[negative_idx]
        
        return {
            'anchor': anchor_text,
            'positive': positive_text,
            'negative': negative_text,
            # Optional: include specialties for debugging
            'anchor_specialty': anchor_specialty,
            'negative_specialty': negative_specialty
        }


def load_triplet_data(train_size=0.8):
    """
    Load MTSamples data and create train/validation split
    
    Args:
        train_size: Fraction of data for training (0.8 = 80%)
        
    Returns:
        train_dataset, val_dataset
    """
    # Load the cached data
    with open('cache/data_info.pkl', 'rb') as f:
        data = pickle.load(f)
    
    texts = data['texts']
    specialties = data['specialties']
    
    print(f"\nLoaded {len(texts)} medical notes")
    
    # Create train/val split
    n = len(texts)
    indices = list(range(n))
    random.seed(42)  # For reproducibility
    random.shuffle(indices)
    
    split_idx = int(train_size * n)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Split data
    train_texts = [texts[i] for i in train_indices]
    train_specialties = [specialties[i] for i in train_indices]
    
    val_texts = [texts[i] for i in val_indices]
    val_specialties = [specialties[i] for i in val_indices]
    
    print(f"Train: {len(train_texts)} samples")
    print(f"Val:   {len(val_texts)} samples")
    
    # Create datasets
    # More triplets for training, fewer for validation
    train_dataset = MedicalTripletDataset(
        train_texts, 
        train_specialties, 
        num_triplets=2000,
        seed=42
    )
    
    val_dataset = MedicalTripletDataset(
        val_texts, 
        val_specialties, 
        num_triplets=400,
        seed=43  # Different seed for validation
    )
    
    return train_dataset, val_dataset


# Test/demo code
if __name__ == "__main__":
    print("="*80)
    print("MEDICAL TRIPLET DATASET TEST")
    print("="*80)
    
    # Load datasets
    train_ds, val_ds = load_triplet_data()
    
    print(f"\n{'='*80}")
    print("SAMPLE TRIPLETS")
    print("="*80)
    
    # Show a few examples
    for i in range(3):
        print(f"\n{'='*80}")
        print(f"TRIPLET {i+1}")
        print("="*80)
        
        sample = train_ds[i]
        
        print(f"\n📌 ANCHOR ({sample['anchor_specialty']}):")
        print(f"{sample['anchor'][:150]}...")
        
        print(f"\n✅ POSITIVE ({sample['anchor_specialty']}):")
        print(f"{sample['positive'][:150]}...")
        
        print(f"\n❌ NEGATIVE ({sample['negative_specialty']}):")
        print(f"{sample['negative'][:150]}...")
    
    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print("="*80)
    print(f"Training triplets:   {len(train_ds)}")
    print(f"Validation triplets: {len(val_ds)}")
    print(f"\n✓ Dataset ready for training!")

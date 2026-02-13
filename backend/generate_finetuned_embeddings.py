"""
Generate embeddings using fine-tuned BioBERT model
"""

import torch
import numpy as np
import pickle
from train_triplet import TripletBERT
from tqdm import tqdm

def generate_embeddings():
    """Generate embeddings using fine-tuned model"""
    
    # Load data
    print("Loading data...")
    with open('cache/data_info.pkl', 'rb') as f:
        data = pickle.load(f)
    
    texts = data['texts']
    specialties = data['specialties']
    
    print(f"✓ Loaded {len(texts)} texts")
    
    # Load fine-tuned model
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    print("Loading fine-tuned model...")
    model = TripletBERT(projection_dim=768)  # Match training dim
    
    checkpoint = torch.load('models/triplet_biobert_best.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded (epoch {checkpoint['epoch']}, val_acc: {checkpoint['val_accuracy']:.2%})")
    
    # Generate embeddings in batches
    batch_size = 32
    all_embeddings = []
    
    print("Generating embeddings...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        with torch.no_grad():
            embeddings = model.encode(batch_texts, device)
            all_embeddings.append(embeddings.cpu().numpy())
    
    # Concatenate
    all_embeddings = np.vstack(all_embeddings)
    print(f"✓ Generated embeddings shape: {all_embeddings.shape}")
    
    # Save
    np.save('cache/embeddings_triplet-biobert.npy', all_embeddings)
    print("✓ Saved to: cache/embeddings_triplet-biobert.npy")


if __name__ == "__main__":
    generate_embeddings()

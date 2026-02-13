"""
Create sample data for demo purposes
Run this after cloning the repo
"""

import numpy as np
import pickle
import os

# Create directories
os.makedirs('backend/cache', exist_ok=True)

# Create sample data (smaller dataset for demo)
print("Creating sample data...")

# Sample texts and specialties
specialties_list = ['Surgery', 'Orthopedic', 'Radiology', 'Gastroenterology', 'Neurology', 'Cardiovascular / Pulmonary']
num_samples = 100  # Smaller for demo

texts = [f"Sample medical note {i} for {specialties_list[i % 6]}" for i in range(num_samples)]
specialties = [specialties_list[i % 6] for i in range(num_samples)]

# Save data_info.pkl
data_info = {
    'texts': texts,
    'specialties': specialties
}

with open('backend/cache/data_info.pkl', 'wb') as f:
    pickle.dump(data_info, f)

print("✓ Created data_info.pkl")

# Create sample embeddings for each model
for model_name in ['sentence-bert', 'biobert', 'clinical-bert', 'triplet-biobert']:
    # Random embeddings (just for demo)
    if model_name == 'sentence-bert':
        dim = 384
    else:
        dim = 768
    
    embeddings = np.random.randn(num_samples, dim).astype(np.float32)
    np.save(f'backend/cache/embeddings_{model_name}.npy', embeddings)
    print(f"✓ Created embeddings_{model_name}.npy")

# Create sample t-SNE coordinates
for model_name in ['sentence-bert', 'biobert', 'clinical-bert', 'triplet-biobert']:
    coords = np.random.randn(num_samples, 3).astype(np.float32)
    np.save(f'backend/cache/tsne_{model_name}.npy', coords)
    print(f"✓ Created tsne_{model_name}.npy")

print("\n sample data created successfully!")Sample data created successfully!")
print("You can now run: python -m streamlit run streamlit_app.py")

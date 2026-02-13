import pandas as pd
import os

def load_mtsamples(filepath='../data/raw/mtsamples.csv'):
    """Load and clean MTSamples dataset"""
    df = pd.read_csv(filepath)
    
    # Remove rows with missing data
    df = df.dropna(subset=['description', 'medical_specialty'])
    
    # Clean text
    df['description'] = df['description'].str.strip()
    df['medical_specialty'] = df['medical_specialty'].str.strip()
    
    print(f"Loaded {len(df)} samples")
    print(f"Specialties: {df['medical_specialty'].nunique()}")
    
    return df

def get_specialty_counts(df):
    """Get sample counts per specialty"""
    counts = df['medical_specialty'].value_counts()
    return counts

def filter_by_specialties(df, specialties, n_samples_per_specialty=100):
    """
    Filter dataset to specific specialties with balanced samples
    
    Args:
        df: DataFrame with medical samples
        specialties: List of specialty names to include
        n_samples_per_specialty: Max samples per specialty
    """
    filtered_dfs = []
    
    for specialty in specialties:
        specialty_df = df[df['medical_specialty'] == specialty]
        # Take up to n_samples_per_specialty
        sampled = specialty_df.sample(n=min(len(specialty_df), n_samples_per_specialty), random_state=42)
        filtered_dfs.append(sampled)
    
    result = pd.concat(filtered_dfs, ignore_index=True)
    print(f"\nFiltered to {len(result)} samples across {len(specialties)} specialties")
    
    return result

if __name__ == "__main__":
    # Test the loader
    df = load_mtsamples()
    print("\nTop specialties by count:")
    print(get_specialty_counts(df).head(10))

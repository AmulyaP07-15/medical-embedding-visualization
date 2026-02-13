from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


class EmbeddingModel:
    """Base class for all embedding models"""

    def __init__(self, name):
        self.name = name
        self.model = None

    def encode(self, texts):
        """Generate embeddings for a list of texts"""
        raise NotImplementedError


class SentenceBERTModel(EmbeddingModel):
    """Sentence-BERT (general purpose)"""

    def __init__(self):
        super().__init__("Sentence-BERT")
        print(f"Loading {self.name}...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"✓ {self.name} loaded")

    def encode(self, texts):
        return self.model.encode(texts, show_progress_bar=True)


class BioBERTModel(EmbeddingModel):
    """BioBERT (biomedical domain)"""

    def __init__(self):
        super().__init__("BioBERT")
        print(f"Loading {self.name}...")
        self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        self.model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
        print(f"✓ {self.name} loaded")

    def encode(self, texts):
        embeddings = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                    max_length=512, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding (first token)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)


class ClinicalBERTModel(EmbeddingModel):
    """ClinicalBERT (clinical notes domain)"""

    def __init__(self):
        super().__init__("ClinicalBERT")
        print(f"Loading {self.name}...")
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        print(f"✓ {self.name} loaded")

    def encode(self, texts):
        embeddings = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                    max_length=512, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)


class TripletBioBERTModel(EmbeddingModel):
    """Fine-tuned BioBERT with Triplet Loss"""

    def __init__(self):
        super().__init__("Fine-tuned BioBERT")
        print(f"Loading {self.name}...")

        # Import the trained model class
        import sys
        sys.path.insert(0, '.')
        from train_triplet import TripletBERT

        # Load fine-tuned model
        self.model = TripletBERT(projection_dim=768)
        checkpoint = torch.load('models/triplet_biobert_best.pt', map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✓ {self.name} loaded (val_acc: {checkpoint['val_accuracy']:.2%})")

    def encode(self, texts):
        embeddings = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            with torch.no_grad():
                batch_embeddings = self.model.encode(batch, device='cpu')
                embeddings.append(batch_embeddings.numpy())

        return np.vstack(embeddings)


def load_all_models():
    """Load all embedding models"""
    models = {
        'sentence-bert': SentenceBERTModel(),
        'biobert': BioBERTModel(),
        'clinical-bert': ClinicalBERTModel(),
        'triplet-biobert': TripletBioBERTModel()  # NEW!
    }
    return models


# Available models metadata
AVAILABLE_MODELS = {
    'sentence-bert': {
        'name': 'Sentence-BERT (General)',
        'dimensions': 384,
        'description': 'General-purpose sentence embeddings, trained on diverse text'
    },
    'biobert': {
        'name': 'BioBERT (Biomedical)',
        'dimensions': 768,
        'description': 'Pretrained on biomedical literature (PubMed, PMC)'
    },
    'clinical-bert': {
        'name': 'ClinicalBERT (Clinical Notes)',
        'dimensions': 768,
        'description': 'Pretrained on clinical notes from MIMIC-III'
    },
    'triplet-biobert': {
        'name': 'Fine-tuned BioBERT (Triplet Loss)',
        'dimensions': 768,
        'description': 'BioBERT fine-tuned with triplet margin loss for specialty clustering (+59% retrieval accuracy)'
    }
}

if __name__ == "__main__":
    # Test loading models
    print("Testing model loading...\n")
    models = load_all_models()

    # Test with a sample sentence
    test_text = ["Patient presents with chest pain and shortness of breath."]

    for name, model in models.items():
        print(f"\nTesting {name}:")
        embeddings = model.encode(test_text)
        print(f"Embedding shape: {embeddings.shape}")
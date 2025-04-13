import argparse  
import torch
from torch import nn
from sentence_transformers import SentenceTransformer

class SimilarityLoss(nn.Module):
    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the similarity loss with sentence transformers embeddings.
        
        Args:
            embedding_model_name: Name of the sentence transformer model
        """
        super(SimilarityLoss, self).__init__()
        self.encoder = SentenceTransformer(embedding_model_name)

    def forward(self, query_texts, original_texts, generated_texts):
        """
        Compute custom loss: -cos(query, generated) + cos(original, generated)

        Args:
            query_texts: List of query texts (e.g. prompts)
            original_texts: List of original documents
            generated_texts: List of rewritten/generated documents

        Returns:
            Total loss (scalar)
        """
        with torch.no_grad():
            query_embs = self.encoder.encode(query_texts, convert_to_tensor=True)
            orig_embs = self.encoder.encode(original_texts, convert_to_tensor=True)
            gen_embs = self.encoder.encode(generated_texts, convert_to_tensor=True)

        # Compute cosine similarities
        sim_query_gen = torch.nn.functional.cosine_similarity(query_embs, gen_embs)
        sim_orig_gen = torch.nn.functional.cosine_similarity(orig_embs, gen_embs)

        # Final loss: -cos(query, generated) + cos(original, generated)
        loss = -sim_query_gen.mean() + sim_orig_gen.mean()
        return loss

# Define your loss class
similarity_loss_fn = SimilarityLoss()

# Example sentences
query_sentence = "what is best dog food for puppies"
original_sentence = "x is a good dog food"
generated_sentence = "a good dog food is x"

# Compute the loss (you don’t need logits/labels for this loss anymore)
loss = similarity_loss_fn(
    query_texts=[query_sentence],
    original_texts=[original_sentence],
    generated_texts=[generated_sentence],
)

print("Loss:", loss.item())

import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch import nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer

class SimilarityLoss(nn.Module):
    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the similarity loss with sentence transformers embeddings.
        
        Args:
            embedding_model_name: Name of the sentence transformer model
        """
        super(SimilarityLoss, self).__init__()
        # Initialize pre-trained sentence transformer model for semantic similarity
        self.encoder = SentenceTransformer(embedding_model_name)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels, original_texts, generated_texts):
        """
        Compute the custom loss function: cross-entropy + semantic similarity loss
        
        Args:
            logits: Predicted token logits from the model [batch, seq_len, vocab_size]
            labels: True token labels [batch, seq_len]
            original_texts: List of original documents
            generated_texts: List of generated documents (rewritten text)
            
        Returns:
            Total loss (cross-entropy + semantic similarity)
        """
        # Cross-entropy loss for token prediction (still applicable if labels are available)
        ce = self.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Compute semantic similarity (cosine similarity) between the original and generated documents
        with torch.no_grad():
            orig_embs = self.encoder.encode(original_texts, convert_to_tensor=True)
            generated_embs = self.encoder.encode(generated_texts, convert_to_tensor=True)

        # Cosine similarity (higher is better)
        cosine_sim = torch.nn.functional.cosine_similarity(orig_embs, generated_embs)
        similarity_loss = 1 - cosine_sim.mean()  # Minimize the difference

        # Total loss is a combination of both
        total_loss = ce + 1* similarity_loss  # You can adjust the weight of similarity loss
        return total_loss
    

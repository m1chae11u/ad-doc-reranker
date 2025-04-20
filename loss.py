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

    def forward(self, queries, original_doc, rewritten_doc, doc_domain, doc_subdomain):
        """
        Compute custom loss: 
        - avg(cos(matching queries, rewritten_doc)) + cos(original_doc, rewritten_doc)

        Args:
            queries: List of dicts with keys: 'domain', 'subdomain', 'query'
            original_doc: original ad used to create faiss index
            rewritten_doc: rewritten ad
            doc_domain: domain of doc
            doc_subdomain: subdomain of doc

        Returns:
            loss (scalar)
        """
        
        matching_queries = [q["query"] for q in queries if q["domain"] == doc_domain and q["subdomain"] == doc_subdomain]

        with torch.no_grad():
            gen_emb = self.encoder.encode([rewritten_doc], convert_to_tensor=True).squeeze(0)

            sims = []
            for q in matching_queries:
                q_emb = self.encoder.encode([q], convert_to_tensor=True).squeeze(0)
                sim = torch.nn.functional.cosine_similarity(q_emb, gen_emb, dim=0)
                sims.append(sim)
            if sims:
                avg_query_gen_sim = torch.stack(sims).mean()
                
        with torch.no_grad():
            orig_emb = self.encoder.encode([original_doc], convert_to_tensor=True).squeeze(0)
            gen_emb = self.encoder.encode([rewritten_doc], convert_to_tensor=True).squeeze(0)
            sim_orig_gen = torch.nn.functional.cosine_similarity(orig_emb, gen_emb, dim=0)

        # Final loss
        lambda_param = 1.0  # Can be adjusted based on how much you want to preserve content
        loss = -avg_query_gen_sim.mean() + lambda_param * (1 - sim_orig_gen.mean())
        return loss

# testing
queries = [
    {"domain": "electronics", "subdomain": "audio equipment", "query": "behringer juno chorus"},
    {"domain": "electronics", "subdomain": "audio equipment", "query": "best USB audio interface"},
    {"domain": "electronics", "subdomain": "monitors", "query": "studio monitors under 100"},
]

# Original and rewritten ad
original_doc = {
    "user_query": "behringer juno chorus",
    "title": "Behringer U-Phoria UMC22 USB Audio Interface 48kHz, 2-channel USB Audio Interface with 1 MIDAS Preamp, Phantom Power, and Instrument Input",
    "text": "Audio Interfaces - Recording quality audio in your Mac or Windows PC home studio is easy and rewarding, when you're recording with a Behringer U-Phoria UMC22. Any Sales Engineer here at Sweetwater, will be happy to tell you that you don't need the fanciest equipment out there to capture your ideas and bring your music to life, and the U-Phoria UMC22 USB audio interface gives you everything you need to get started. Onboard, you'll find a genuine MIDAS microphone preamp, which delivers truly impressive sound. There's also a dedicated instrument input onboard the UMC22, so you can record yourself singing as you play guitar or keys. And since the Behringer U-Phoria UMC22 includes a full copy of Tracktion, you're ready to start recording, right from day one.Real MIDAS microphone preamplifier technology onboardOne of the things that really makes the Behringer U-Phoria UMC22 stand out is its genuine MIDAS preamplifier technology. Known throughout the world of live sound for delivering ultra clear sound with plenty of headroom, MIDAS preamps are among the most popular mic pres on Earth. What's more, these preamps come standard with +48V phantom power onboard, so you can use your choice of quality condenser microphones with your UMC22. Call your Sweetwater Sales Engineer, and turn this U-Phoria UMC22 USB audio interface into a killer package deal with the perfect microphones and monitors for your complete recording rig.Behringer U-Phoria UMC22 USB Audio Interface Features: Quality 2-channel USB recording interface for your Mac or Windows PC. Combo input with MIDAS preamp lets you plug in any microphone or line-level gear. +48V phantom power lets you use studio condenser microphones. Dedicated instrument-level input accommodates your guitar or bass. Headphone and stereo 1/4\" outputs provide easy monitoring. Includes Tracktion DAW and 150 downloadable instrument/effect plug-ins. Start recording with a Behringer U-Phoria UMC22 USB audio interface! USB Audio Interfaces",
    "url": "sweetwater.com",
    "seller": "Sweetwater",
    "brand": "Behringer",
    "source": "microsoft_commercial_ads",
    "ad_id": "a767622a-e5f8-48f9-b026-e960bb488e96"
}
rewritten_doc = {
    "user_query": "behringer juno chorus",
    "title": "Behringer U-Phoria UMC22 USB Audio Interface 48kHz, 2-channel USB Audio Interface with 1 MIDAS Preamp, Phantom Power, and Instrument Input",
    "text": "Audio Interfaces - Recording quality audio in your Mac or Windows PC home studio is easy and rewarding, when you're recording with a Behringer U-Phoria UMC22. Any Sales Engineer here at Sweetwater, will be happy to tell you that you don't need the fanciest equipment out there to capture your ideas and bring your music to life, and the U-Phoria UMC22 USB audio interface gives you everything you need to get started. Onboard, you'll find a genuine MIDAS microphone preamp, which delivers truly impressive sound. There's also a dedicated instrument input onboard the UMC22, so you can record yourself singing as you play guitar or keys. And since the Behringer U-Phoria UMC22 includes a full copy of Tracktion, you're ready to start recording, right from day one.Real MIDAS microphone preamplifier technology onboardOne of the things that really makes the Behringer U-Phoria UMC22 stand out is its genuine MIDAS preamplifier technology. Known throughout the world of live sound for delivering ultra clear sound with plenty of headroom, MIDAS preamps are among the most popular mic pres on Earth. What's more, these preamps come standard with +48V phantom power onboard, so you can use your choice of quality condenser microphones with your UMC22. Call your Sweetwater Sales Engineer, and turn this U-Phoria UMC22 USB audio interface into a killer package deal with the perfect microphones and monitors for your complete recording rig.Behringer U-Phoria UMC22 USB Audio Interface Features: Quality 2-channel USB recording interface for your Mac or Windows PC. Combo input with MIDAS preamp lets you plug in any microphone or line-level gear. +48V phantom power lets you use studio condenser microphones. Dedicated instrument-level input accommodates your guitar or bass. Headphone and stereo 1/4\" outputs provide easy monitoring. Includes Tracktion DAW and 150 downloadable instrument/effect plug-ins. Start recording with a Behringer U-Phoria UMC22 USB audio interface! USB Audio Interfaces",
    "url": "sweetwater.com",
    "seller": "Sweetwater",
    "brand": "Behringer",
    "source": "microsoft_commercial_ads",
    "ad_id": "a767622a-e5f8-48f9-b026-e960bb488e96"
}
doc_domain = "electronics"
doc_subdomain = "audio equipment"

# Compute the loss
similarity_loss_fn = SimilarityLoss()
loss = similarity_loss_fn(
    queries=queries,
    original_doc=original_doc,
    rewritten_doc=rewritten_doc,
    doc_domain=doc_domain,
    doc_subdomain=doc_subdomain
)

print("Loss:", loss.item())

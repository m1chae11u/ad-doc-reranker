from trl import RewardTrainer
from transformers import Trainer

class CustomRewardTrainer(RewardTrainer):
    def __init__(self, tokenizer, loss_fn, responses, classified_ads, top_k_docs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.loss_fn = loss_fn
        self.responses = responses
        self.classified_ads = classified_ads
        self.top_k_docs = top_k_docs
        
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = self.tokenizer(
            [f"Original Ad: {ad['title']}\nRewrite the ad:" for ad in raw_ads],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        generated_responses = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        decoded_responses = self.tokenizer.batch_decode(generated_responses, skip_special_tokens=True)
        raw_ads = inputs["original_ads"]

        total_losses = []
        for original, rewritten in zip(raw_ads, decoded_responses):
            
            relevant_queries = [] # queries that have same domain subdomain as document
            for query, q_info in self.responses.items():
                ad_domain = self.classified_ads.get(original['ad_id'],{}).get("domain")
                ad_subdomain = self.classified_ads.get(original['ad_id'],{}).get("subdomain")
                if q_info.get("domain") == ad_domain and q_info.get("subdomain") == ad_subdomain:
                    relevant_queries.append(query)
            
            losses = []
            for query in relevant_queries:
                loss = self.loss_fn(query, original, rewritten, self.top_k_docs)
                losses.append(loss)
            total_losses.append(sum(losses)/len(losses))
        return (sum(total_losses) / len(total_losses))
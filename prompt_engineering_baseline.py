import json
import google.generativeai as genai
import os

def load_api_key() -> str:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "keys.json")
    with open(config_path, 'r') as f:
        return json.load(f)["google_api_key"]

def initialize_gemini():
    api_key = load_api_key()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-pro")

def create_prompt(query: str, ad: str) -> str:
    return f"""You are given an ad and a user query. Rewrite the ad to better match the user query while keeping factual content and tone.

Query: {query}
Original Ad: {ad}
Rewritten Ad:"""


def rewrite_ads(ads: List[Dict], queries: List[Dict], classified_ads: List[Dict], model) -> List[Dict]:
    rewritten_ads = []

# Process the ads
for a, c in zip(ads, classified_ads):
    ad = a.get("text")
    ad_domain = c.get("domain")
    ad_subdomain = c.get("subdomain")

    # print(f"Processing query: {query}, ad: {ad}")  # Check if it's entering the loop

    for query in queries:
        if query.get('domain')==ad_domain and query['subdomain']==ad_subdomain:
            prompt = create_prompt(query['query'], ad)
            print(f"Generated prompt: {prompt}")  # Check the prompt that is being generated

            # Generate the rewritten ad
            response = model.generate_content(prompt)
            print(f"Response: {response.text}")  # Check the response from the model

            rewritten.append(response.text.strip())
        

# Save the updated data to prompt_output.json
with open("prompt_output.json", "w", encoding='utf-8') as f:
    json.dump(rewritten, f, ensure_ascii=False, indent=2)

print("Output saved to prompt_output.json")  # Confirm when it's done

import json
import google.generativeai as genai

# Configure the Gemini model with the API key
genai.configure(api_key="AIzaSyBo9tI8t-MckqNzfQBnf2UOfZKwLI-_0Zc")

# Load input from sampled_ads.json
with open("sampled_ads.json", "r") as f:
    data = json.load(f)

# Define the prompt template
def create_prompt(query, ad):
    return f"""You are given an ad and a user query. Rewrite the ad to better match the user query while keeping factual content and tone.

Query: {query}
Original Ad: {ad}
Rewritten Ad:"""

# Initialize the model
model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# Prepare a list to store rewritten ads
rewritten = []

# Process the ads
for item in data:
    query = item.get("user_query")
    ad = item.get("title")

    print(f"Processing query: {query}, ad: {ad}")  # Check if it's entering the loop

    if query and ad:
        prompt = create_prompt(query, ad)
        print(f"Generated prompt: {prompt}")  # Check the prompt that is being generated

        try:
            # genertae the rewritten ad
            response = model.generate_content(prompt)
            print(f"Response: {response.text}")  # chk the response from the model

            # Save the rewritten ad to the 'rewritten_ad' field
            item["rewritten_ad"] = response.text.strip()
        except Exception as e:
            # In case of an error, save the error message
            print(f"Error: {e}")  # Print the error for debugging
            item["rewritten_ad"] = f"ERROR: {e}"

    rewritten.append(item)

# Save the updated data to prompt_output.json
with open("prompt_output.json", "w") as f:
    json.dump(rewritten, f, indent=2)

print("Output saved to prompt_output.json")  # Confirm when it's done

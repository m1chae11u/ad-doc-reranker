import json

# Load the full JSON data
with open('ds/sampled_data.json', encoding='utf-8') as f:
    data = json.load(f)

# Ensure the input is a list
if not isinstance(data, list):
    raise ValueError("The input JSON must be a list of entries.")

# Split the data
train_data = data[:10000]
test_data = data[-1000:]

# Save to train.json
with open('ds/train_data.json', 'w') as f:
    json.dump(train_data, f, indent=2)

# Save to test.json
with open('ds/test_data.json', 'w') as f:
    json.dump(test_data, f, indent=2)

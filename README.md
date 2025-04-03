# ad-doc-reranker
Reranking Advertisement Documents in RAG for Blackbox LLMs 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Data Processing]
  - [Building the Index]
  - [Using the Retriever]
  - [Running the RAG Interface]
- [Configuration](#configuration)
  - [Embedding Model](#embedding-model)
  - [LLM Settings](#llm-settings)

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/YOURUSERNAME/ad-doc-reranker.git
cd ad-doc-reranker
```

2.	**Create and activate a virtual environment:**

```bash
chmod +x env_setup.sh
./env_setup.sh
conda activate ad_doc_ranker
```
3. **Set up your API keys:**
    
Create a file at configs/keys.json with the following content:

```json
{
    "google_api_key": "YOUR_GEMINI_API_KEY"
}
```

## Usage

First reformat the Microsoft CommercialAdsDataset to the format expected by our RAG system. (json format)
```bash
# Data Processing
python data_processing/convert_microsoft_ads.py --input_dir /path/to/commercial_ads_dataset --output_file commercial_ads.json

# Optional: Sample X number of ads from the dataset for preliminary testing

python data_processing/sample_microsoft_ads.py --input_file /full/path/to/commercial_ads.json --output_file /full/path/to/sampled_ads.json --sample_size 1000

# Build the FAISS Index

python data_processing/build_index.py --input_path /full/path/to/sampled_ads.json --output_dir commercial_ads_faiss_index

# Using the Retriever: run retriever.py to test the index

python retriever.py --query "What are some popular games you can recommend?" --index_dir commercial_ads_faiss_index --full_docs

# Using the RAG Interface: run rag_interface.py to test the index

python rag_interface.py --query "What are some popular games you can recommend?" --index_dir commercial_ads_faiss_index
```

## Configuration

### Embedding Model
By default, the system uses sentence-transformers/all-MiniLM-L6-v2 for document embeddings. You can modify this in:
data_processing/build_index.py (for index creation)
retriever.py (for search)

### LLM Settings
The system uses Google's Gemini 1.5 Flash by default. You can change the model or adjust its parameters in rag_interface.py.
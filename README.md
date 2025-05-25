# Rewrite-to-Rank: Optimizing Ad Visibility via Retrieval-Aware Text Rewriting

This repository contains the implementation for the paper "Rewrite-to-Rank: Optimizing Ad Visibility via Retrieval-Aware Text Rewriting". The system improves advertisement visibility in search and retrieval systems by rewriting ad content using various LLM-based approaches.

**Note**: The PPO LoRA training is implemented in a separate repository: https://anonymous.4open.science/r/ad_ppo_lora-582C

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Data Processing](#data-processing)
  - [Ad Rewriting Models](#ad-rewriting-models)
  - [Metric Evaluation](#metric-evaluation)
- [Configuration](#configuration)
  - [Embedding Model](#embedding-model)
  - [LLM Settings](#llm-settings)

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/YOURUSERNAME/ad-doc-reranker.git
cd ad-doc-reranker
```

2. **Create and activate a virtual environment:**

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

### Data Processing

First, prepare your advertisement dataset and build the necessary indices:

```bash
# Convert Microsoft Commercial Ads Dataset to expected format
python data_processing/convert_microsoft_ads.py --input_dir /path/to/commercial_ads_dataset --output_file commercial_ads.json

# Sample ads for testing (optional)
python data_processing/sample_microsoft_ads.py --input_file commercial_ads.json --output_file sampled_ads.json --sample_size 200

# Build FAISS index for retrieval
python data_processing/build_index.py --input_path sampled_ads.json --output_dir faiss_index_original
```

### Ad Rewriting Models

This repository supports two different approaches for rewriting advertisements to improve their ranking and retrieval performance:

#### 1. Prompt Engineering Baseline
```bash
python prompt_engineering.py --ads_file sampled_ads.json --output_file prompt_rewritten_ads.json
```

#### 2. Supervised Fine-Tuned (SFT) Model
```bash
python using_sft_model.py --ads_file sampled_ads.json --output_file sft_rewritten_ads.json
```

**Note**: For PPO-based ad rewriting, refer to the separate PPO LoRA repository linked above.

### Metric Evaluation

Evaluate the performance improvements using two key metrics:

```bash
# Run comprehensive metric evaluation
python metric_calculations.py
```

The evaluation measures:
- **Inclusion Accuracy Improvement**: How often rewritten ads are included in LLM responses (higher is better, range: -100 to +100)
- **ΔMRR@K**: Change in Mean Reciprocal Rank for retrieval (higher is better, range: -1 to +1)

#### Individual Metric Components

You can also run individual components:

```bash
# Rank documents for retrieval evaluation
python rank_documents.py

# Generate RAG responses for inclusion evaluation
python rag.py --query_file queries.json --index_dir faiss_index --output_file responses.json
```

### Training Your Own Models

#### Create Training Data for PPO
```bash
python create_ppo_data.py --ads_file sampled_ads.json --output_file ppo_training_data.json
```

#### Create Reward Model Data
```bash
python create_reward_data.py --ads_file sampled_ads.json --output_file reward_training_data.json
```

## Configuration

### Embedding Model
By default, the system uses `sentence-transformers/all-MiniLM-L6-v2` for document embeddings. You can modify this in:
- `data_processing/build_index.py` (for index creation)
- `retriever.py` (for search)

### LLM Settings
- **RAG Interface**: Uses Google's Gemini 1.5 Flash by default (configurable in `rag.py`)
- **Ad Rewriting**: Uses LLaMA-3-8B-Instruct variants (SFT/PPO models)
- **Prompt Engineering**: Uses Google Gemini (configurable in `prompt_engineering.py`)

### Model Paths
Update model paths in the respective scripts:
- SFT model: `sft_output/` directory
- PPO model: LLaMA-Factory saves directory
- Base models: Hugging Face model IDs

## Metrics Explanation

- **Inclusion Accuracy Improvement**: Measures the percentage point change in how often your ads appear in generated responses. Positive values mean your rewritten ads are being included more frequently.

- **ΔMRR@K**: Measures the change in ranking position. Positive values mean your ads are ranking higher in search results after rewriting.

Both metrics use a "delta" approach (after - before), so positive numbers indicate improvement.
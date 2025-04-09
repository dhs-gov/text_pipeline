# Text Classification Pipeline

A modular Python framework for processing, analyzing, and classifying text data with LLMs. 

## Features

- **Modular Pipeline Design**: Easy-to-extend processing steps and classifiers
- **YAML Configuration**: Run the pipeline from config files
- **LLM Integration**: Uses Ollama for local LLM inference
- **Testing Framework**: Evaluate different prompts, models, and temperatures on labeled data


### Ollama Setup

This pipeline uses [Ollama](https://ollama.ai/) for local LLM inference:

1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull the Llama model:
   ```bash
   ollama pull llama3.1
   ```
3. Start the Ollama server:
   ```bash
   ollama serve
   ```

## Project Structure

```
text-classification-pipeline/
├── code/
│   ├── process_text.py       # Text processing module
│   ├── analyze_text.py       # Classification module
│   ├── pipeline.py           # Core pipeline functionality
│   └── test_prompts.py       # Testing framework
├── data/
│   └── OnionOrNot.csv        # Sample dataset
├── config/
│   ├── config.yaml           # Main pipeline configuration
│   └── test_config.yaml      # Test configuration
├── results/                  # Main pipeline results (timestamped folders)
│   └── 20250408_093015/      # Example run folder with results
└── tests/
    ├── prompts.json          # Test prompt variations
    └── results/              # Test run results
        └── run_20250408_093015/  # Example test run results
```

## Usage

### Running the Pipeline

Run the full pipeline with a configuration file:

```bash
python code/pipeline.py --config config/config.yaml
```

### Configuration

The pipeline is configured with YAML files:

```yaml
# config.yaml - Main pipeline configuration
input_file: data/OnionOrNot.csv
processed_file: data/processed.json
output_file: results/output.json
text_field: text

# Model configuration
model: llama3.1
api_base: http://localhost:11434/api
temperature: 0.3

# Classifier configuration
classifiers:
  - onion

# Processing options
deduplicate: true
top_n: 10
```

### Running Tests

Test different prompts, models, and temperature settings:

```bash
python code/test_prompts.py --config config/test_config.yaml
```

Test configuration example:

```yaml
# test_config.yaml
test_prompts_file: tests/prompts.json
test_sample_size: 5
models: llama3.1
temperatures:
  - 0.1
  - 0.3
text_field: text
```

## Extending the Pipeline

### Adding a New Processor

1. Create a function in `process_text.py`
2. Register it with the `@register_processor` decorator:

```python
@register_processor("my_processor")
def my_processor(items, custom_param=None):
    """
    My custom processor function.
    
    Args:
        items: List of items to process
        custom_param: Custom parameter
    """
    # Process items here
    return processed_items
```

### Adding a New Classifier

1. Create a new classifier by subclassing `BaseClassifier` in `analyze_text.py`
2. Register it with the `@register_classifier` decorator:

```python
@register_classifier("my_classifier")
class MyClassifier(BaseClassifier):
    def get_system_prompt(self):
        return "Your system prompt here"
        
    def get_tool_spec(self):
        return {
            "type": "function",
            "function": {
                "name": "my_classification_function",
                "description": "My classifier description",
                "parameters": {
                    # Tool parameters spec
                }
            }
        }
```

3. Update your config file to use the new classifier:

```yaml
classifiers:
  - onion
  - my_classifier
```

## Example Workflow

1. Prepare your text data in CSV format with a text column and optional label column
2. Configure `config.yaml` with your input file and settings
3. Run the pipeline to classify your text
4. Review results in the output files and logs
5. Use the testing framework to optimize prompts and parameters
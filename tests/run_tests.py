"""
Testing utility for evaluating different prompts against a small set of examples.
"""
import logging
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import sys
import argparse

# Force-add your code directory manually - keep this if your imports need it
BASE_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = BASE_DIR / "code"
sys.path.insert(0, str(CODE_DIR))

# Import directly from analyze_text module
from analyze_text import (
    BaseClassifier,
    Item,
    Pipeline,
    classify_items,
    CLASSIFIER_REGISTRY
)


# === Flexible Prompt Classifier ===
class PromptVariationClassifier(BaseClassifier):
    def __init__(self, prompt: str, name: str, model: str, api_base: str, timeout: int = 30):
        super().__init__(model=model, api_base=api_base, timeout=timeout)
        self._prompt = prompt
        self.name = name

    def get_system_prompt(self):
        return self._prompt

    def get_tool_spec(self):
        return {
            "type": "function",
            "function": {
                "name": f"classify_with_{self.name}",
                "description": "Classify text using a custom prompt",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "classification": {
                            "type": "string",
                            "enum": ["0", "1"],
                            "description": "1 if satirical, 0 if not"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation for classification"
                        },
                        "satirical_indicators": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Words or phrases indicating satire"
                        }
                    },
                    "required": ["classification", "reasoning", "satirical_indicators"]
                }
            }
        }

    def process_results(self, classification: str, info: Dict):
        indicators = info.get("satirical_indicators", [])
        if isinstance(indicators, str):
            try:
                indicators = json.loads(indicators)
            except json.JSONDecodeError:
                indicators = []
        indicators = [word for word in indicators if isinstance(word, str) and word.strip()]
        return classification, {
            "reasoning": info.get("reasoning", ""),
            "satirical_indicators": indicators
        }


# === Utility functions ===

def load_config(config_path: Path) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_prompts(prompts_file: Path) -> List[Dict]:
    with open(prompts_file, 'r') as f:
        return json.load(f)


def prepare_results_directory() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("tests/results") / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up file logging
    log_file = results_dir / "test_run.log"
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Add file handler to root logger
    logging.getLogger().addHandler(file_handler)
    logging.info(f"Log file: {log_file}")
    
    return results_dir


def create_sample_items(sample_size: int = None) -> List[Item]:
    # Define sample items with their true labels
    items_data = [
        {"id": "1", "text": "Scientists Teach Gorilla To Apologize For What He Did To Jeff", "true_label": "1"},
        {"id": "2", "text": "World Bank Warns of Global Recession Risks", "true_label": "0"},
        {"id": "3", "text": "Man Finally Put In Charge Of Struggling Feminist Movement", "true_label": "1"},
        {"id": "4", "text": "Supreme Court Rules 5-4 To Uphold Gun Rights", "true_label": "0"},
        {"id": "5", "text": "Area Man Passionate Defender Of What He Imagines Constitution To Be", "true_label": "1"},
        {"id": "6", "text": "Sorry, Cancer Patients And Boston Marathon Victims: Indiana's Memories Pizza Raised More Than You On GoFundMe", "true_label": "1"},
    ]
    
    # Create Item objects with the data
    items = [Item(data["id"], text=data["text"], true_label=data["true_label"]) for data in items_data]
    
    return items[:sample_size] if sample_size else items


def register_classifier(classifier: PromptVariationClassifier, classifier_name: str):
    CLASSIFIER_REGISTRY[classifier_name] = type(
        f"{classifier_name}_Cls",
        (PromptVariationClassifier,),
        {"__new__": staticmethod(lambda cls, **kwargs: classifier)}
    )


def run_pipeline(items: List[Item], classifier_name: str, model: str, api_base: str, text_field: str) -> List[Item]:
    pipeline = Pipeline(name=f"test_{classifier_name}")
    # Using only the classification step
    pipeline.add_step(classify_items, field=text_field, model=model, api_base=api_base, classifiers=[classifier_name])
    return pipeline.process(items)

def evaluate_model_results(model_results: List[Dict]) -> Dict:
    total = len(model_results)
    high_priority_total = sum(1 for r in model_results if r["true_label"] == "1")
    true_positives = sum(1 for r in model_results if r["prediction"] == "1" and r["true_label"] == "1")
    flagged_total = sum(1 for r in model_results if r["prediction"] == "1")

    coverage = round((true_positives / high_priority_total) * 100, 1) if high_priority_total else 0.0
    flagged = round((flagged_total / total) * 100, 1) if total else 0.0

    return {
        "total": total,
        "high_priority_total": high_priority_total,
        "true_positives": true_positives,
        "flagged_total": flagged_total,
        "coverage": coverage,
        "flagged": flagged
    }


def evaluate_results(all_results: List[Dict]) -> Dict:
    summary = {
        "recommended_model": all_results[0]["model_id"] if all_results else "",
        "models": []
    }

    # Group results by model_id
    grouped = {}
    for result in all_results:
        key = result["model_id"]
        grouped.setdefault(key, []).append(result)

    # Evaluate each model separately
    for model_id, model_results in grouped.items():
        model_summary = evaluate_model_results(model_results)
        model_summary["name"] = model_id  # Add model name to summary
        summary["models"].append(model_summary)

    return summary


def save_results(results_dir: Path, summary: Dict, detailed: List[Dict]):
    summary_file = results_dir / "summary.json"
    detailed_file = results_dir / "detailed_results.json"

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    with open(detailed_file, 'w') as f:
        json.dump(detailed, f, indent=2)

    logging.info(f"Results saved to {results_dir}")


# === Default hardcoded prompts ===
def get_default_prompts():
    return [
        {
            "name": "Basic",
            "prompt": "You are an expert at detecting satirical news articles. Analyze the text carefully and determine if it's satirical or not."
        },
        {
            "name": "Detailed",
            "prompt": "You are an expert at detecting satirical news articles. Look for exaggeration, irony, absurdity, and humor. Consider both the content and the tone."
        }
    ]


# === Main runner ===
def run_prompt_tests(config_path=None):
    # Set up defaults
    prompts = get_default_prompts()
    models = ["llama3.1"]
    temperatures = [0.1]
    text_field = "text"
    sample_size = None
    api_base = "http://localhost:11434/api"  # Default Ollama API endpoint
    
    # If config path is provided, try to load from it
    if config_path:
        try:
            config_path = Path(config_path)
            config = load_config(config_path)
            
            # Load prompts from file if specified
            if "test_prompts_file" in config:
                prompts_file = Path(config["test_prompts_file"])
                prompts = load_prompts(prompts_file)
            
            # Support both comma-separated string and list formats for models
            if "models" in config:
                models_config = config["models"]
                models = [m.strip() for m in models_config.split(",")] if isinstance(models_config, str) else models_config
            
            # Get other config values
            temperatures = config.get("temperatures", temperatures)
            text_field = config.get("text_field", text_field)
            sample_size = config.get("test_sample_size", sample_size)
            api_base = config.get("api_base", api_base)
            
            logging.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logging.error(f"Error loading config from {config_path}: {e}")
            logging.info("Using default configuration")
    
    results_dir = prepare_results_directory()

    logging.info(f"Testing {len(prompts)} prompt variations, {len(models)} models, {len(temperatures)} temperatures.")
    logging.info(f"Results directory: {results_dir}")

    items = create_sample_items(sample_size)
    all_results = []

    for prompt in prompts:
        prompt_name = prompt["name"]
        system_prompt = prompt["prompt"]

        for model_name in models:
            for temp in temperatures:
                classifier_id = f"{prompt_name}__{model_name}__temp{temp}"
                logging.info(f"Running: {classifier_id}")

                classifier = PromptVariationClassifier(
                    prompt=system_prompt,
                    name=classifier_id,
                    model=model_name,
                    api_base=api_base,
                    timeout=30
                )

                register_classifier(classifier, classifier_id)

                processed_items = run_pipeline(
                    items=items,
                    classifier_name=classifier_id,
                    model=model_name,
                    api_base=api_base,
                    text_field=text_field
                )

                for item in processed_items:
                    all_results.append({
                        "id": item.id,
                        "text": item.get(text_field),
                        "model_id": classifier_id,
                        "prompt_name": prompt_name,
                        "model_name": model_name,
                        "temperature": temp,
                        "prediction": item.get(f"{classifier_id}_result"),
                        "reasoning": item.get(f"{classifier_id}_reasoning"),
                        "indicators": item.get(f"{classifier_id}_satirical_indicators"),
                        "true_label": item.get("true_label")
                    })

    # Evaluate and save
    summary = evaluate_results(all_results)
    save_results(results_dir, summary, all_results)
    
    # Print a summary to console
    print("\nTest Results Summary:")
    print(f"Total models tested: {len(summary['models'])}")
    for model in summary['models']:
        print(f"\nModel: {model['name']}")
        print(f"  Total examples: {model['total']}")
        print(f"  Satire examples: {model['high_priority_total']}")
        print(f"  True positives: {model['true_positives']}")
        print(f"  Coverage: {model['coverage']}%")
        print(f"  Flagged: {model['flagged']}%")


if __name__ == "__main__":
    # Basic logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Run prompt variation tests")
    parser.add_argument("--config", type=str, help="Path to test config file (optional)")
    args = parser.parse_args()

    config_path = args.config
    if config_path and not Path(config_path).exists():
        logging.error(f"Config file not found: {config_path}")
        config_path = None
        
    run_prompt_tests(config_path)
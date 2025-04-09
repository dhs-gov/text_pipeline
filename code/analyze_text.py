"""
Framework for detecting satirical headlines in text.
Uses a class-based classifier with automatic registration and YAML configuration.
"""
import logging
import json
import yaml
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
import requests
import re
import time 


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Define results directory at the project root level
# Get the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up to the project root if we're in the 'code' subdirectory
if os.path.basename(script_dir) == 'code':
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir

# Create results directory in project root
RESULTS_DIR = os.path.join(project_root, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Registry to store processor functions
PROCESSOR_REGISTRY = {}

# Registry to store classifier classes
CLASSIFIER_REGISTRY = {}

def register_processor(name):
    """Decorator to register a processor function"""
    def decorator(func):
        PROCESSOR_REGISTRY[name] = func
        return func
    return decorator

def register_classifier(name):
    """Decorator to register a classifier class"""
    def decorator(cls):
        CLASSIFIER_REGISTRY[name] = cls
        # Store the registration name on the class
        cls.registry_name = name
        return cls
    return decorator

def get_registered_processors():
    """Return all registered processors"""
    return PROCESSOR_REGISTRY.copy()

def get_registered_classifiers():
    """Return all registered classifiers"""
    return CLASSIFIER_REGISTRY.copy()


class BaseClassifier:
    """Base class for all LLM classifiers"""
    
    def __init__(self, model="llama3.1", api_base="http://localhost:11434/api", timeout=30):
        """
        Initialize a classifier.
        
        Args:
            model: LLM model name
            api_base: Base URL for LLM API
            timeout: Request timeout in seconds
        """
        self.model = model
        self.api_base = api_base
        self.timeout = timeout
        # Get the class name for logging
        self.logger = logging.getLogger(f"classifier.{self.__class__.__name__}")
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the classifier.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_system_prompt")
    
    def get_tool_spec(self) -> Dict:
        """
        Get the tool specification for the classifier.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_tool_spec")
    
    def process_results(self, classification: str, info: Dict) -> Tuple[str, Dict]:
        """
        Process the raw classification results.
        Can be overridden by subclasses to provide custom processing.
        
        Args:
            classification: Raw classification value
            info: Raw additional info
            
        Returns:
            Tuple of (processed classification, processed info)
        """
        return classification, info
    
    def get_default_result(self) -> Tuple[str, Dict]:
        """
        Get default result to return on error.
        Should be overridden by subclasses to provide appropriate defaults.
        """
        # Return a generic error result without assuming specific fields
        return "", {"error": "Classification failed"}
    
    def get_user_prompt(self, text: str) -> str:
        """
        Get the user prompt for classification.
        Can be overridden by subclasses to customize the prompt.
        
        Args:
            text: Text to classify
            
        Returns:
            User prompt string
        """
        return f"Analyze this text: \"{text}\""
    
    def classify(self, text: str, max_retries=2) -> Tuple[str, Dict[str, Any]]:
        """
        Classify text using the LLM with tool calling only.
        
        Args:
            text: Text to classify
            max_retries: Maximum number of retry attempts for tool calling
            
        Returns:
            Tuple of (classification result, additional info dict)
        """
        attempt = 0
        
        while attempt <= max_retries:
            try:
                # Get the system prompt and tool spec from the subclass
                system_prompt = self.get_system_prompt()
                tool_spec = self.get_tool_spec()
                user_prompt = self.get_user_prompt(text)
                
                # Create the chat request
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "tools": [tool_spec],
                    "stream": False,
                    "temperature": 0.1 + (attempt * 0.1)  # Slightly increase temperature on retries
                }
                
                # Make the request
                url = f"{self.api_base}/chat"
                self.logger.debug(f"Making request to {url} (attempt {attempt+1}/{max_retries+1})")
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                
                # Process response
                if "message" in result:
                    message = result["message"]
                    tool_calls = message.get("tool_calls", [])
                    
                    # Try tool calls first
                    for tool_call in tool_calls:
                        if tool_call.get("function", {}).get("name") == tool_spec["function"]["name"]:
                            # Parse arguments
                            args = tool_call["function"]["arguments"]
                            if isinstance(args, str):
                                parsed_args = json.loads(args)
                            else:
                                parsed_args = args
                            
                            # Extract classification result
                            prop_names = list(tool_spec["function"]["parameters"]["properties"].keys())
                            classification_field = prop_names[0]
                            classification = parsed_args.get(classification_field)
                            
                            # Remove classification from additional info
                            info = {k: v for k, v in parsed_args.items() if k != classification_field}
                            
                            # Let the subclass process the results
                            return self.process_results(classification, info)
                
                # If we get here, no valid tool calls were found
                self.logger.warning(f"No valid tool calls in response (attempt {attempt+1}/{max_retries+1})")
                
            except Exception as e:
                self.logger.error(f"Classification failed (attempt {attempt+1}/{max_retries+1}): {str(e)}")
            
            # Increment attempt counter
            attempt += 1
            
            # Add a small delay between retries
            if attempt <= max_retries:
                time.sleep(1.0)
        
        # If we get here, all attempts failed
        self.logger.error("All classification attempts failed")
        return self.get_default_result()


# OnionHeadlineClassifier implementation
@register_classifier("onion")
class OnionHeadlineClassifier(BaseClassifier):
    """Classifier for detecting satirical (Onion-like) headlines"""
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the classifier"""
        return """You are a text classifier that determines if a headline is satirical (like The Onion) or real news.
        
        RULES:

    Satirical headlines often use absurdity, exaggeration, or irony
    They may describe implausible scenarios or contain unexpected juxtapositions
    They often use extreme language, hyperbole, or absurd premises
    They sometimes parody the formal tone of real news with ridiculous content
    If the headline seems designed to be humorous rather than informative, it's likely satirical
    IMPORTANT: Just because a headline describes an unusual or absurd event does NOT automatically make it satirical - real news can document genuinely bizarre incidents

    EXAMPLES:

    "Pokémon Go player stabbed, keeps playing" → Not Satirical (0)
    "Job Placement Service Helps Students Who Fail Out Of Dad's Alma Mater Find Work At Dad's Company" → Satirical (1)
    "Idiot Zoo Animal With Zero Predators Still Protective Of Young" → Satirical (1)
    "Point/Counterpoint: Oh, Are The PC Police Here To Arrest Me For Havin' Opinions? vs. Sir, We Are The Regular Police And You Need To Come Out Of That Slide" → Satirical (1)
    "Woman's rejected "8theist" license plate violates First Amendment" → Not Satirical (0)
    "Man Tries to Rob a Bank After Paying $500 to a Wizard to Make Him Invisible" → Not Satirical (0)

    For each headline, determine if it's satirical (1) or not satirical (0), and provide your reasoning along with specific words or phrases that indicate satire (if applicable)."""
    
    def get_tool_spec(self) -> Dict:
        """Get the tool specification for the classifier"""
        return {
            "type": "function",
            "function": {
                "name": "classify_satirical_headline",
                "description": "Classify if a headline is satirical (like The Onion) or not",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "classification": {
                            "type": "string",
                            "enum": ["0", "1"],
                            "description": "Whether the headline is satirical: 1 for satirical, 0 for not satirical"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation for the classification"
                        },
                        "satirical_indicators": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of words or phrases that indicate satire (if applicable)"
                        }
                    },
                    "required": ["classification", "reasoning", "satirical_indicators"]
                }
            }
        }
    
    def get_default_result(self) -> Tuple[str, Dict]:
        """Get default result specific to this classifier"""
        return "0", {"reasoning": "Classification failed", "satirical_indicators": []}
    
    def process_results(self, classification: str, info: Dict) -> Tuple[str, Dict]:
        """Process classification results specific to this classifier"""
        # Ensure satirical_indicators is a list
        indicators = info.get("satirical_indicators", [])
        if not isinstance(indicators, list):
            try:
                # Try to parse it if it's a JSON string
                if isinstance(indicators, str):
                    indicators = json.loads(indicators)
                else:
                    indicators = []
            except:
                indicators = []
        
        # Filter out any non-string or empty items
        indicators = [word for word in indicators if word and isinstance(word, str)]
        
        return classification, {
            "reasoning": info.get("reasoning", ""),
            "satirical_indicators": indicators
        }
    
class Pipeline:
    """A simple processing pipeline that runs processors on items"""
    
    def __init__(self, name="pipeline"):
        self.name = name
        self.steps = []
        self.logger = logging.getLogger(f"pipeline.{name}")
    
    def add_step(self, processor_func, **kwargs):
        """Add a processing step to the pipeline"""
        self.steps.append((processor_func, kwargs))
        return self  # For method chaining
    
    def process(self, items):
        """Run all processing steps on the items"""
        self.logger.info(f"Starting pipeline with {len(self.steps)} steps")
        
        current_items = items
        
        for i, (processor_func, kwargs) in enumerate(self.steps):
            step_name = getattr(processor_func, "__name__", f"Step {i+1}")
            self.logger.info(f"Running step: {step_name} ({i+1}/{len(self.steps)})")
            
            try:
                current_items = processor_func(current_items, **kwargs)
                self.logger.info(f"Step {step_name} completed with {len(current_items)} items")
            except Exception as e:
                self.logger.error(f"Error in step {step_name}: {str(e)}")
                raise
        
        self.logger.info(f"Pipeline completed with {len(current_items)} final items")
        return current_items

class Item:
    """Simple dictionary-like class for storing data"""
    
    def __init__(self, id, **fields):
        self.id = id
        self.fields = {"id": id, **fields}
    
    def get(self, key, default=None):
        """Get a field value"""
        return self.fields.get(key, default)
    
    def set(self, key, value):
        """Set a field value"""
        self.fields[key] = value
        return self
    
    def get_all(self):
        """Get all fields"""
        return self.fields.copy()
    
    def __str__(self):
        return f"Item(id={self.id}, fields={len(self.fields)})"



@register_processor("deduplicate")
def deduplicate_items(items, field="content"):
    """
    Remove duplicate items based on content.
    Simplified version that just looks for exact matches after normalization.
    """
    logger = logging.getLogger("processor.deduplicate")
    logger.info("Starting deduplication")
    
    seen_content = set()
    unique_items = []
    duplicates = 0
    
    for item in items:
        content = item.get(field)
        if content:
            # Normalize content (lowercase and strip whitespace)
            normalized = content.lower().strip()
            
            if normalized not in seen_content:
                seen_content.add(normalized)
                unique_items.append(item)
            else:
                duplicates += 1
        else:
            # Keep items with no content
            unique_items.append(item)
    
    logger.info(f"Deduplication complete. Removed {duplicates} duplicates.")
    return unique_items


# Add JSON loading and saving processors

@register_processor("load_json")
def load_json_file(items, input_file, text_field):
    """
    Load items from a JSON file.
    
    Args:
        items: Initial list of items (usually empty)
        input_file: Path to JSON file
        text_field: Name of the field containing text to classify
    """
    logger = logging.getLogger("processor.load_json")
    logger.info(f"Loading JSON from {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert JSON data to Items
        loaded_items = []
        
        # Check if data is a list or a dict with a data field
        if isinstance(data, dict) and "data" in data:
            data_list = data["data"]
        elif isinstance(data, list):
            data_list = data
        else:
            data_list = [data]  # Single item case
        
        for i, entry in enumerate(data_list):
            # Get ID from entry or generate one
            item_id = entry.get("id", str(i+1))
            
            # Create item with all fields from the entry
            item = Item(item_id, **entry)
            
            # Ensure the text field exists
            if text_field not in entry:
                logger.warning(f"Item {item_id} is missing the required text field '{text_field}'")
                # Add empty field so the item isn't skipped entirely
                item.set(text_field, "")
            
            loaded_items.append(item)
        
        logger.info(f"Loaded {len(loaded_items)} items from {input_file}")
        return loaded_items
        
    except Exception as e:
        logger.error(f"Failed to load JSON from {input_file}: {str(e)}")
        raise
    



# Enhanced function to validate config for analyze_text
def validate_analyze_config(config):
    """Validate the configuration for analyze_text"""
    # Required fields
    required = ['processed_file', 'output_file', 'text_field']
    missing = [field for field in required if field not in config]
    if missing:
        raise ValueError(f"Missing required fields in config: {', '.join(missing)}")
    
    # Check file existence
    if not os.path.exists(config['processed_file']):
        raise FileNotFoundError(f"Input file not found: {config['processed_file']}")
    
    # Check model configuration
    if 'model' in config and not isinstance(config['model'], str):
        raise ValueError("Model must be a string")
    
    # Check classifier configuration
    if 'classifiers' in config:
        if isinstance(config['classifiers'], str):
            config['classifiers'] = [config['classifiers']]
        elif not isinstance(config['classifiers'], list):
            raise ValueError("Classifiers must be a list or string")
        
        # Check if specified classifiers exist
        available = set(get_registered_classifiers().keys())
        requested = set(config['classifiers'])
        not_found = requested - available
        
        if not_found:
            raise ValueError(f"Requested classifiers not found: {', '.join(not_found)}")
    
    return True

# Enhanced results directory setup
def setup_results_directory():
    """Set up the results directory with better organization"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Navigate up to the project root if we're in the 'code' subdirectory
        if os.path.basename(script_dir) == 'code':
            project_root = os.path.dirname(script_dir)
        else:
            project_root = script_dir
        
        # Create results directory in project root
        results_dir = os.path.join(project_root, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Create timestamped subdirectory for current run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(results_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        
        # Configure file handler for logging to this directory
        log_file = os.path.join(run_dir, "pipeline.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        
        logging.info(f"Results will be saved to: {run_dir}")
        return run_dir
    except Exception as e:
        logging.error(f"Failed to set up results directory: {str(e)}")
        # Fall back to the original location
        fallback_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(fallback_dir, exist_ok=True)
        return 
    
def export_results_log(items, model_name="llama3.1", prompt_name="baseline", temperature=0.3):
    """
    Export results to a single JSON file in the results directory.
    Simplified to minimize extra file creation.
    """
    logger = logging.getLogger("export_results")
    logger.info("Exporting results")
    
    try:
        # Use the global RESULTS_DIR that has already been set by set_results_dir()
        results_dir = RESULTS_DIR
        
        # Skip this function if requested by the pipeline
        if not items:
            logger.info("No items to export")
            return None
        
        # Prepare export data with minimal fields
        export_data = []
        for item in items:
            # Get the fields we need
            fields = item.get_all() if hasattr(item, 'get_all') else {}
            
            # Create a simpler result entry
            export_entry = {
                "id": fields.get("id", ""),
                "text": fields.get("text", "")
            }
            
            # Add classifier results only
            for field, value in fields.items():
                if field.endswith("_result"):
                    export_entry[field] = value
            
            export_data.append(export_entry)
        
        # The pipeline already creates a summary.json so we don't need another one
        # We'll just return the data to be used by the pipeline
        
        logger.info(f"Results processed for export: {len(export_data)} items")
        return export_data
        
    except Exception as e:
        logger.error(f"Failed to export results: {str(e)}")
        return None
    
@register_processor("classify")
def classify_items(items, field="text", model="llama3.1", api_base="http://localhost:11434/api", 
                   classifiers=None, temperature=0.3, prompt_name="baseline", export_results=False):
    """
    Classify items using all registered classifiers or only specified ones.
    Now preserves all fields from classifier results.
    
    Args:
        items: List of items to classify
        field: Name of the field containing content to classify
        model: LLM model name
        api_base: Base URL for LLM API
        classifiers: List of classifier names to use (None means use all)
        temperature: Temperature setting for the model
        prompt_name: Name of the prompt used (for logging)
        export_results: Whether to export results after classification
    """
    logger = logging.getLogger("processor.classify")
    logger.info("Starting classification")
    
    # Get all registered classifiers
    registered = get_registered_classifiers()
    
    # Filter classifiers if specific ones are requested
    if classifiers:
        selected = {name: registered[name] for name in classifiers if name in registered}
        if not selected:
            logger.warning(f"None of the requested classifiers {classifiers} are registered, skipping")
            return items
        classifiers_to_use = selected
        logger.info(f"Using {len(selected)} specified classifiers: {', '.join(selected.keys())}")
    else:
        classifiers_to_use = registered
        logger.info(f"Using all {len(registered)} registered classifiers")
    
    if not classifiers_to_use:
        logger.warning("No classifiers available, skipping classification")
        return items
    
    # Instantiate selected classifiers
    classifier_instances = {}
    for name, cls in classifiers_to_use.items():
        classifier_instances[name] = cls(model=model, api_base=api_base)
    
    # Process each item with selected classifiers
    for item in items:
        # Get content to classify
        content = item.get(field)
        if not content:
            logger.warning(f"Item {item.id} has no content in field '{field}'")
            continue
        
        # Run all selected classifiers
        for name, classifier in classifier_instances.items():
            try:
                # Call the classifier
                classification, info = classifier.classify(content)
                
                # Store the main classification result with the classifier name
                result_field = f"{name}_result"
                item.set(result_field, classification)
                
                # Store ALL fields from info with appropriate prefixes
                # This ensures we don't miss any fields like satirical_indicators
                for key, value in info.items():
                    item.set(f"{name}_{key}", value)
                
                logger.info(f"Classification for item {item.id} with {name}: {classification}")
                
            except Exception as e:
                logger.error(f"Classification with {name} failed for item {item.id}: {str(e)}")
                
                # Set default result with error on failure
                default_result, default_info = classifier.get_default_result()
                
                # Store the default result
                item.set(f"{name}_result", default_result)
                
                # Store ALL fields from default_info
                for key, value in default_info.items():
                    item.set(f"{name}_{key}", value)
    
    # The main pipeline handles file creation now, so we don't export here
    return items

def main():
    """Run the pipeline with sample data directly, no config file."""
    # We'll use the current directory as RESULTS_DIR if none has been set yet
    global RESULTS_DIR
    if not RESULTS_DIR or RESULTS_DIR == os.path.join(project_root, "results"):
        RESULTS_DIR = setup_results_directory()
    else:
        logging.info(f"Using existing results directory: {RESULTS_DIR}")
    
    # Create sample items
    items = [
        Item("1", text="Scientists Teach Gorilla To Apologize For What He Did To Jeff"),
        Item("2", text="World Bank Warns of Global Recession Risks"),
        Item("3", text="New Study Finds College Rankings Incredibly Flawed"),
        Item("4", text="Man Finally Put In Charge Of Struggling Feminist Movement"),
        Item("5", text="Area Man Passionate Defender Of What He Imagines Constitution To Be"),
    ]

    # Define parameters
    model = "llama3.1"
    api_base = "http://localhost:11434/api"
    text_field = "text"
    classifiers = ["onion"]
    temperature = 0.3
    prompt_name = "baseline"

    # Set up the pipeline programmatically
    pipeline = Pipeline("manual_test")
    pipeline.add_step(deduplicate_items, field=text_field)
    pipeline.add_step(classify_items, 
                      field=text_field, 
                      model=model, 
                      api_base=api_base, 
                      classifiers=classifiers,
                      temperature=temperature,
                      prompt_name=prompt_name,
                      export_results=True)

    # Run the pipeline
    results = pipeline.process(items)

    # Output results
    print("\nResults:")
    for item in results:
        print(f"Item {item.id}: {item.get(text_field)}")
        print(f"  Satirical: {item.get('onion_result', 'Unknown')}")
        print(f"  Reasoning: {item.get('onion_reasoning', 'N/A')}")
        indicators = item.get("onion_satirical_indicators", [])
        if indicators:
            print(f"  Indicators: {', '.join(indicators)}")
        print()

    return results
def run_from_config(config_file):
    """
    Run the classification pipeline based on a YAML config file.
    Fixed to place output directly in the results directory.
    
    Args:
        config_file: Path to YAML configuration file
    """
    logger = logging.getLogger("run_from_config")
    logger.info(f"Loading configuration from {config_file}")
    
    try:
        # Load configuration
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Validate configuration
        validate_analyze_config(config)
        
        # Extract configuration
        model = config.get("model", "llama3.1")
        api_base = config.get("api_base", "http://localhost:11434/api")
        input_file = config.get("processed_file")
        output_file = config.get("output_file")
        text_field = config.get("text_field", "text")
        classifiers = config.get("classifiers")
        temperature = config.get("temperature", 0.3)
        
        # Get top_n parameter if specified
        top_n = config.get("top_n", None)
        if top_n is not None:
            try:
                top_n = int(top_n)
                logger.info(f"Will process only the first {top_n} items")
            except (ValueError, TypeError):
                logger.warning(f"Invalid top_n value: {top_n}, will process all items")
                top_n = None
        
        # IMPORTANT: Make output_file path directly in RESULTS_DIR
        # Without creating any additional subfolders
        if not os.path.isabs(output_file):
            # Get just the filename part without any directory
            output_filename = os.path.basename(output_file)
            # Use it directly in RESULTS_DIR
            output_file = os.path.join(RESULTS_DIR, output_filename)
            logger.info(f"Output file path resolved to: {output_file}")
        
        # Create pipeline
        pipeline = Pipeline("analyze_pipeline")
        
        # Add steps based on configuration
        pipeline.add_step(load_json_file, input_file=input_file, text_field=text_field)
        
        # Add top_n limit step if specified
        if top_n is not None and top_n > 0:
            pipeline.add_step(limit_items, n=top_n)
            logger.info(f"Added limit step to process only {top_n} items")
        
        # Add classification step
        pipeline.add_step(classify_items, 
                          field=text_field, 
                          model=model, 
                          api_base=api_base, 
                          classifiers=classifiers,
                          temperature=temperature)
        
        # Add save step
        pipeline.add_step(save_json_file, output_file=output_file)
        
        # Run the pipeline
        logger.info("Starting analysis pipeline execution")
        start_time = datetime.now()
        results = pipeline.process([])
        execution_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Analysis pipeline completed in {execution_time:.2f} seconds with {len(results)} items.")
        return results
        
    except Exception as e:
        logger.error(f"Analysis pipeline failed: {str(e)}")
        raise

@register_processor("save_json")
def save_json_file(items, output_file):
    """
    Save items to a JSON file.
    Preserves ALL fields from each item.
    
    Args:
        items: List of items to save
        output_file: Path to output JSON file
    """
    logger = logging.getLogger("processor.save_json")
    logger.info(f"Saving {len(items)} items to {output_file}")
    
    try:
        # Convert items to list of dicts, preserving ALL fields
        output_data = []
        for item in items:
            if hasattr(item, 'get_all'):
                # Get all fields from the item
                item_data = item.get_all()
                output_data.append(item_data)
            else:
                # Fallback for non-Item objects
                logger.warning(f"Item {getattr(item, 'id', 'unknown')} does not have get_all method")
                output_data.append({"id": getattr(item, 'id', 'unknown')})
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(os.path.abspath(output_file))
        if output_dir and not os.path.exists(output_dir):
            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Write to file with all fields preserved
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"data": output_data}, f, indent=2)
        
        logger.info(f"Successfully saved to {output_file} with all fields preserved")
        return items
        
    except Exception as e:
        logger.error(f"Failed to save JSON to {output_file}: {str(e)}")
        raise

# Add a simple processor for limiting number of items
@register_processor("limit")
def limit_items(items, n=10):
    """
    Limit the number of items to process.
    
    Args:
        items: List of items
        n: Maximum number of items to keep
    """
    logger = logging.getLogger("processor.limit")
    
    if n >= len(items):
        logger.info(f"No limiting needed, items count ({len(items)}) <= limit ({n})")
        return items
    
    logger.info(f"Limiting items from {len(items)} to {n}")
    return items[:n]

def set_results_dir(directory_path):
    """
    Set the results directory for the current run.
    Simplified to minimize file creation.
    
    Args:
        directory_path: Path to the results directory
    """
    global RESULTS_DIR
    RESULTS_DIR = directory_path
    logging.info(f"analyze_text: Using results directory: {RESULTS_DIR}")
    
    # No need to set up additional log files - we'll use the pipeline's log
    
    return RESULTS_DIR

if __name__ == "__main__":
    main()
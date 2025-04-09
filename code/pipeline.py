"""
Core pipeline module for text processing and analysis.
Provides shared utilities for directory setup, logging, and configuration.
"""
import logging
import yaml
import json
from datetime import datetime
from pathlib import Path
import os  # Keep for compatibility with existing code

# Import the text processing modules
import process_text
import analyze_text

# Set up root-level logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Only console by default
    ]
)
logger = logging.getLogger("pipeline")


def configure_logging(run_dir):
    """
    Configure logging to write to the run directory.
    Simplified to just create one log file.
    
    Args:
        run_dir: Path to the run directory (pathlib.Path object)
    
    Returns:
        Path to the log file
    """
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Remove any existing file handlers from ALL loggers
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
    
    # Add new file handler for the log file in the run directory
    log_file = run_dir / "pipeline.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Add to the root logger so all modules use it
    root_logger.addHandler(file_handler)
    
    # Log that we've configured logging
    logger.info(f"Logging configured: output to console and {log_file}")
    return log_file


def setup_run_directory():
    """
    Create a timestamped run directory at the project root level.
    Returns the path to the run directory.
    
    Returns:
        pathlib.Path: Path to the created run directory
    """
    # Get the script's directory
    script_dir = Path(__file__).resolve().parent
    
    # Go up one level to the project root (assuming the script is in a 'code' subdirectory)
    project_root = script_dir.parent if script_dir.name == 'code' else script_dir
    
    # Create results directory in project root
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / timestamp
    run_dir.mkdir(exist_ok=True)
    
    logger.info(f"Created run directory: {run_dir}")
    return run_dir


def validate_config(config):
    """
    Validate configuration parameters and provide helpful error messages.
    
    Args:
        config: Dictionary containing configuration parameters
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = {
        'processed_file': 'Path to the processed data file',
        'output_file': 'Path to the output results file',
        'text_field': 'Name of the field containing text to classify'
    }
    
    # Check if process_text step will be run
    processed_file_path = Path(config.get('processed_file', ''))
    if not processed_file_path.exists():
        # If processed file doesn't exist, input_file is required
        required_fields['input_file'] = 'Path to the input data file'
    
    # Check for missing fields
    missing = [field for field in required_fields if field not in config]
    
    if missing:
        error_msg = "Missing required configuration fields:\n"
        for field in missing:
            error_msg += f"  - {field}: {required_fields[field]}\n"
        raise ValueError(error_msg)
    
    # Validate file paths
    if 'input_file' in config:
        input_file_path = Path(config['input_file'])
        if not input_file_path.exists():
            logger.warning(f"Input file not found: {input_file_path}. Will need to be created.")
    
    # Ensure directories exist for output paths
    for field in ['processed_file', 'output_file']:
        if field in config:
            directory = Path(config[field]).parent
            if directory.name and not directory.exists():
                logger.info(f"Creating directory: {directory}")
                directory.mkdir(parents=True, exist_ok=True)
    
    # Additional validations
    if 'model' in config and not isinstance(config['model'], str):
        raise ValueError("Model must be a string")
    
    if 'classifiers' in config:
        if isinstance(config['classifiers'], str):
            # Convert single string to list
            config['classifiers'] = [config['classifiers']]
        elif not isinstance(config['classifiers'], list):
            raise ValueError("Classifiers must be a list or string")
        
        # Check if classifiers exist
        from analyze_text import get_registered_classifiers
        available_classifiers = get_registered_classifiers().keys()
        invalid_classifiers = [c for c in config['classifiers'] if c not in available_classifiers]
        if invalid_classifiers:
            logger.warning(f"Unknown classifiers: {', '.join(invalid_classifiers)}. "
                          f"Available classifiers: {', '.join(available_classifiers)}")
    
    return True


def save_config_copy(config, config_file, run_dir):
    """
    Save a copy of the configuration to the run directory.
    
    Args:
        config: Dictionary containing configuration parameters
        config_file: Path to the original config file
        run_dir: Path to the run directory
        
    Returns:
        Path to the saved config file
    """
    # Create a simple filename for the config copy
    config_copy_path = run_dir / "config.yaml"
    
    # Write the config to file
    with open(config_copy_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    
    logger.info(f"Configuration saved to {config_copy_path}")
    return config_copy_path


def run_pipeline(config_file, continue_on_error=False):
    """
    Run the full text processing and analysis pipeline.
    Simplified to create fewer files - only essential ones.
    
    Args:
        config_file: Path to YAML configuration file
        continue_on_error: Whether to continue execution after errors
    
    Returns:
        Processed items or None on failure
    """
    # Convert config_file to Path object
    config_file = Path(config_file)
    
    # Early logging setup to capture start message in console only
    start_time = datetime.now()
    logger.info(f"Pipeline started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Running pipeline with config: {config_file}")
    
    try:
        # Load config
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Validate configuration
        validate_config(config)
        
        # Set up the run directory FIRST (at project root level)
        run_dir = setup_run_directory()
        
        # Configure logging - just a single pipeline.log file
        configure_logging(run_dir)
        
        # Re-log the startup info to capture it in the log file
        logger.info(f"Pipeline started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Running pipeline with config: {config_file}")
        
        # Check for top_n parameter and log it
        top_n = config.get("top_n")
        if top_n is not None:
            try:
                top_n = int(top_n)
                logger.info(f"Will process only the first {top_n} items")
            except (ValueError, TypeError):
                logger.warning(f"Invalid top_n value: {top_n}, will process all items")
                top_n = None
        
        # Save a copy of the configuration to the run directory
        save_config_copy(config, config_file, run_dir)
        
        # === Step 1: Process Text ===
        logger.info("Step 1: Processing text...")
        process_success = False
        try:
            # Pass the run directory to process_text
            process_text.set_results_dir(run_dir)
            
            # Pass top_n parameter to process_text if specified
            if top_n is not None:
                config['top_n'] = top_n
                
            process_text.run_from_config(config_file)
            logger.info("✅ Text processing completed successfully")
            process_success = True
        except Exception as e:
            logger.error(f"❌ Text processing failed: {str(e)}", exc_info=True)
            if not continue_on_error:
                raise
        
        # Check if the processed file exists even if process_text failed
        processed_file_path = Path(config.get('processed_file', ''))
        if not process_success and 'processed_file' in config:
            if not processed_file_path.exists():
                logger.error(f"Processed file not found: {processed_file_path}")
                if not continue_on_error:
                    raise FileNotFoundError(f"Processed file not found: {processed_file_path}")
                return None
            else:
                logger.warning("Processed file exists despite processing error. Continuing...")
        
        # === Step 2: Analyze Text ===
        logger.info("Step 2: Analyzing text...")
        try:
            # Pass the run directory to analyze_text
            analyze_text.set_results_dir(run_dir)
            
            # Make sure processed_file and output_file paths are in the run directory
            # This ensures we get exactly the files we want where we want them
            if 'processed_file' in config and not Path(config['processed_file']).is_absolute():
                # Update the processed_file path to be in the run directory
                config['processed_file'] = str(run_dir / Path(config['processed_file']).name)
                logger.info(f"Updated processed_file path: {config['processed_file']}")
            
            if 'output_file' in config and not Path(config['output_file']).is_absolute():
                # Update the output_file path to be in the run directory
                config['output_file'] = str(run_dir / Path(config['output_file']).name)
                logger.info(f"Updated output_file path: {config['output_file']}")
                
            # Make sure top_n is passed to analyze_text if specified
            if top_n is not None:
                config['top_n'] = top_n
                
            # Save the updated config
            save_config_copy(config, config_file, run_dir)
            
            results = analyze_text.run_from_config(config_file)
            logger.info("✅ Text analysis completed successfully")
            
            # Create a generic summary that works with any classifier
            if results:
                # Analyze all classifiers used in the results
                classifiers_used = set()
                for item in results:
                    for field in item.get_all().keys():
                        if field.endswith("_result"):
                            classifier_name = field.replace("_result", "")
                            classifiers_used.add(classifier_name)
                
                summary = {
                    "total_items": len(results),
                    "duration_seconds": (datetime.now() - start_time).total_seconds(),
                    "top_n": top_n,  # Include top_n in summary
                    "classifiers": {}
                }
                
                # Generate statistics for each classifier
                for classifier in classifiers_used:
                    # Count all possible values for this classifier
                    value_counts = {}
                    for item in results:
                        result_value = item.get(f"{classifier}_result")
                        if result_value:
                            value_counts[result_value] = value_counts.get(result_value, 0) + 1
                    
                    summary["classifiers"][classifier] = value_counts
                
                # Save summary to the run directory (just one summary.json file)
                summary_file = run_dir / "summary.json"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2)
                
                logger.info(f"Summary saved to {summary_file}")
                logger.info(f"Summary: {json.dumps(summary, indent=2)}")
            
            return results
        except Exception as e:
            logger.error(f"❌ Text analysis failed: {str(e)}", exc_info=True)
            if not continue_on_error:
                raise
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Pipeline completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total duration: {duration.total_seconds():.1f} seconds")
        logger.info(f"All results saved to: {run_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        if not continue_on_error:
            raise
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the full text processing and analysis pipeline")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue pipeline execution on errors")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set verbose logging if requested - do this BEFORE any file handlers
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Resolve the config file path relative to the current working directory
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        exit(1)
    
    logger.info(f"Using config file: {config_path}")
    
    try:
        # Run the pipeline - it will set up proper logging internally
        run_pipeline(config_path, continue_on_error=args.continue_on_error)
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        exit(1)
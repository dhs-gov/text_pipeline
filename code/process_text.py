"""
Framework for preparing text data for the classification pipeline.
Converts CSV data to pipeline-ready JSON format.
Supports YAML configuration or direct execution with sample data.
"""

import pandas as pd
import json
import yaml
import logging
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("data_preparation")


def convert_csv_to_pipeline_json(input_csv_path, output_json_path, text_column="text", label_column="label"):
    """
    Converts a CSV file to the JSON format expected by the classification pipeline.
    
    Args:
        input_csv_path: Path to input CSV file (str or Path)
        output_json_path: Path to output JSON file (str or Path)
        text_column: Name of the column containing text
        label_column: Name of the column containing labels
    """
    # Convert paths to Path objects
    input_csv_path = Path(input_csv_path)
    output_json_path = Path(output_json_path)
    
    try:
        logger.info(f"Loading CSV from {input_csv_path}")
        df = pd.read_csv(input_csv_path)

        # Check if required columns exist
        if text_column not in df.columns or label_column not in df.columns:
            raise ValueError(f"CSV must contain '{text_column}' and '{label_column}' columns.")

        logger.info(f"CSV contains {len(df)} rows")

        # Prepare data format
        data = {"data": df.to_dict(orient="records")}

        # Ensure the output directory exists
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to JSON file
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully converted '{input_csv_path}' to '{output_json_path}'")

    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        raise


def run_from_config(config_file):
    """
    Run the text processing step based on YAML configuration.
    
    Args:
        config_file: Path to YAML configuration file (str or Path)
    """
    # Convert config_file to Path object
    config_file = Path(config_file)
    
    try:
        logger.info(f"Loading configuration from {config_file}")
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        input_csv_path = config.get("input_file")
        output_json_path = config.get("processed_file")
        text_column = config.get("text_field", "text")
        label_column = config.get("label_field", "label")

        if not input_csv_path or not output_json_path:
            raise ValueError("Config must specify 'input_file' and 'processed_file'.")

        # If RESULTS_DIR is set and output_json_path is not an absolute path,
        # make it relative to RESULTS_DIR
        if RESULTS_DIR is not None:
            output_path = Path(output_json_path)
            if not output_path.is_absolute():
                original_path = output_json_path
                output_json_path = str(RESULTS_DIR / output_path.name)
                logger.info(f"Using results directory: {output_json_path} (was: {original_path})")
        
        # Ensure the output directory exists
        output_dir = Path(output_json_path).parent
        if output_dir.name and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

        convert_csv_to_pipeline_json(
            input_csv_path=input_csv_path,
            output_json_path=output_json_path,
            text_column=text_column,
            label_column=label_column
        )

        logger.info(f"✅ Text processing completed. Output written to {output_json_path}")

    except Exception as e:
        logger.error(f"Failed to process configuration: {e}")
        raise


def main():
    """
    Developer-friendly test mode: run directly without config file.
    Generates sample data, saves to a temp directory, and converts to pipeline-ready JSON.
    """
    try:
        # Define sample data
        sample_data = [
            {"text": "Scientists Teach Gorilla To Apologize For What He Did To Jeff", "label": 1},
            {"text": "World Bank Warns of Global Recession Risks", "label": 0},
            {"text": "New Study Finds College Rankings Incredibly Flawed", "label": 0},
            {"text": "Man Finally Put In Charge Of Struggling Feminist Movement", "label": 1},
            {"text": "Area Man Passionate Defender Of What He Imagines Constitution To Be", "label": 1},
        ]

        # Use temp directory for safe test run
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            input_csv_path = tmpdir / "sample_input.csv"
            output_json_path = tmpdir / "sample_output.json"

            # Save sample data to CSV
            logger.info(f"Saving sample data to {input_csv_path}")
            df = pd.DataFrame(sample_data)
            df.to_csv(input_csv_path, index=False)

            # Convert to pipeline-ready JSON
            convert_csv_to_pipeline_json(
                input_csv_path=input_csv_path,
                output_json_path=output_json_path,
                text_column="text",
                label_column="label"
            )

            logger.info(f"✅ Sample data preparation complete! Output JSON located at: {output_json_path}")

            # Optional: Show output path
            with open(output_json_path, "r", encoding="utf-8") as f:
                output_data = json.load(f)
                logger.info(f"Sample output JSON preview: {json.dumps(output_data, indent=2)}")

    except Exception as e:
        logger.error(f"Failed in main(): {e}")
        raise


# Global variable for results directory
RESULTS_DIR = None

def set_results_dir(directory_path):
    """
    Set the results directory for the current run.
    
    Args:
        directory_path: Path to the results directory (str or Path)
    
    Returns:
        Path to the results directory
    """
    global RESULTS_DIR
    RESULTS_DIR = Path(directory_path) if directory_path else None
    logging.info(f"process_text: Using results directory: {RESULTS_DIR}")
    return RESULTS_DIR


if __name__ == "__main__":
    main()
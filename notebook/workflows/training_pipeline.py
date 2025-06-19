"""
Module: training_pipeline.py
============================
This module provides a training pipeline for executing model training tasks for multiple countries.
It reads the training configuration from a YAML file, prepares the necessary file paths and directories,
and then iterates through the list of countries to execute a notebook-based training process for each.
The execution is handled by invoking an external script (notebook_controller.py) via a subprocess call,
embedding parameters such as the country name, training start and end dates, and the base model path.
Functions
---------
    Executes the training pipeline by:
        - Reading the training configuration from a YAML file.
        - Creating output directories for storing trained models.
        - Iterating over the list of countries specified in the configuration.
        - Running the notebook controller on the provided template notebook for each country.
        - Logging the command output and any errors encountered during execution.
Usage
-----
Run this module as a standalone script:
    $ python training_pipeline.py
Notes
-----
- The training configuration file is expected at "../../training configuration/Q1_2025.yml".
- The notebook template is expected at "../production/model_training.ipynb".
- The output of the training process is stored in a directory structure under "../../.ignore_folder/Trained Models".
- This module relies on an external script, "notebook_controller.py", to execute the notebook training process.
"""
import yaml
import os
import subprocess
from datetime import datetime
from pathlib import Path

def run_training_pipeline():
    """
    Main training pipeline that reads configuration and executes model training
    for each country using the notebook controller.
    """
    
    # Read the training configuration
    training_config_path = "../../training configuration/Q1_2025.yml"
    with open(training_config_path, 'r') as file:
        training_config = yaml.safe_load(file)

    print("\nTraining Configuration:")
    print(training_config)
    
    # Extract configuration parameters
    countries = training_config.get('countries', [])
    training_start_date = training_config.get('training_start_date')
    training_end_date = training_config.get('training_end_date')
    base_model_path = training_config.get('base_model_path')
    
    # Define paths
    template_notebook = "../production/model_training.ipynb"
    output_dir = Path("../../.ignore_folder/Trained Models")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training pipeline for {len(countries)} countries...")
    print(f"Training period: {training_start_date} to {training_end_date}")
    print(f"Base model path: {base_model_path}")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Execute training for each country
    for country in countries:
        print(f"\n{'='*50}")
        print(f"Processing country: {country}")
        print(f"{'='*50}")
        
        # Generate output notebook name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create subfolder for the country if it doesn't exist
        country_output_dir = output_dir / country
        country_output_dir.mkdir(parents=True, exist_ok=True)
        output_notebook = country_output_dir / f"executed_{timestamp}.ipynb"
        
        # Prepare the command for notebook_controller.py
        cmd = [
            "python", "notebook_controller.py",
            template_notebook,  # input notebook
            str(output_notebook),  # output notebook
            "-p", f"country={country}",
            "-p", f"training_start_date={training_start_date}",
            "-p", f"training_end_date={training_end_date}",
            "-p", f"base_model_path={base_model_path}"
        ]
        
        try:
            print(f"Executing command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Successfully executed training for {country}")
            print(f"📁 Output saved to: {output_notebook}")
            
            if result.stdout:
                print(f"📋 Output: {result.stdout}")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Error executing training for {country}:")
            print(f"Exit code: {e.returncode}")
            print(f"Error output: {e.stderr}")
            print(f"Standard output: {e.stdout}")
            continue
        except Exception as e:
            print(f"❌ Unexpected error for {country}: {str(e)}")
            continue
    
    print(f"\n{'='*50}")
    print("Training pipeline completed!")
    print(f"Check the output directory: {output_dir.absolute()}")
    print(f"{'='*50}")


if __name__ == "__main__":
    run_training_pipeline()


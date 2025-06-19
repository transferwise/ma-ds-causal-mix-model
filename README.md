# Media Mix Model - Wise

A comprehensive Media Mix Modeling (MMM) solution built with PyMC and designed for analyzing marketing channel effectiveness and attribution. This project provides tools for data preprocessing, model training, exploration, and production deployment of media mix models.

## 🚀 Quick Start

slack channel: #marketing-ds

### Prerequisites

- **Conda**: This project uses conda for environment management. Make sure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd media_mix_model
   ```

2. **Create and activate the conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate wise_env
   ```

3. **Verify the setup**
   ```bash
   python -c "import pymc_marketing; import pymc; print('Setup successful!')"
   ```

## 📊 Getting Started

### 1. Data Exploration

Start by exploring the demo notebooks to understand the data structure and model capabilities:

**PyMC Exploration:**
- Navigate to `notebook/exploration/pymc/demo_base_direct_model.ipynb`
- This notebook demonstrates the basic MMM workflow using PyMC-Marketing

**Meridian Exploration:**
- Check out `notebook/exploration/meridian/demo.ipynb` for alternative modeling approaches

### 2. Data Preparation

Your data should be in CSV format with the following structure:
- Date column (daily data)
- Target variable (conversions, revenue, etc.)
- Media channel impression columns (following the naming convention in `model builders/base_model.yml`)
- Market/country identifier

**Sample data structure:**
```
date,conversions,paidm_soc_fb_excl_cbpr2_imp,paidm_dsp_google_imp,...,market
2022-01-01,1250,50000,25000,...,GBR
2022-01-02,1180,48000,23000,...,GBR
...
```

Place your data file in the `data/csv/` directory.

### 3. Model Configuration

The project uses YAML configuration files to define model parameters:

**Base Model Configuration** (`model builders/base_model.yml`):
- Defines channel columns, target variable, date column
- Sets up adstock and saturation transformations
- Configures model architecture and priors

**Training Configuration** (`training configuration/Q1_2025.yml`):
- Specifies training date range
- Lists countries/markets to include
- References the base model configuration

### 4. Model Training

#### Option A: Interactive Training
Open and run `notebook/production/model_training.ipynb` for interactive model development and training.

#### Option B: Automated Pipeline
Use the training pipeline for automated execution:

```bash
cd notebook/workflows/
python training_pipeline.py
```

This will:
- Read the training configuration
- Execute model training for each specified country
- Save trained models with timestamps

#### Option C: Parameterized Execution
For specific configurations:

```bash
cd notebook/workflows/
python notebook_controller.py \
  --in_nb "../production/model_training.ipynb" \
  --out_nb "output_model.ipynb" \
  --param country=GBR \
  --param training_start_date=2022-01-01 \
  --param training_end_date=2024-12-31 \
  --param base_model_path="../../model builders/base_model.yml"
```

### 5. Model Loading and Analysis

Use `notebook/production/load_model.ipynb` to:
- Load trained models
- Perform posterior analysis
- Generate insights and visualizations
- Calculate channel contributions and ROI

## 📁 Project Structure

```
media_mix_model/
├── data/                          # Data storage
│   ├── csv/                       # CSV data files
│   └── data.sql                   # SQL data queries
├── model builders/                # Model configuration
│   └── base_model.yml            # Base model definition
├── training configuration/        # Training parameters
│   └── Q1_2025.yml               # Example training config
├── notebook/
│   ├── exploration/              # Exploratory analysis
│   │   ├── pymc/                 # PyMC demos and exploration
│   │   └── meridian/             # Meridian framework exploration
│   ├── production/               # Production notebooks
│   │   ├── model_training.ipynb  # Main training notebook
│   │   └── load_model.ipynb      # Model loading and analysis
│   ├── workflows/                # Automation scripts
│   │   ├── notebook_controller.py # Parameterized notebook execution
│   │   └── training_pipeline.py   # Automated training pipeline
│   └── causal discovery/         # Causal analysis tools
├── wise/                         # Core library
│   ├── preprocessing.py          # Data preprocessing utilities
│   ├── utils.py                  # General utilities
│   └── causal.py                 # Causal discovery tools
└── environment.yml               # Conda environment specification
```

## 🔧 Key Features

- **Multi-dimensional modeling**: Support for multiple markets/countries
- **Flexible data preprocessing**: Automated data cleaning and transformation
- **Adstock and saturation**: Built-in media transformations
- **Bayesian inference**: Full uncertainty quantification with PyMC
- **Automated workflows**: Pipeline for batch training and execution
- **Causal discovery**: Tools for understanding causal relationships
- **Visualization**: Rich plotting capabilities for model insights

## 📋 Common Use Cases

1. **Channel Attribution**: Understand which media channels drive the most conversions
2. **Budget Optimization**: Optimize media spend allocation across channels
3. **Incrementality Testing**: Measure the incremental impact of marketing activities
4. **ROI Analysis**: Calculate return on investment for different media channels
5. **Market Comparison**: Compare media effectiveness across different markets

## 🔍 Next Steps

1. **Start with exploration**: Run the demo notebooks to understand the framework
2. **Prepare your data**: Format your data according to the expected structure
3. **Configure your model**: Modify the base model configuration for your use case
4. **Train your model**: Use either interactive or automated training approaches
5. **Analyze results**: Load trained models and generate insights

## 📝 Notes

- Model training can take 30 minutes to several hours depending on data size and model complexity
- Trained models are automatically saved with timestamps in the `.ignore_folder/` directory
- The project supports both local development and production deployment scenarios
- For large datasets, consider using the automated pipeline for efficient batch processing

## 🤝 Contributing

This is a production-ready framework for media mix modeling. Feel free to extend the preprocessing functions, add new model configurations, or enhance the visualization capabilities to meet your specific requirements.

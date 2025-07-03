
# ml-startup-scoring

Machine learning script for analyzing and scoring startup investment opportunities.

## Overview

This project provides a Python-based machine learning tool designed to analyze and score startups for investment purposes. It helps investors evaluate startups by processing various features and generating a comprehensive score reflecting the potential for success and investment worthiness.

## Machine Learning Methods Used

The code uses several regression models to predict and score startup success:

- **Linear Regression:** A simple approach modeling the relationship between features and the target (total revenue).
- **Decision Tree Regressor:** Builds a tree-like model of decisions for regression tasks.
- **K-Nearest Neighbors Regressor (KNN):** Predicts based on the average of the nearest neighbors in the training data.
- **Random Forest Regressor:** An ensemble of multiple decision trees to improve prediction accuracy and control overfitting.

All models are trained and evaluated, and their performances are compared using metrics such as Mean Squared Error (MSE) and R² score. The user can select the model based on their preferred metric.

## Features

- Data preprocessing and cleaning tools (including normalization and encoding)
- Feature engineering for startup data
- Multiple regression model training and evaluation
- Scoring functions for investment decisions
- Visualization of results
- Easy-to-use script and configuration

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/TamarTscholkowsky/ml-startup-scoring.git
    cd ml-startup-scoring
    ```
2. Install required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your startup dataset in the required format (see `data/README.md` for details, if available).
2. Edit configuration settings as needed.
3. Run the main script:
    ```bash
    python py_AI_project.py
    ```
4. View the results and analyze the scores and model visualizations.

## Project Structure

- `py_AI_project.py` - Main script to run analysis and scoring
- `models/` - Saved or pre-trained models (if used)
- `utils/` - Utility scripts and functions

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.

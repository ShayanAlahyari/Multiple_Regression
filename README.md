# Linear Regression on 50 Startups Dataset

This repository contains a Python script for performing linear regression on the 50 Startups dataset. The script includes steps for handling categorical variables, splitting the dataset into training and test sets, training the linear regression model, and visualizing the results.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Script Overview](#script-overview)
- [Dependencies](#dependencies)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ShayanAlahyari/50_Startups_Linear_Regression.git
    ```

2. Navigate to the repository directory:
    ```bash
    cd 50_Startups_Linear_Regression
    ```

3. Install the required dependencies:
    ```bash
    pip install numpy pandas matplotlib scikit-learn
    ```

## Usage

1. Place your dataset file (`50_Startups.csv`) in the repository directory.
2. Run the linear regression script:
    ```bash
    python linear_regression_50_startups.py
    ```

## Script Overview

The `linear_regression_50_startups.py` script performs the following steps:

1. **Importing Libraries**:
    - Imports necessary libraries such as `numpy`, `pandas`, `matplotlib`, and `sklearn` modules.

2. **Creating the Dataset**:
    - Loads the dataset from `50_Startups.csv`.
    - Creates a matrix of features (`x`) and an output column (`y`).

3. **Handling Categorical Predictors**:
    - Uses `OneHotEncoder` within `ColumnTransformer` to encode categorical features.

4. **Splitting the Dataset**:
    - Splits the dataset into training and test sets using `train_test_split`.

5. **Training the Model**:
    - Trains a `LinearRegression` model on the training set.

6. **Predicting and Visualizing**:
    - Predicts values for the training set.
    - Plots residuals for training and test sets.
    - Plots actual vs. predicted values for training and test sets.

## Dependencies

The script requires the following Python libraries:

- numpy
- pandas
- matplotlib
- scikit-learn

You can install these dependencies using `pip`:

```bash
pip install numpy pandas matplotlib scikit-learn

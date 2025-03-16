# Used Car Price Prediction

This project aims to predict the prices of used cars based on various features using machine learning models. The project involves data preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation.

## Project Structure

- `notebooks/used_car_price_prediction.ipynb`: Jupyter notebook containing the code for data preprocessing, model training, and evaluation.
- `dataset/cardekho_imputated.csv`: Dataset containing information about used cars.
- `requirements.txt`: File containing the list of dependencies required for the project.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone <repository_url>
    cd Used-Car-Price-Prediction
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Open the Jupyter notebook:
    ```sh
    jupyter notebook notebooks/used_car_price_prediction.ipynb
    ```

2. Run the cells in the notebook to preprocess the data, train the models, and evaluate their performance.

## Models Used

- Linear Regression
- Ridge Regression
- Lasso Regression
- K-Nearest Neighbors Regressor
- Decision Tree Regressor
- Random Forest Regressor
- AdaBoost Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

## Evaluation Metrics

The models are evaluated using the following metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R2 Score

## Summary

After hyperparameter tuning, the Random Forest Regressor demonstrated strong performance on both the training and testing datasets. The model achieved a high R2 score, indicating a good fit to the data, and relatively low error metrics (RMSE and MAE).

The XGBoost Regressor also performed well, particularly on the training set, but showed slightly higher error metrics on the testing set compared to the Random Forest Regressor. This suggests that while XGBoost is a powerful model, the Random Forest Regressor may be better suited for this particular dataset.

Overall, the Random Forest Regressor appears to be the more effective model for predicting used car prices based on the given features.

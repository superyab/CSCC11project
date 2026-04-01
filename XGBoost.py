from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from helpers import convert_to_numeric, draw_importance, inject_missing_values
from sklearn.model_selection import GridSearchCV

DEVICE = 'cpu'
MISSING_RATE = 0.05

def train_model(X_train, y_train) -> xgb.XGBRegressor:
    param_grid = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
    }
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=69, device=DEVICE)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def main():
    # Load the data
    df = pd.read_csv("./data/Medical_insurance.csv")
    y = df["charges"]
    X = df.drop("charges", axis=1)
    X = convert_to_numeric(X)
    X = inject_missing_values(X, missing_rate=MISSING_RATE)
    print(f"injected missing values rate: {MISSING_RATE}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

    # Train the model
    best_model = train_model(X_train, y_train)

    # Predict the test set
    y_predicted = best_model.predict(X_test)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
    r2 = r2_score(y_test, y_predicted)
    print(f"Mean Squared Error: {rmse}")
    print(f"R2 Score: {r2}")
    # print(f"Best Parameters: {best_model.get_params()}")

    draw_importance(best_model)

if __name__ == "__main__":
    main()


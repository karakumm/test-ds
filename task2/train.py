import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split


def load_data(path):
    """
    Load and prepare training data
    """
    data = pd.read_csv(path)
    data.drop(columns=['8'], inplace=True)

    X = data.drop(columns=['target'])
    y = data['target']
    return X,y 


def train_model(data, out):
    X, y = load_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_dist = {
        'n_estimators': [100, 300, 400],
        'max_depth': [None, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    rf = RandomForestRegressor(random_state=42)
    folds = KFold(n_splits=5, shuffle=True, random_state=100)
    
    model = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=100,
        cv=folds,
        random_state=42,
        scoring='neg_root_mean_squared_error',
        verbose=1
    )
    
    model.fit(X_train, y_train)
    best_model = model.best_estimator_

    print("Best parameters:", model.best_params_)
    print("Best RMSE:", -model.best_score_)

    val_pred = best_model.predict(X_test)
    val_rmse = root_mean_squared_error(val_pred, y_test)
    print("Validation RMSE:", val_rmse)
    
    with open(out, 'wb') as f:
        pickle.dump(best_model, f)
    
    return best_model


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python train.py <data_path> <output_model_path>"
    data_path = sys.argv[1]
    output_model_path = sys.argv[2]
    train_model(data_path, output_model_path)
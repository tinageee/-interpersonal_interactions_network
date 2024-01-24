'''
run the script after getting the degree data
since mmy labtop is M1, I use sklearn.neural_network to run the NN
'''

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd


from sklearn.model_selection import GridSearchCV
import numpy as np




# find the optimal parameters for the gradient boosting machine

# Define parameter grid
# Create a dictionary of parameters to test
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [2,3,4],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['sqrt', 'log2', None, 4]
}

# Initialize the GBM model
gbm = GradientBoostingClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit GridSearchCV, for both feature sets
# grid_search.fit(features_set1, target)
grid_search.fit(features_set2, target)

# Best parameters
print("Best parameters:", grid_search.best_params_)

# feature set 1:
# Best parameters: {'learning_rate': 0.01, 'max_depth': 3, 'max_features': None, 'n_estimators': 50, 'subsample': 1.0}
# feature set 2:
# Best parameters: {'learning_rate': 0.05, 'max_depth': 4, 'max_features': None, 'n_estimators': 150, 'subsample': 1.0}

# fro consistant results
# Best parameters: {'learning_rate': 0.05, 'max_depth': 4, 'max_features': None, 'n_estimators': 150, 'subsample': 1.0}

# find the optimal parameters for the NN machine


def perform_grid_search_sklearn(X, y):
    # Ensure X is a numpy array
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X = X.values
    X = X.astype(np.float32)

    # Encode y if it's categorical
    if isinstance(y, (pd.DataFrame, pd.Series)) or np.issubdtype(y.dtype, np.object_):
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))

    # Initialize MLPClassifier
    mlp = MLPClassifier(max_iter=1000)  # Increased max_iter for convergence

    # Define parameter grid
    param_grid = {
        'hidden_layer_sizes': [(5,), (8,), (10,), (5, 5), (8, 4)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.005, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'batch_size': [16, 32],
    }

    # Perform Grid Search
    grid = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_result = grid.fit(X, y)
    return grid_result.best_params_, grid_result.best_score_, grid_result.best_estimator_

# Example Usage:
best_params, best_score, best_model = perform_grid_search_sklearn(features_set2, target)
print("Best Parameters:", best_params)
print("Best Score:", best_score)

#Best Parameters: {'activation': 'relu', 'alpha': 0.001, 'batch_size': 32, 'hidden_layer_sizes': (8,), 'learning_rate': 'constant', 'solver': 'adam'}

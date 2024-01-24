
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

def predict_roles_based_on_score(game_nodes, score_attr):
    '''
    predict the roles based on the ranking scores
    This function is designed to predict roles (such as 'spy' or 'villager') for players in a game,
    based on a specified scoring attribute.
    The prediction is done per game, and the roles are assigned according to the scores, with the lowest scores typically getting the 'spy' role.
    The function handles NaN values in the scores by assigning 'NA' as the predicted role for those cases.
    :param game_nodes:  DataFrame containing the nodes for each game
    :param score_attr:  Attribute to use for scoring
    :return:    DataFrame with predicted roles
    '''
    # Loop over each unique game
    for game in game_nodes['game_name'].unique():

        # Filter the DataFrame for the current game
        current_game_nodes = game_nodes[game_nodes['game_name'] == game]

        # Count the number of spies
        num_spies = sum(current_game_nodes['Game_Role'] == 'Spy')

        # Sort nodes by the specified score
        sorted_nodes = current_game_nodes.sort_values(by=score_attr)

        # Determine the cut-off rank for spies
        spy_rank_cutoff = num_spies

        # Update game_nodes DataFrame with predicted roles
        for _, row in sorted_nodes.iterrows():
            player_number = row['Player_Number']
            # Check if the score is NaN and assign roles
            if pd.isna(row[score_attr]):
                predicted_role = pd.NA
            else:
                predicted_role = 'Spy' if spy_rank_cutoff > 0 else 'Villager'
                spy_rank_cutoff -= 1

            # Update the role in game_nodes
            condition = (game_nodes['game_name'] == game) & (game_nodes['Player_Number'] == player_number)
            game_nodes.loc[condition, f'{score_attr}_pred'] = predicted_role

    return game_nodes


def evaluate_performance(metric_name, actual, predicted):
    # Combine actual and predicted into a temporary DataFrame
    temp_df = pd.DataFrame({'Actual': actual, 'Predicted': predicted})

    # Drop rows with NaN in either column
    temp_df = temp_df.dropna(subset=['Actual', 'Predicted'])

    # Convert actual and predicted to binary
    temp_df['Actual'] = temp_df['Actual'].map({'Spy': 1, 'Villager': 0})
    temp_df['Predicted'] = temp_df['Predicted'].map({'Spy': 1, 'Villager': 0})

    # Recalculate the actual and predicted without NaN values
    actual_no_na = temp_df['Actual']
    predicted_no_na = temp_df['Predicted']

    # Calculate metrics
    accuracy = accuracy_score(actual_no_na, predicted_no_na)
    precision = precision_score(actual_no_na, predicted_no_na, average='binary')
    recall = recall_score(actual_no_na, predicted_no_na, average='binary')
    f1 = f1_score(actual_no_na, predicted_no_na, average='binary')

    print(f"{metric_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")


def perform_custom_cross_validation(model, features, target):
    """
    Perform 5-fold cross-validation and calculate accuracy, precision, recall, and F1 score.

    Args:
    model: The machine learning model to evaluate.
    features: Features dataset.
    target: Target dataset.

    Returns:
    Dictionary with mean of accuracy, precision, recall, and F1 score across all folds.
    """

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(target)
    spy_label = le.transform(['Spy'])[0]  # Get the encoded label for 'Spy'

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Lists to store metrics for each fold
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    # Perform cross-validation
    for train_index, test_index in skf.split(features, y_encoded):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        # Fit the model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate metrics
        accuracies.append(accuracy_score(y_test, predictions))
        precisions.append(precision_score(y_test, predictions, pos_label=spy_label))
        recalls.append(recall_score(y_test, predictions, pos_label=spy_label))
        f1_scores.append(f1_score(y_test, predictions, pos_label=spy_label))

    # Calculate mean of each metric
    results = {
        'mean_accuracy': np.mean(accuracies),
        'mean_precision': np.mean(precisions),
        'mean_recall': np.mean(recalls),
        'mean_f1_score': np.mean(f1_scores)
    }

    return results


# read the nodes infomration from the csv file
game_nodes = pd.read_csv(
    '/Users/saiyingge/Coding Projects/PyCharmProjects/NetworkProject/Data/all_nodes_W_rankings.csv')


# calculate the prediction based on the ranking scores
# create column to save for prediction
game_nodes = predict_roles_based_on_score(game_nodes, 'receivedTrust')
game_nodes = predict_roles_based_on_score(game_nodes, 'prestige')
game_nodes = predict_roles_based_on_score(game_nodes, 'hits')
game_nodes = predict_roles_based_on_score(game_nodes, 'pageRank')



# TODO: check the na's
# rows with missing values
game_nodes[game_nodes.isnull().any(axis=1)]
# percentage of missing values
game_nodes.isnull().sum() / len(game_nodes)
# drop any rows with missing values
game_nodes = game_nodes.dropna()

# evaluate the performance of the models
# the baseline is the actual roles: spy/total number of players
baseline = 1 - game_nodes['Game_Role'].value_counts()['Spy'] / len(game_nodes)
print(f'Baseline: {baseline}')
# receivedTrust
evaluate_performance('receivedTrust', game_nodes['Game_Role'], game_nodes['receivedTrust_pred'])
# prestige_scores
evaluate_performance('prestige_scores', game_nodes['Game_Role'], game_nodes['prestige_pred'])
# hits_scores
evaluate_performance('hits_scores', game_nodes['Game_Role'], game_nodes['hits_pred'])
# pageRank
evaluate_performance('pageRank', game_nodes['Game_Role'], game_nodes['pageRank_pred'])

# implement machine learning models to predict the roles

game_nodes.loc[:, 'Spy'] = (game_nodes['Game_Role'] == 'Spy').astype(int)
target = game_nodes['Game_Role']

# Best parameters from grid search for MLPClassifier
best_mlp_params = {
    'activation': 'relu',
    'alpha': 0.001,
    'batch_size': 32,
    'hidden_layer_sizes': (8,),
    'learning_rate': 'constant',
    'solver': 'adam',
    'max_iter': 1000  # Adjust based on convergence behavior
}

# Initialize MLPClassifier with the best parameters
mlp_classifier = MLPClassifier(**best_mlp_params)

# List of models
models = [
    {"name": "Logistic Regression", "model": LogisticRegression()},
    {"name": "Gradient Boosting Machine", "model": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth =4, max_features=None, subsample=1.0,
                                                                              random_state=42)},
    {"name": "Neural Network", "model": mlp_classifier}
]

# for node attributes only
game_nodes.loc[:, 'GameExperience'] = (game_nodes['play_b4'] == 'yes').astype(int)
game_nodes.loc[:, 'NativeEngSpeaker'] = (game_nodes['Eng_nativ'] == 'native speaker').astype(int)
game_nodes.loc[:, 'HomogeneousGroupCulture'] = (game_nodes['homogeneous'] == 'Yes').astype(int)
game_nodes.loc[:, 'Male'] = (game_nodes['sex'] == 'Male').astype(int)


features_set1 = game_nodes[['GameExperience','NativeEngSpeaker','HomogeneousGroupCulture','Male']]

for model_info in models:
    model = model_info["model"]
    model_name = model_info["name"]
    cv_results = perform_custom_cross_validation(model, features_set1, target)
    print(f"Results for {model_name}:")
    print(cv_results)




# for node ranking sore and other player and game attributes

features_set2 = game_nodes[['receivedTrust', 'prestige', 'hits','pageRank',
                            'GameExperience','NativeEngSpeaker','HomogeneousGroupCulture','Male']]
for model_info in models:
    model = model_info["model"]
    model_name = model_info["name"]
    cv_results = perform_custom_cross_validation(model, features_set2, target)
    print(f"Results for {model_name}:")
    print(cv_results)

#todo: add plot



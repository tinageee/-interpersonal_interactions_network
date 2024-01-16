'''
calculate_degrees(network) takes a network as input and returns the positive and negative in- and out-degrees of each node in the network.
run the function in the loop to get the degrees for each game
use regression to see if the degrees are correlated with game roles for each game
'''

import networkx as nx
import os
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer

def calculate_degrees(network):
    pos_in_degree = {}
    neg_in_degree = {}
    pos_out_degree = {}
    neg_out_degree = {}

    for node in network.nodes():
        pos_in_degree[node] = sum(1 for _, _, data in network.in_edges(node, data=True) if data['sign'] > 0)
        neg_in_degree[node] = sum(1 for _, _, data in network.in_edges(node, data=True) if data['sign'] < 0)
        pos_out_degree[node] = sum(1 for _, _, data in network.out_edges(node, data=True) if data['sign'] > 0)
        neg_out_degree[node] = sum(1 for _, _, data in network.out_edges(node, data=True) if data['sign'] < 0)

    return pos_in_degree, neg_in_degree, pos_out_degree, neg_out_degree

# read the networks in all the folders
Code_dir = '/Users/saiyingge/Coding Projects/PyCharmProjects/NetworkProject/'

# get the player(nodes) information
game_nodes = pd.read_csv(Code_dir + 'Data/all_nodes.csv')

#get game names
games = game_nodes['game_name'].unique()

# read the networks by game
for game in games:
    network = nx.read_graphml(Code_dir + 'Data/Networks/' + game + '.graphml')
    # Example usage within your loop
    pos_in_degree, neg_in_degree, pos_out_degree, neg_out_degree = calculate_degrees(network)

    # merging degree data with node attributes when the game name and player_number matches
    for node in network.nodes():
        condition = (game_nodes['Player_Number'] == int(node)) & (game_nodes['game_name'] == game)

        game_nodes.loc[condition, 'pos_in_degree'] = pos_in_degree.get(node, 0)
        game_nodes.loc[condition, 'neg_in_degree'] = neg_in_degree.get(node, 0)
        game_nodes.loc[condition, 'pos_out_degree'] = pos_out_degree.get(node, 0)
        game_nodes.loc[condition, 'neg_out_degree'] = neg_out_degree.get(node, 0)




#OLS regression with degree as dependent variable and other attributes as independent variables

# Fit the models
model_pos_in_fitted = smf.ols('pos_in_degree ~ Game_Role + sex + play_b4 + homogeneous + WinLose', data=game_nodes).fit()
model_neg_in_fitted = smf.ols('neg_in_degree ~ Game_Role + sex + play_b4 + homogeneous + WinLose', data=game_nodes).fit()
model_pos_out_fitted = smf.ols('pos_out_degree ~ Game_Role + sex + play_b4 + homogeneous + WinLose', data=game_nodes).fit()
model_neg_out_fitted = smf.ols('neg_out_degree ~ Game_Role + sex + play_b4 + homogeneous + WinLose', data=game_nodes).fit()

# Create a list of fitted models
fitted_models = [model_pos_in_fitted, model_neg_in_fitted, model_pos_out_fitted, model_neg_out_fitted]

# Use Stargazer to format the table
stargazer = Stargazer(fitted_models)

# Configure the stargazer settings (optional)
stargazer.title("Degree Regression Results")
stargazer.custom_columns(["Positive In-Degree", "Negative In-Degree", "Positive Out-Degree", "Negative Out-Degree"], [1, 1, 1, 1])

#todo:
#need to fix the column names
# check the fixed effects
stargazer.rename_covariates({"Game_Role": "Role", "sex": "Gender",  "play_b4": "Played Before", "WinLose": "Game Outcome", "homogeneous": "Game Culture"})
stargazer.add_line("Game Fixed Effects", ["Yes", "Yes", "Yes", "Yes"])

html = stargazer.render_html()
with open("degree_analysis_results.html", "w") as f:
    f.write(html)

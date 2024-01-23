'''
calculate_degrees(network) takes a network as input and returns the positive and negative in- and out-degrees of each node in the network.
run the function in the loop to get the degrees for each game
use regression to see if the degrees are correlated with game roles for each game
'''

import networkx as nx
import pandas as pd
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer
import numpy as np

# show all columns
pd.set_option('display.max_columns', None)


def calculate_degrees(network):
    '''
    calculate the positive and negative in- and out-degrees of each node in the network
    :param network:     multi-directed network
    :return:    pos_in_degree, neg_in_degree, pos_out_degree, neg_out_degree
    '''
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

# get game names
games = game_nodes['game_name'].unique()

# read the networks by game and calculate the degrees
for game in games:
    try:
        network = nx.read_graphml(Code_dir + 'Data/Networks/' + game + '.graphml')

        pos_in_degree, neg_in_degree, pos_out_degree, neg_out_degree = calculate_degrees(network)

        # merging degree data with node attributes when the game name and player_number matches
        for node in network.nodes():
            condition = (game_nodes['Player_Number'] == int(node)) & (game_nodes['game_name'] == game)

            game_nodes.loc[condition, 'pos_in_degree'] = pos_in_degree.get(node, 0)
            game_nodes.loc[condition, 'neg_in_degree'] = neg_in_degree.get(node, 0)
            game_nodes.loc[condition, 'pos_out_degree'] = pos_out_degree.get(node, 0)
            game_nodes.loc[condition, 'neg_out_degree'] = neg_out_degree.get(node, 0)
    except:
        print(f'Error processing {game}.graphml')
        continue

# todo: check the na's


# OLS regression with degree as dependent variable and other attributes as independent variables
# rows with missing values
game_nodes[game_nodes.isnull().any(axis=1)]
# percentage of missing values
game_nodes.isnull().sum() / len(game_nodes)
# drop any rows with missing values
game_nodes = game_nodes.dropna()

# convert the following variables to categorical variables, and order them

# Set 'Spy' and 'SpyWin' as the reference categories

game_nodes['Spy'] = (game_nodes['Game_Role'] == 'Spy').astype(int)
game_nodes['SpyWin'] = (game_nodes['game_result'] == 'SpyWin').astype(int)

game_nodes['GameExperience'] = (game_nodes['play_b4'] == 'yes').astype(int)
game_nodes['NativeEngSpeaker'] = (game_nodes['Eng_nativ'] == 'native speaker').astype(int)
game_nodes['HomogeneousGroupCulture'] = (game_nodes['homogeneous'] == 'Yes').astype(int)
game_nodes['Male'] = (game_nodes['sex'] == 'Male').astype(int)

# Fit the models
# todo: check if the model still given warning with  convergence issue, may need to change the method


model_pos_in_fitted = smf.mixedlm(
    "pos_in_degree ~  Spy+ SpyWin+Male+GameExperience+NativeEngSpeaker+ HomogeneousGroupCulture",
    data=game_nodes,
    groups=game_nodes['game_name']).fit()

model_neg_in_fitted = smf.mixedlm(
    "neg_in_degree ~ Spy+ SpyWin+Male+GameExperience+NativeEngSpeaker+ HomogeneousGroupCulture",
    data=game_nodes,
    groups=game_nodes['game_name']).fit()

model_pos_out_fitted = smf.mixedlm(
    "pos_out_degree ~ Spy+ SpyWin+Male+GameExperience+NativeEngSpeaker+ HomogeneousGroupCulture",
    data=game_nodes,
    groups=game_nodes['game_name']).fit()

model_neg_out_fitted = smf.mixedlm(
    "neg_out_degree ~ Spy+ SpyWin+Male+GameExperience+NativeEngSpeaker+ HomogeneousGroupCulture",
    data=game_nodes,
    groups=game_nodes['game_name']).fit()

# Create a list of fitted models
fitted_models = [model_pos_in_fitted, model_neg_in_fitted, model_pos_out_fitted, model_neg_out_fitted]

# Use Stargazer to format the table
stargazer = Stargazer(fitted_models)

# #todo: need to change the columns name add the hypothesis and support or not
# Configure the stargazer settings (optional)
stargazer.title("Degree Regression Results")
stargazer.custom_columns(["Positive In-Degree", "Negative In-Degree", "Positive Out-Degree", "Negative Out-Degree"],
                         [1, 1, 1, 1])

# change the variable name with stargazer table also, change the order of the variables
stargazer.rename_covariates({'Group Var': 'Group Effect'})
stargazer.covariate_order(
    ['Spy', 'SpyWin', 'Male', 'NativeEngSpeaker', 'GameExperience', 'HomogeneousGroupCulture', 'Group Var',
     'Intercept'])

stargazer.add_line("Hypothesis", ["H?", "H?", "H?", "H?"])
stargazer.add_line("Support or Not", ["?", "?", "?", "?"])

# print(stargazer.render_latex())

html = stargazer.render_html()
with open(Code_dir + "Data/Analysis_Results/degree_analysis_results.html", "w") as f: f.write(html)

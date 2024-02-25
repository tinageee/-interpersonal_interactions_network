'''
calculate_degrees(network) takes a network as input and returns the positive and negative in- and out-degrees of each node in the network.
run the function in the loop to get the degrees for each game
calculate_node_connection(network, node) takes a network and a node as input and returns the trust and distrust outdegrees towards Villager and Spy for the specified node.

merge the degree data with node attributes when the game name and player_number matches
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

def calculate_node_connection(network, node):
    # Initialize dictionaries to store the specific node's trust and distrust outdegrees towards Villager and Spy
    trust_outdegree = {'Villager': 0, 'Spy': 0}
    distrust_outdegree = {'Villager': 0, 'Spy': 0}

    # Ensure the node exists in the network
    if node not in network:
        print(f"Node {node} not found in the network.")
        return trust_outdegree, distrust_outdegree

    # Iterate through edges originating from the specified node to count trust and distrust towards each role
    for u, v, data in network.edges(node, data=True):
        # Only proceed if the current edge starts from the specified node
        if u == node:
            role_v = network.nodes[v]['role']

            if data['sign'] == 1:  # Trust
                trust_outdegree[role_v] += 1
            elif data['sign'] == -1:  # Distrust
                distrust_outdegree[role_v] += 1

    return trust_outdegree, distrust_outdegree

# read the networks in all the folders
Code_dir = '/Users/saiyingge/Coding Projects/PyCharmProjects/NetworkProject/'

# get the player(nodes) information
game_nodes = pd.read_csv(Code_dir + 'Data/all_nodes.csv')

# get game names
games = game_nodes['game_name'].unique()

count = 0
# read the networks by game and calculate the degrees
for game in games:
    #game='018AZ'
    try:
        network = nx.read_graphml(Code_dir + 'Data/Networks/' + game + '.graphml')

        pos_in_degree, neg_in_degree, pos_out_degree, neg_out_degree = calculate_degrees(network)

        # merging degree data with node attributes when the game name and player_number matches
        for node in network.nodes():
            # Calculate the trust and distrust outdegrees towards Villager and Spy for each node
            trust_outdegree, distrust_outdegree = calculate_node_connection(network, node)

            condition = (game_nodes['Player_Number'] == int(node)) & (game_nodes['game_name'] == game)

            game_nodes.loc[condition, 'pos_in_degree'] = pos_in_degree.get(node, 0)
            game_nodes.loc[condition, 'neg_in_degree'] = neg_in_degree.get(node, 0)
            game_nodes.loc[condition, 'pos_out_degree'] = pos_out_degree.get(node, 0)
            game_nodes.loc[condition, 'neg_out_degree'] = neg_out_degree.get(node, 0)

            game_nodes.loc[condition, 'trust_outdegree_villager'] = trust_outdegree['Villager']
            game_nodes.loc[condition, 'trust_outdegree_spy'] = trust_outdegree['Spy']
            game_nodes.loc[condition, 'distrust_outdegree_villager'] = distrust_outdegree['Villager']
            game_nodes.loc[condition, 'distrust_outdegree_spy'] = distrust_outdegree['Spy']

        count += 1
    except:
        print(f'Error processing {game}.graphml')
        continue

print(f'Finished processing {count} files')

# who the row that trust out degree sum not equal to pos out degree
game_nodes[game_nodes['trust_outdegree_villager'] + game_nodes['trust_outdegree_spy'] != game_nodes['pos_out_degree']]
# check distrust
game_nodes[game_nodes['distrust_outdegree_villager'] + game_nodes['distrust_outdegree_spy'] != game_nodes['neg_out_degree']]

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
#
# #no interact facts
# model_pos_in_fitted = smf.mixedlm(
#     "pos_in_degree ~  Spy+ SpyWin+Male+GameExperience+NativeEngSpeaker+ HomogeneousGroupCulture",
#     data=game_nodes,
#     groups=game_nodes['game_name']).fit()
#
# model_neg_in_fitted = smf.mixedlm(
#     "neg_in_degree ~ Spy+ SpyWin+Male+GameExperience+NativeEngSpeaker+ HomogeneousGroupCulture",
#     data=game_nodes,
#     groups=game_nodes['game_name']).fit()
#
# model_pos_out_fitted = smf.mixedlm(
#     "pos_out_degree ~ Spy+ SpyWin+Male+GameExperience+NativeEngSpeaker+ HomogeneousGroupCulture",
#     data=game_nodes,
#     groups=game_nodes['game_name']).fit()
#
# model_neg_out_fitted = smf.mixedlm(
#     "neg_out_degree ~ Spy+ SpyWin+Male+GameExperience+NativeEngSpeaker+ HomogeneousGroupCulture",
#     data=game_nodes,
#     groups=game_nodes['game_name']).fit()
#
# # Create a list of fitted models
# fitted_models = [model_pos_in_fitted, model_neg_in_fitted, model_pos_out_fitted, model_neg_out_fitted]
#
# # Use Stargazer to format the table
# stargazer = Stargazer(fitted_models)
#

# # Configure the stargazer settings (optional)
# stargazer.title("Degree Regression Results")
# stargazer.custom_columns(["Positive In-Degree", "Negative In-Degree", "Positive Out-Degree", "Negative Out-Degree"],
#                          [1, 1, 1, 1])
#
# # change the variable name with stargazer table also, change the order of the variables
# stargazer.rename_covariates({'Group Var': 'Group Effect'})
# stargazer.covariate_order(
#     ['Spy', 'SpyWin', 'Male', 'NativeEngSpeaker', 'GameExperience', 'HomogeneousGroupCulture', 'Group Var',
#      'Intercept'])
#
# stargazer.add_line("Hypothesis", ["H?", "H?", "H?", "H?"])
# stargazer.add_line("Support or Not", ["?", "?", "?", "?"])
#
# # print(stargazer.render_latex())
#
# html = stargazer.render_html()
# with open(Code_dir + "Data/Analysis_Results/degree_analysis_results.html", "w") as f: f.write(html)


#with interact facts between Spy and SpyWin

model_pos_in_fitted = smf.mixedlm(
    "pos_in_degree ~  Spy+ SpyWin+Spy*SpyWin+Male+GameExperience+NativeEngSpeaker",
    data=game_nodes,
    groups=game_nodes['game_name']).fit()

model_neg_in_fitted = smf.mixedlm(
    "neg_in_degree ~ Spy+ SpyWin+Spy*SpyWin+Male+GameExperience+NativeEngSpeaker",
    data=game_nodes,
    groups=game_nodes['game_name']).fit()

model_pos_out_fitted = smf.mixedlm(
    "pos_out_degree ~ Spy+ SpyWin+Spy*SpyWin+Male+GameExperience+NativeEngSpeaker",
    data=game_nodes,
    groups=game_nodes['game_name']).fit()

model_neg_out_fitted = smf.mixedlm(
    "neg_out_degree ~ Spy+ SpyWin+Spy*SpyWin+Male+GameExperience+NativeEngSpeaker",
    data=game_nodes,
    groups=game_nodes['game_name']).fit()


# Create a list of fitted models
fitted_models = [model_pos_in_fitted, model_neg_in_fitted, model_pos_out_fitted, model_neg_out_fitted]

# Use Stargazer to format the table
stargazer = Stargazer(fitted_models)

stargazer.title("Degree Regression Results")
stargazer.custom_columns(["Perceived Trust", "Perceived Distrust", "Expressed Trust", "Expressed Distrust"],
                         [1, 1, 1, 1])

# change the variable name with stargazer table also, change the order of the variables
stargazer.rename_covariates({'Group Var': 'Group Effect'})
stargazer.covariate_order(
    ['Spy', 'SpyWin','Spy:SpyWin', 'Male', 'NativeEngSpeaker', 'GameExperience',  'Group Var',
     'Intercept'])

stargazer.add_line("Hypothesis", ["H1a", "H1b", "H2a", "H2b"])
stargazer.add_line("Support or Not", ["Yes", "No", "Yes", "No"])


html = stargazer.render_html()
with open(Code_dir + "Data/Analysis_Results/degree_analysis_results_interact.html", "w") as f: f.write(html)



### run regression for the out degree


model_pos_in_fitted = smf.mixedlm(
    " trust_outdegree_villager~  Spy+ SpyWin+Spy*SpyWin+Male+GameExperience+NativeEngSpeaker",
    data=game_nodes,
    groups=game_nodes['game_name']).fit()

model_neg_in_fitted = smf.mixedlm(
    "trust_outdegree_spy ~ Spy+ SpyWin+Spy*SpyWin+Male+GameExperience+NativeEngSpeaker",
    data=game_nodes,
    groups=game_nodes['game_name']).fit()

model_pos_out_fitted = smf.mixedlm(
    "distrust_outdegree_villager ~ Spy+ SpyWin+Spy*SpyWin+Male+GameExperience+NativeEngSpeaker",
    data=game_nodes,
    groups=game_nodes['game_name']).fit()

model_neg_out_fitted = smf.mixedlm(
    "distrust_outdegree_spy ~ Spy+ SpyWin+Spy*SpyWin+Male+GameExperience+NativeEngSpeaker",
    data=game_nodes,
    groups=game_nodes['game_name']).fit()


# Create a list of fitted models
fitted_models = [model_pos_in_fitted, model_neg_in_fitted, model_pos_out_fitted, model_neg_out_fitted]

# Use Stargazer to format the table
stargazer = Stargazer(fitted_models)

stargazer.title("Degree Regression Results")
stargazer.custom_columns(["Trust Villager", "Trust Spy", "Distrust Villager", "Distrust Spy"],
                         [1, 1, 1, 1])

# change the variable name with stargazer table also, change the order of the variables
stargazer.rename_covariates({'Group Var': 'Group Effect'})
stargazer.covariate_order(
    ['Spy', 'SpyWin','Spy:SpyWin', 'Male', 'NativeEngSpeaker', 'GameExperience',  'Group Var',
     'Intercept'])

stargazer.add_line("Hypothesis", ["H3a", "H3b", "H3c", "H3d"])
stargazer.add_line("Support or Not", ["?", "?", "?", "?"])


html = stargazer.render_html()
with open(Code_dir + "Data/Analysis_Results/degree_analysis_results_outdegree.html", "w") as f: f.write(html)

#save the game_nodes to a csv file
# game_nodes.to_csv(Code_dir + 'Data/Analysis_Results/game_nodes_with_degrees.csv', index=False)
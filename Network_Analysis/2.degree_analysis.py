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
from matplotlib import pyplot as plt
from stargazer.stargazer import Stargazer
import numpy as np
import seaborn as sns
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

stargazer.add_line("Hypothesis", ["H1a&b", "H2a&b", "H3", "H4"])
stargazer.add_line("Support or Not", ["Yes", "No", "No", "No"])


html = stargazer.render_html()
with open(Code_dir + "Data/Analysis_Results/degree_analysis_results_interact.html", "w") as f: f.write(html)



# # run the data statistics
# # total number of players
# print("total number of players:", len(game_nodes))
# # total number of games
# print("total number of games:", len(game_nodes['game_name'].unique()))
#
# # number of spy and villager
# print(game_nodes['Game_Role'].value_counts())
# # number of win and lose
# print(game_nodes['game_result'].value_counts())
# # number of male and female
# print(game_nodes['sex'].value_counts())
# # number of players who have played the game before
# print(game_nodes['play_b4'].value_counts())
# # number of native and non-native english speakers
# print(game_nodes['Eng_nativ'].value_counts())

# the number of positive links
# the number of negative links
game_nodes['pos_in_degree'].sum()
game_nodes['neg_in_degree'].sum()


# one by one table
# Create a list of fitted models
fitted_models = [model_pos_in_fitted]

# Use Stargazer to format the table
stargazer = Stargazer(fitted_models)

stargazer.title("Degree Regression Results")
stargazer.custom_columns(["Perceived Trust"])

# change the variable name with stargazer table also, change the order of the variables
stargazer.rename_covariates({'Group Var': 'Group Effect'})
stargazer.covariate_order(
    ['Spy', 'SpyWin','Spy:SpyWin', 'Male', 'NativeEngSpeaker', 'GameExperience',  'Group Var',
     'Intercept'])

stargazer.add_line("Hypothesis", ["H1a&b"])
stargazer.add_line("Support or Not", ["Yes"])


html = stargazer.render_html()
with open(Code_dir + f"Data/Analysis_Results/degree_analysis_results_{'model_pos_in_fitted'}.html", "w") as f: f.write(html)


###
fitted_models = [model_neg_in_fitted]

# Use Stargazer to format the table
stargazer = Stargazer(fitted_models)

stargazer.title("Degree Regression Results")
stargazer.custom_columns(["Perceived Distrust"])

# change the variable name with stargazer table also, change the order of the variables
stargazer.rename_covariates({'Group Var': 'Group Effect'})
stargazer.covariate_order(
    ['Spy', 'SpyWin','Spy:SpyWin', 'Male', 'NativeEngSpeaker', 'GameExperience',  'Group Var',
     'Intercept'])

stargazer.add_line("Hypothesis", ["H2a&b"])
stargazer.add_line("Support or Not", ["No"])


html = stargazer.render_html()
with open(Code_dir + f"Data/Analysis_Results/degree_analysis_results_{'model_neg_in_fitted'}.html", "w") as f: f.write(html)


fitted_models = [model_pos_out_fitted]

# Use Stargazer to format the table
stargazer = Stargazer(fitted_models)

stargazer.title("Degree Regression Results")
stargazer.custom_columns(["Expressed Trust"])

# change the variable name with stargazer table also, change the order of the variables
stargazer.rename_covariates({'Group Var': 'Group Effect'})
stargazer.covariate_order(
    ['Spy', 'SpyWin','Spy:SpyWin', 'Male', 'NativeEngSpeaker', 'GameExperience',  'Group Var',
     'Intercept'])

stargazer.add_line("Hypothesis", ["H3.1"])
stargazer.add_line("Support or Not", ["No"])


html = stargazer.render_html()
with open(Code_dir + f"Data/Analysis_Results/degree_analysis_results_{'model_pos_out_fitted'}.html", "w") as f: f.write(html)


fitted_models = [model_neg_out_fitted]

# Use Stargazer to format the table
stargazer = Stargazer(fitted_models)

stargazer.title("Degree Regression Results")
stargazer.custom_columns(["Expressed Distrust"])

# change the variable name with stargazer table also, change the order of the variables
stargazer.rename_covariates({'Group Var': 'Group Effect'})
stargazer.covariate_order(
    ['Spy', 'SpyWin','Spy:SpyWin', 'Male', 'NativeEngSpeaker', 'GameExperience',  'Group Var',
     'Intercept'])

stargazer.add_line("Hypothesis", ["H4"])
stargazer.add_line("Support or Not", ["No"])


html = stargazer.render_html()
with open(Code_dir + f"Data/Analysis_Results/degree_analysis_results_{'model_neg_out_fitted'}.html", "w") as f: f.write(html)

##

## run regression for the out degree

# adjust the out degree by the number of villager and spy
#get the Num0fVillager and Num0fSpy
game_nodes['Num0fVillager'] = game_nodes['game_name'].map(game_nodes[game_nodes['Game_Role'] == 'Villager'].groupby('game_name').size())
game_nodes['Num0fSpy'] = game_nodes['game_name'].map(game_nodes[game_nodes['Game_Role'] == 'Spy'].groupby('game_name').size())

game_nodes['trust_outdegree_villager'] = game_nodes['trust_outdegree_villager'] / game_nodes['Num0fVillager']
game_nodes['trust_outdegree_spy'] = game_nodes['trust_outdegree_spy'] / game_nodes['Num0fSpy']
game_nodes['distrust_outdegree_villager'] = game_nodes['distrust_outdegree_villager'] / game_nodes['Num0fVillager']
game_nodes['distrust_outdegree_spy'] = game_nodes['distrust_outdegree_spy'] / game_nodes['Num0fSpy']



trust_outdegree_villager = smf.mixedlm(
    " trust_outdegree_villager~  Spy+ SpyWin+Spy*SpyWin+Male+GameExperience+NativeEngSpeaker",
    data=game_nodes,
    groups=game_nodes['game_name']).fit()

trust_outdegree_spy = smf.mixedlm(
    "trust_outdegree_spy ~ Spy+ SpyWin+Spy*SpyWin+Male+GameExperience+NativeEngSpeaker",
    data=game_nodes,
    groups=game_nodes['game_name']).fit()

distrust_outdegree_villager = smf.mixedlm(
    "distrust_outdegree_villager ~ Spy+ SpyWin+Spy*SpyWin+Male+GameExperience+NativeEngSpeaker",
    data=game_nodes,
    groups=game_nodes['game_name']).fit()

distrust_outdegree_spy = smf.mixedlm(
    "distrust_outdegree_spy ~ Spy+ SpyWin+Spy*SpyWin+Male+GameExperience+NativeEngSpeaker",
    data=game_nodes,
    groups=game_nodes['game_name']).fit()


# Create a list of fitted models
fitted_models = [trust_outdegree_villager, trust_outdegree_spy, distrust_outdegree_villager, distrust_outdegree_spy]

# Use Stargazer to format the table
stargazer = Stargazer(fitted_models)

stargazer.title("Expressed Trust/Distrust Results")
stargazer.custom_columns(["Trust Villager", "Trust Spy", "Distrust Villager", "Distrust Spy"],
                         [1, 1, 1, 1])

# change the variable name with stargazer table also, change the order of the variables
stargazer.rename_covariates({'Group Var': 'Group Effect'})
stargazer.covariate_order(
    ['Spy', 'SpyWin','Spy:SpyWin', 'Male', 'NativeEngSpeaker', 'GameExperience',  'Group Var',
     'Intercept'])

stargazer.add_line("Hypothesis", ["", "H3.2", "", "H4.2"])
stargazer.add_line("Support or Not", ["", "No", "", "Yes"])
# todo: need to change the columns name add the hypothesis and support or not

html = stargazer.render_html()
with open(Code_dir + "Data/Analysis_Results/degree_analysis_results_outdegree.html", "w") as f: f.write(html)




# express different in role
game_nodes['trust_outdegree_diff'] = game_nodes['trust_outdegree_spy'] - game_nodes['trust_outdegree_villager']
game_nodes['distrust_outdegree_diff'] = game_nodes['distrust_outdegree_spy'] - game_nodes['distrust_outdegree_villager']

trust_outdegree_diff = smf.mixedlm(
    "distrust_outdegree_spy ~ Spy+ SpyWin+Spy*SpyWin+Male+GameExperience+NativeEngSpeaker",
    data=game_nodes,
    groups=game_nodes['game_name']).fit()

distrust_outdegree_diff = smf.mixedlm(
    "distrust_outdegree_diff ~ Spy+ SpyWin+Spy*SpyWin+Male+GameExperience+NativeEngSpeaker",
    data=game_nodes,
    groups=game_nodes['game_name']).fit()

# Create a list of fitted models
fitted_models = [trust_outdegree_diff, distrust_outdegree_diff]

stargazer = Stargazer(fitted_models)

stargazer.title("Deceiver Expressed Trust/Distrust Results")
stargazer.custom_columns(["Expressed Trust diff", "Expressed Distrust diff" ],
                         [1, 1])

# change the variable name with stargazer table also, change the order of the variables
stargazer.rename_covariates({'Group Var': 'Group Effect'})
stargazer.covariate_order(
    ['Spy', 'SpyWin','Spy:SpyWin', 'Male', 'NativeEngSpeaker', 'GameExperience',  'Group Var',
     'Intercept'])

stargazer.add_line("Hypothesis", ["", ""])
stargazer.add_line("Support or Not", ["", "No"])


html = stargazer.render_html()
with open(Code_dir + "Data/Analysis_Results/degree_analysis_results_outdegree_diff.html", "w") as f: f.write(html)


#save the game_nodes to a csv file
game_nodes.to_csv(Code_dir + 'Data/Analysis_Results/game_nodes_with_degrees.csv', index=False)

# using anova
# compare only the spy data
spy_nodes = game_nodes[game_nodes['Game_Role'] == 'Spy']

# compare trust toward village
# using anova compare the trust_outdegree_villager and trust_outdegree_spy
# compare the trust_outdegree_villager and trust_outdegree_spy
# anova
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Perform One-Way ANOVA
f_value, p_value = stats.f_oneway(spy_nodes['trust_outdegree_villager'], spy_nodes['trust_outdegree_spy'])

print("F-value:", f_value)
print("P-value:", p_value)


#plot the boxplot


# plot box plot for trust_outdegree_villager, trust_outdegree_spy, distrust_outdegree_villager, distrust_outdegree_spy
# plot the boxplot
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
sns.boxplot(ax=axes[0, 0], x='Game_Role', y='trust_outdegree_villager', data=game_nodes)
sns.boxplot(ax=axes[0, 1], x='Game_Role', y='trust_outdegree_spy', data=game_nodes)
sns.boxplot(ax=axes[1, 0], x='Game_Role', y='distrust_outdegree_villager', data=game_nodes)
sns.boxplot(ax=axes[1, 1], x='Game_Role', y='distrust_outdegree_spy', data=game_nodes)

# show the plot
plt.show()

# spy data only. plot box plot for trust_outdegree_villager, trust_outdegree_spy, distrust_outdegree_villager, distrust_outdegree_spy
# plot the boxplot, in one plot
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
sns.boxplot(ax=axes[0, 0], x='WinLose', y='trust_outdegree_villager', data=spy_nodes)
sns.boxplot(ax=axes[0, 1], x='WinLose', y='trust_outdegree_spy', data=spy_nodes)
sns.boxplot(ax=axes[1, 0], x='WinLose', y='distrust_outdegree_villager', data=spy_nodes)
sns.boxplot(ax=axes[1, 1], x='WinLose', y='distrust_outdegree_spy', data=spy_nodes)

plt.show()

# With the provided column names, we'll focus on creating a boxplot for:
# 'trust_outdegree_villager', 'trust_outdegree_spy', 'distrust_outdegree_villager', 'distrust_outdegree_spy'

# Extracting the relevant columns for the plot
relevant_columns = spy_nodes[['trust_outdegree_villager', 'trust_outdegree_spy',
                              'distrust_outdegree_villager', 'distrust_outdegree_spy']]

# Converting to long format for plotting
relevant_columns_long = pd.melt(relevant_columns, var_name='Connection Type', value_name='Outdegree')

# Plotting the boxplot
plt.figure(figsize=(12, 8))
sns.boxplot(x='Connection Type', y='Outdegree', data=relevant_columns_long)
plt.title('Deceiver Trust and Distrust Outdegrees')
plt.xticks(rotation=45)
plt.show()


# Create a list of fitted models for expressed trust
fitted_models = [model_pos_in_fitted,trust_outdegree_villager, trust_outdegree_spy]

# Use Stargazer to format the table
stargazer = Stargazer(fitted_models)

stargazer.title("Expressed Trust Results")
stargazer.custom_columns(["Expressed Trust","Trust Villager", "Trust Spy"],
                         [1, 1, 1])

# change the variable name with stargazer table also, change the order of the variables
stargazer.rename_covariates({'Group Var': 'Group Effect'})
stargazer.covariate_order(
    ['Spy', 'SpyWin','Spy:SpyWin', 'Male', 'NativeEngSpeaker', 'GameExperience',  'Group Var',
     'Intercept'])

stargazer.add_line("Hypothesis", ["H3.1", "", ""])
stargazer.add_line("Support or Not", ["No", "", ""])
# todo: need to change the columns name add the hypothesis and support or not

html = stargazer.render_html()
with open(Code_dir + "Data/Analysis_Results/degree_analysis_results_outdegree_Trust_3.html", "w") as f: f.write(html)

# Create a list of fitted models for expressed distrust
fitted_models = [model_neg_in_fitted, distrust_outdegree_villager, distrust_outdegree_spy]

# Use Stargazer to format the table
stargazer = Stargazer(fitted_models)

stargazer.title("Expressed Distrust Results")
stargazer.custom_columns(["Expressed Distrust","Distrust Villager", "Distrust Spy"],
                         [1, 1, 1])

# change the variable name with stargazer table also, change the order of the variables
stargazer.rename_covariates({'Group Var': 'Group Effect'})
stargazer.covariate_order(
    ['Spy', 'SpyWin','Spy:SpyWin', 'Male', 'NativeEngSpeaker', 'GameExperience',  'Group Var',
     'Intercept'])

stargazer.add_line("Hypothesis", ["H4.1", "", ""])
stargazer.add_line("Support or Not", ["No", "", ""])
# todo: need to change the columns name add the hypothesis and support or not

html = stargazer.render_html()
with open(Code_dir + "Data/Analysis_Results/degree_analysis_results_outdegree_Distrust_3.html", "w") as f: f.write(html)

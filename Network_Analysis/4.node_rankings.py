'''
This script is designed to calculate the ranking scores for each node in the network
The ranking scores include:
    1. receivedTrust
    2. prestige_scores
    3. hits_scores
    4. pageRank

'''

import networkx as nx
import pandas as pd
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer
import numpy as np


def calculate_receivedTrust(network):
    """
    Calculate the positive and negative in- and out-degrees of each node in the network.
    :param network: A NetworkX DiGraph where edges have a 'sign' attribute indicating trust (positive) or distrust (negative).
    :return: Four dictionaries of node degrees: positive in-degree, negative in-degree, positive out-degree, negative out-degree.
    """
    pos_in_degree = {}

    for node in network.nodes():
        pos_in_degree[node] = sum(1 for _, _, data in network.in_edges(node, data=True) if data['sign'] > 0)

    # normalize the received trust
    receivedTrust = {k: v / sum(pos_in_degree.values()) for k, v in pos_in_degree.items()}

    return receivedTrust


def calculate_prestige(G):
    """
    Calculate the Prestige of each node in a signed trust-distrust network.
    :param G: A NetworkX DiGraph where edges have a 'sign' attribute indicating trust (positive) or distrust (negative).
    :return: A dictionary of node Prestige scores.
    """
    prestige_scores = {}
    for node in G.nodes():
        trust_links = sum(1 for _, _, sign in G.in_edges(node, data='sign') if sign > 0)
        distrust_links = sum(1 for _, _, sign in G.in_edges(node, data='sign') if sign < 0)
        if trust_links + distrust_links > 0:
            prestige_scores[node] = (trust_links - distrust_links) / (trust_links + distrust_links)
        else:
            prestige_scores[node] = 0

    return prestige_scores


def calculate_HITS(G):
    """
    Calculate the HITS scores of each node in a signed trust-distrust network.
    :param G: A NetworkX DiGraph where edges have a 'weight' attribute indicating trust (positive) or distrust (negative).
    :return: A dictionary of node authority score differences.
    """
    # Initialize dictionaries to store the hub and authority scores
    authority_scores = {}
    # Split the graph into positive and negative subgraphs
    G_positive = nx.DiGraph([(u, v, d) for u, v, d in G.edges(data=True) if d['sign'] > 0])
    G_negative = nx.DiGraph([(u, v, d) for u, v, d in G.edges(data=True) if d['sign'] < 0])
    # Calculate the hub and authority scores for the positive and negative subgraphs
    _, authority_scores_positive = nx.hits(G_positive)
    _, authority_scores_negative = nx.hits(G_negative)
    # Combine the authority scores
    for node in G.nodes():
        positive_score = authority_scores_positive.get(node, 0)
        negative_score = authority_scores_negative.get(node, 0)
        authority_scores[node] = positive_score - negative_score

    return authority_scores


def calculate_signed_pageRank(G):
    """
    Calculate the signed PageRank scores of each node in a signed network.
    :param G: A NetworkX DiGraph where edges have a 'weight' attribute indicating positive or negative relationships.
    :return: A dictionary of node PageRank score differences (positive - negative).
    """
    # Split the graph into positive and negative subgraphs
    G_positive = nx.DiGraph([(u, v) for u, v, d in G.edges(data=True) if d['sign'] > 0])
    G_negative = nx.DiGraph([(u, v) for u, v, d in G.edges(data=True) if d['sign'] < 0])

    # Calculate the PageRank for the positive and negative subgraphs
    pageRank_positive = nx.pagerank(G_positive)
    pageRank_negative = nx.pagerank(G_negative)

    # Calculate the difference between positive and negative PageRank scores
    pageRank_diff = {node: pageRank_positive.get(node, 0) - pageRank_negative.get(node, 0)
                     for node in G.nodes()}

    return pageRank_diff


# read the networks in all the folders
Code_dir = '/Users/saiyingge/Coding Projects/PyCharmProjects/NetworkProject/'

# get the player(nodes) information
game_nodes = pd.read_csv(Code_dir + 'Data/all_nodes.csv')

# get game names
games = game_nodes['game_name'].unique()

# count the successful games
successful_games = 0

# read the networks by game and calculate the degrees
for game in games:
    # game = '001ISR'

    try:
        network = nx.read_graphml(Code_dir + 'Data/Networks/' + game + '.graphml')

        receivedTrust = calculate_receivedTrust(network)
        prestige_scores = calculate_prestige(network)
        hits_scores = calculate_HITS(network)
        pageRank = calculate_signed_pageRank(network)
        # add the ranking scores to the game_nodes
        for node in network.nodes():
            condition = (game_nodes['Player_Number'] == int(node)) & (game_nodes['game_name'] == game)
            game_nodes.loc[condition, 'receivedTrust'] = receivedTrust.get(node, 0)
            game_nodes.loc[condition, 'prestige'] = prestige_scores.get(node, 0)
            game_nodes.loc[condition, 'hits'] = hits_scores.get(node, 0)
            game_nodes.loc[condition, 'pageRank'] = pageRank.get(node, 0)

        print(f'Finished processing {game}.graphml')
        successful_games += 1
    except:
        print(f'Error processing {game}.graphml')
        continue

print(f'Finished processing {successful_games} files')

# save the game_nodes
game_nodes.to_csv(Code_dir + 'Data/all_nodes_W_rankings.csv', index=False)

# TODO: check the na's
# rows with missing values
game_nodes[game_nodes.isnull().any(axis=1)]
# percentage of missing values
game_nodes.isnull().sum() / len(game_nodes)
# drop any rows with missing values
game_nodes = game_nodes.dropna()

# calculate the correlation between the ranking scores
print(game_nodes[['receivedTrust', 'prestige', 'hits', 'pageRank']].corr())

# run regression with ranking scores as dependent variables and other attributes as independent variables
game_nodes.loc[:, 'Spy'] = (game_nodes['Game_Role'] == 'Spy').astype(int)
game_nodes.loc[:, 'SpyWin'] = (game_nodes['game_result'] == 'SpyWin').astype(int)

game_nodes.loc[:, 'GameExperience'] = (game_nodes['play_b4'] == 'yes').astype(int)
game_nodes.loc[:, 'NativeEngSpeaker'] = (game_nodes['Eng_nativ'] == 'native speaker').astype(int)
game_nodes.loc[:, 'HomogeneousGroupCulture'] = (game_nodes['homogeneous'] == 'Yes').astype(int)
game_nodes.loc[:, 'Male'] = (game_nodes['sex'] == 'Male').astype(int)

# check the correlation between the independent variables
print(game_nodes[['Spy', 'SpyWin', 'GameExperience', 'NativeEngSpeaker','Male']].corr())


model_receivedTrust_fitted = smf.mixedlm(
    "receivedTrust ~  Spy+ SpyWin+Male+Spy*SpyWin+GameExperience+NativeEngSpeaker",
    data=game_nodes,
    groups=game_nodes['game_name']).fit()

model_prestige_fitted = smf.mixedlm(
    "prestige ~ Spy+ SpyWin+Male+Spy*SpyWin+GameExperience+NativeEngSpeaker",
    data=game_nodes,
    groups=game_nodes['game_name']).fit()

model_hits_fitted = smf.mixedlm(
    "hits ~ Spy+ SpyWin+Male+Spy*SpyWin+GameExperience+NativeEngSpeaker",
    data=game_nodes,
    groups=game_nodes['game_name']).fit()

model_pageRank_fitted = smf.mixedlm(
    "pageRank ~ Spy+ SpyWin+Male+Spy*SpyWin+GameExperience+NativeEngSpeaker",
    data=game_nodes,
    groups=game_nodes['game_name']).fit()

# Create a list of fitted models
fitted_models = [model_receivedTrust_fitted, model_prestige_fitted, model_hits_fitted, model_pageRank_fitted]

# Use Stargazer to format the table
stargazer = Stargazer(fitted_models)

# #todo: need to change the columns name add the hypothesis and support or not
# Configure the stargazer settings (optional)
stargazer.title("Node Ranking Regression Results")
stargazer.custom_columns(["Received Trust", "Prestige", "HITS", "PageRank"],
                         [1, 1, 1, 1])

# change the variable name with stargazer table also, change the order of the variables
stargazer.rename_covariates({'Group Var': 'Group Effect'})
stargazer.covariate_order(
    ['Spy', 'SpyWin','Spy:SpyWin', 'Male', 'NativeEngSpeaker', 'GameExperience', 'Group Var',
     'Intercept'])


# print(stargazer.render_latex())

html = stargazer.render_html()
with open(Code_dir + "Data/Analysis_Results/Ranking_analysis_results.html", "w") as f: f.write(html)

#todo: check the results
'''

construct networks base on the links and node information extracted from the experiment data
create a plot for each network

input: all_links.csv, all_nodes.csv
output: networks for further analysis, network plots

'''

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def combine_edge_to_weight(network):
    '''
    combine the edges with the same from, to and sign to one edge with weight
    :param network: multi-directed network
    :return: new network with weights
    '''
    # change from multiple edges to single edges and add wight
    combined_network = nx.DiGraph()

    for u, v, data in network.edges(data=True):
        sign = data.get('sign', 0)  # Default to 1 if sign is not present

        # Check if an edge with the same u, v, and sign already exists
        if combined_network.has_edge(u, v):

            existing_edge_data = combined_network.get_edge_data(u, v)
            if existing_edge_data.get('sign') == sign:
                # If the edge exists and has the same sign, increment its weight
                # If the edge exists and has the same sign, increment its weight
                combined_network[u][v]['weight'] = existing_edge_data.get('weight', 1) + 1

        else:
            # If no such edge exists, add a new edge with weight 1
            combined_network.add_edge(u, v, sign=sign, weight=1)
    return combined_network


def visualize_network(network, game_nodes, game):
    '''
    visualize the network
    :param network:  multi-directed network
    :param game_nodes:  nodes information
    :param game:    game name
    :return:    plot
    '''
    # prepare for the visualization
    combined_network = combine_edge_to_weight(network)

    # Check if all nodes have a 'role' attribute
    for node in network.nodes():
        if 'role' not in network.nodes[node]:
            print(f"Node {node} at {game} does not have a 'role' attribute.")

    # Set node colors based on roles, use information from the original network
    node_colors = ['gold' if network.nodes.get(node, {}).get('role') == 'Villager' else 'tomato' for node in
                   combined_network]

    # Set edge colors based on sentiment score
    edge_colors = ['darkred' if combined_network[u][v].get('sign') == -1 else 'slategrey' for u, v in
                   combined_network.edges()]
    edge_widths = [data['weight'] for _, _, data in combined_network.edges(data=True)]

    # Sort nodes by Player_Number
    sorted_nodes = sorted(combined_network.nodes())
    # Using a circular layout with sorted nodes
    pos = nx.circular_layout(sorted_nodes)

    # plot the network
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(combined_network, pos, node_color=node_colors, edge_color=edge_colors, node_size=800,
                     alpha=0.7, arrowsize=20, font_size=16, font_color='black', font_weight='bold', width=edge_widths)

    # Adding game name as the title
    plt.title("Game Network Visualization " + "(" + game + ")", fontsize=20)
    # Creating legend items
    villager_patch = plt.Line2D([0], [0], marker='o', color='gold', label='Villager', markersize=12)
    non_villager_patch = plt.Line2D([0], [0], marker='o', color='tomato', label='Spy', markersize=12)
    positive_edge_patch = plt.Line2D([0], [0], color='slategrey', lw=4, label='Positive Link')
    negative_edge_patch = plt.Line2D([0], [0], color='darkred', lw=4, label='Negative Link')

    # First legend (for Villager, Spy, Positive Link, Negative Link)
    first_legend = plt.legend(handles=[villager_patch, non_villager_patch, positive_edge_patch, negative_edge_patch],
                              loc='lower left')

    # Add the first legend manually to the current Axes
    plt.gca().add_artist(first_legend)

    # Second legend (for Game Result and Game Culture)
    game_result_patch = mpatches.Patch(color='none', label='Game Result: ' + game_nodes['game_result'].iloc[0])
    if game_nodes['homogeneous'].iloc[0] == 'Yes':
        game_culture_patch = mpatches.Patch(color='none', label='Game Culture: Homogeneous')
    else:
        game_culture_patch = mpatches.Patch(color='none', label='Game Culture: Heterogeneous')
    plt.legend(handles=[game_result_patch, game_culture_patch], loc='lower right')

    # Hiding axes
    plt.axis('off')
    # # Displaying the plot
    # plt.show()
    # Saving the plot
    plt.savefig(Code_dir + 'Data/Networks/Plots/' + game + '.png')


# Directory path
Code_dir = '/Users/saiyingge/Coding Projects/PyCharmProjects/NetworkProject/'

# read the links and nodes
links = pd.read_csv(Code_dir + 'Data/all_links.csv')
nodes = pd.read_csv(Code_dir + 'Data/all_nodes.csv')

# for each game, construct a network and create a plot. save the network and plot
for game in links['game'].unique():
    # filter the links and nodes for each game
    #  game='007NTU'
    game_links = links[links['game'] == game]
    game_nodes = nodes[nodes['game_name'] == game]

    # drop index, then remove duplicates links in each round
    game_links = game_links.drop(columns=['indx'])
    game_links = game_links.drop_duplicates()

    # also remove the self-loops

    game_links = game_links[game_links['from'] != game_links['to']]
    game_links = game_links.reset_index(drop=True)

    # create a new network
    network = nx.MultiDiGraph()

    # add nodes
    for node, role in zip(game_nodes['Player_Number'], game_nodes['Game_Role']):
        network.add_node(node, role=role)

    # add links
    for index, row in game_links.iterrows():
        network.add_edge(row['from'], row['to'], sign=row['sentiment_score'], round=row['round'])

    # save the network
    nx.write_graphml(network, Code_dir + 'Data/Networks/' + game + '.graphml')

    # visualize the network
    visualize_network(network, game_nodes, game)

# error handling
# Node 0 at 007NTU does not have a 'role' attribute.
# Node 24 at 008ISR does not have a 'role' attribute.
# game_nodes[game_nodes['game_name'] == '007NTU' and game_nodes['Player_Number'] == 0]
# game_nodes[game_nodes['game_name'] == '008ISR' and game_nodes['Player_Number'] == 24]
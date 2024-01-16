'''

construct networks base on the links and node information extracted from the experiment data

input: all_links.csv, all_nodes.csv
output: networks for further analysis

'''

import os
import csv
import glob
import pandas as pd
import ast
import re
import networkx as nx

# Directory path
Code_dir = '//'

# read the links and nodes
links = pd.read_csv(Code_dir + 'Data/all_links.csv')
nodes = pd.read_csv(Code_dir + 'Data/all_nodes.csv')

# for each game, construct a network
for game in links['game'].unique():
    # filter the links and nodes for each game
    game_links = links[links['game'] == game]
    game_nodes = nodes[nodes['game_name'] == game]

    # drop index, then remove duplicates links in each round
    game_links = game_links.drop(columns=['indx'])
    game_links=game_links.drop_duplicates()

    # create a new network
    network = nx.DiGraph()

    # add nodes
    for node, role in zip(game_nodes['Player_Number'], game_nodes['Game_Role']):
        network.add_node(node, role=role)

    # add links
    for index, row in game_links.iterrows():
        network.add_edge(row['from'], row['to'], sign=row['sentiment_score'])

    # save the network
    nx.write_graphml(network, Code_dir + 'Data/Networks/' + game + '.graphml')
# todo:
# visualization
# net<-graph_from_data_frame(d=links, vertices=nodes, directed=T)
#
# V_colrs<- c( "gold", "tomato")
# V(net)$color[V(net)$Game_Role=="Spy"] <- V_colrs[2]
# V(net)$color[V(net)$Game_Role=="Villager"] <- V_colrs[1]
# E_colrs<-c( "dark red", "slategrey")
# E(net)$color[E(net)$weight == "1"] <- E_colrs[2]
# E(net)$color[E(net)$weight == "-1"] <- E_colrs[1]
# par(mfrow = c(1, 1))
# plot(net, edge.arrow.size=.4,
#      vertex.label=V(net)$Player_number,
#      layout=layout.circle,
#      main=paste(GameName)
# )
# #legend(x=-1.5, y=-1.1, c("Villager","Spy"), pch=21, col=V_colrs, pt.bg=V_colrs, pt.cex=2, cex=.8, bty="n", ncol=1)
# legend(x=0, y=-1.1, c("Distrust","Trust"), lty=1,lwd=3,col=E_colrs, pt.bg=E_colrs, pt.cex=2, cex=.8, bty="n", ncol=1)

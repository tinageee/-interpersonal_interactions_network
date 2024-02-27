'''
This script is to build a GNN model to predict the role of the player in the game
'''
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import pandas as pd

import random

from torch_geometric.loader import DataLoader

from torch_geometric.nn import GATConv, NNConv, global_mean_pool
from torch.nn import Sequential, Linear, ReLU
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def load_graphml_to_pyg_with_edge_attrs(game):
    # game = '018AZ'
    # Load the graph from a .graphml file
    G = nx.read_graphml(Code_dir + 'Data/Networks/' + game + '.graphml')

    edges = []
    for u, v, data in G.edges(data=True):
        # print(u, v, data)
        sign = data.get('sign')
        round_num = data.get('round')[-1]  # get the last dot in the label
        edges.append((u, v, sign, int(round_num)))

    # Prepare a dictionary to hold processed node attributes (features and labels)
    for node_id, node_data in G.nodes(data=True):
        # print(node_id, node_data)
        # find the node information
        condition = (game_nodes['Player_Number'] == int(node_id)) & (game_nodes['game_name'] == game)

        # Convert features to tensor
        features = \
        game_nodes.loc[condition, ['Male', 'NativeEngSpeaker', 'GameExperience', 'HomogeneousGroupCulture', 'pageRank','hits','prestige','receivedTrust'
                                   ]].iloc[0]
        features_tensor = torch.tensor(features.to_numpy(), dtype=torch.float)

        # Convert label to tensor ( binary classification: spy=1, villager=0), keep the value only  remove the index
        label_tensor = torch.tensor(game_nodes.loc[condition, 'Spy'].iloc[0])

        # Update the node data
        node_data['x'] = features_tensor
        node_data['y'] = label_tensor

    # convert the node name from str to int from 1->n to 0 -> n-1
    # Create a mapping from current labels (1-N) to new labels (0-(N-1))
    mapping = {node: int(node) - 1 for node in G.nodes()}

    # Use nx.relabel_nodes to apply the mapping
    G = nx.relabel_nodes(G, mapping)

    # Create edge_index from the edges, adjust the node name from 1->n to 0 -> n-1
    edge_index = torch.tensor([[int(u)-1, int(v)-1] for u, v, _, _ in edges], dtype=torch.long).t().contiguous()

    # creat index for positive and negtive
    edge_index_pos = torch.tensor([[int(u)-1, int(v)-1] for u, v, sign, _ in edges if sign == 1], dtype=torch.long).t().contiguous()
    edge_index_neg = torch.tensor([[int(u)-1, int(v)-1] for u, v, sign, _ in edges if sign == -1], dtype=torch.long).t().contiguous()

    # Create edge_attr from the edges
    edge_attr = torch.tensor([[sign, round_num] for _, _, sign, round_num in edges], dtype=torch.float)

    # Convert the NetworkX graph to a PyTorch Geometric Data object
    data = from_networkx(G)

    # Add edge_index and edge_attr to the PyG data object
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    data.edge_index_pos = edge_index_pos
    data.edge_index_neg = edge_index_neg

    return data


# read the networks in all the folders
Code_dir = '/Users/saiyingge/Coding Projects/PyCharmProjects/NetworkProject/'

# get the player(nodes) information
game_nodes = pd.read_csv(Code_dir + 'Data/all_nodes_W_rankings.csv')

# get game names
games = game_nodes['game_name'].unique()

# convert to binary
game_nodes['Spy'] = (game_nodes['Game_Role'] == 'Spy').astype(int)
game_nodes['SpyWin'] = (game_nodes['game_result'] == 'SpyWin').astype(int)

game_nodes['Male'] = (game_nodes['sex'] == 'Male').astype(int)
game_nodes['NativeEngSpeaker'] = (game_nodes['Eng_nativ'] == 'native speaker').astype(int)
game_nodes['GameExperience'] = (game_nodes['play_b4'] == 'yes').astype(int)
game_nodes['HomogeneousGroupCulture'] = (game_nodes['homogeneous'] == 'Yes').astype(int)

# count the successful games
successful_games = 0

datasets = []  # List to store loaded PyG Data objects

for game in games:
    try:
        data = load_graphml_to_pyg_with_edge_attrs(game)
        datasets.append(data)
        print(f"Successfully loaded and converted {game}")
    except Exception as e:
        print(f"Failed to load {game}: {e}")

print(f"Successfully loaded {len(datasets)} games")

# Split the dataset into training, validation, and test sets
random.shuffle(datasets)  # Shuffle the dataset

train_size = int(0.7 * len(datasets))
val_size = int(0.15 * len(datasets))

train_dataset = datasets[:train_size]
val_dataset = datasets[train_size:train_size + val_size]
test_dataset = datasets[train_size + val_size:]

# Create PyTorch Geometric DataLoaders for each set

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define the GNN models

class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, 32)  # Increase the output features
        self.conv2 = GCNConv(32, 64)  # Adding an intermediate layer
        self.conv3 = GCNConv(64, num_classes)  # Final layer to num_classes

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))  # Use ReLU before final layer
        x = F.dropout(x, training=self.training)  # Optional: additional dropout
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)


class SignedGCNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SignedGCNLayer, self).__init__()
        # For positive and negative relations
        self.conv_pos = GCNConv(in_channels, out_channels)
        self.conv_neg = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index_pos, edge_index_neg):
        # Apply GCN on positive and negative edges separately
        x_pos = F.relu(self.conv_pos(x, edge_index_pos))
        x_neg = F.relu(self.conv_neg(x, edge_index_neg))
        # Combine positive and negative information
        x = x_pos - x_neg  # Example strategy, could be adjusted
        return x

class SignedGCNModel(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(SignedGCNModel, self).__init__()
        self.layer1 = SignedGCNLayer(num_node_features, 16)
        self.layer2 = SignedGCNLayer(16, num_classes)

    def forward(self, data):
        x, edge_index_pos, edge_index_neg = data.x, data.edge_index_pos, data.edge_index_neg
        x = self.layer1(x, edge_index_pos, edge_index_neg)
        x = F.dropout(x, training=self.training)
        x = self.layer2(x, edge_index_pos, edge_index_neg)
        return F.log_softmax(x, dim=1)

class CustomGNNModel(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_classes):
        super(CustomGNNModel, self).__init__()
        # NNConv layer to incorporate edge features
        nn = Sequential(Linear(num_edge_features, num_node_features * 8), ReLU(),
                        Linear(num_node_features * 8, num_node_features * 8))
        self.edge_conv = NNConv(num_node_features, num_node_features * 8, nn, aggr='mean')

        # GAT layers
        self.gat1 = GATConv(num_node_features * 8, 16, heads=8, dropout=0.6)
        self.gat2 = GATConv(16 * 8, num_classes, heads=1, concat=True, dropout=0.6)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Incorporate edge features
        x = F.relu(self.edge_conv(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)

        # Process with GAT layers
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)

        # Optionally, apply global pooling if graph-level predictions are needed
        # x = global_mean_pool(x, data.batch)  # Use if you have graph-level tasks

        return F.log_softmax(x, dim=1)



# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GNNModel(num_node_features=8, num_classes=2).to(device)
model = CustomGNNModel(num_node_features=8, num_edge_features=2, num_classes=2).to(device)
# model = SignedGCNModel(num_node_features=8, num_classes=2).to(device)

#set the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
model.train()
# print the model
print(model)

# Train the model
for epoch in range(200):

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: Training loss {total_loss / len(train_loader)}")

for name, param in model.named_parameters():
    print(f"{name}: {param.size()}")
    print(param.data)  # Print the weight values


# Evaluate the model
model.eval()
correct = 0
total_nodes = 0  # Keep track of the total number of nodes

for data in test_loader:
    data = data.to(device)
    out = model(data)
    pred = out.argmax(dim=1)  # Get the indices of max log-probability
    correct += pred.eq(data.y).sum().item()
    total_nodes += data.y.size(0)  # Accumulate the total

accuracy = correct / total_nodes  # Calculate accuracy
print(f'Accuracy: {accuracy:.4f}')


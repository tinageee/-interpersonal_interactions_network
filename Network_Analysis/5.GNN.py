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
        edges.append((u, v, sign, int(round_num)-1))

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

    # get the max round number
    max_round = torch.tensor(game_nodes.loc[game_nodes['game_name'] == game, 'max_round'].iloc[0])
    # Convert the NetworkX graph to a PyTorch Geometric Data object
    data = from_networkx(G)

    # Add edge_index and edge_attr to the PyG data object
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    data.edge_index_pos = edge_index_pos
    data.edge_index_neg = edge_index_neg
    data.max_round = max_round

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

train_size = int(0.8 * len(datasets))
test_size = int(0.2 * len(datasets))

train_dataset = datasets[:train_size]
test_dataset = datasets[train_size:]

# Create PyTorch Geometric DataLoaders for each set

batch_size = 1

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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
        #TODO: Adjust the combination strategy, contact
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
        print(f"Input x shape: {data.x.shape}")
        print(f"Edge index shape: {data.edge_index.shape}")
        print(f"Edge attr shape: {data.edge_attr.shape}")

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Incorporate edge features
        x = F.relu(self.edge_conv(x, edge_index, edge_attr))
        print(f"After NNConv x shape: {x.shape}")
        x = F.dropout(x, p=0.6, training=self.training)

        # Process with GAT layers
        x = F.elu(self.gat1(x, edge_index))
        print(f"After GAT1 x shape: {x.shape}")
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)
        print(f"After GAT2 x shape: {x.shape}")

        # Optionally, apply global pooling if graph-level predictions are needed
        # x = global_mean_pool(x, data.batch)  # Use if you have graph-level tasks

        return F.log_softmax(x, dim=1)


class TemporalEmbedding(torch.nn.Module):
    def __init__(self, max_rounds, embedding_dim):
        super(TemporalEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(max_rounds, embedding_dim)

    def forward(self, round_numbers):
        return self.embedding(round_numbers)

class TemporalDynamicLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TemporalDynamicLayer, self).__init__()
        self.rnn = torch.nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x_sequence):
        _, hidden = self.rnn(x_sequence)
        return hidden[-1]

class SignedDynamicGNN3(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, max_rounds, embedding_dim, hidden_dim):
        super(SignedDynamicGNN3, self).__init__()
        self.conv_pos = GCNConv(num_node_features, 16)
        self.conv_neg = GCNConv(num_node_features, 16)
        self.temporal_embedding = TemporalEmbedding(max_rounds, embedding_dim)
        # 注意这里我们调整了TemporalDynamicLayer的输入维度
        self.temporal_dynamic_layer = TemporalDynamicLayer(16 * 2 + embedding_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index_pos, edge_index_neg, edge_attr, round_numbers = data.x, data.edge_index_pos, data.edge_index_neg, data.edge_attr, data.max_round
        batch = data.batch  # Tensor indicating the graph each node belongs to

        x_pos = F.relu(self.conv_pos(x, edge_index_pos))
        x_neg = F.relu(self.conv_neg(x, edge_index_neg))
        x_combined = torch.cat((x_pos, x_neg), dim=1)  # 合并正负特征
        print(f"net x_combined: {x_combined.shape}")
        # 生成轮次嵌入并扩展以匹配节点数量
        round_embeddings = self.temporal_embedding(round_numbers)
        round_embeddings_expanded = round_embeddings[batch]

        # 将结构特征与时间嵌入结合
        x_temporal = torch.cat([x_combined, round_embeddings_expanded], dim=1)
        print(f"net x_temporal: {x_temporal.shape}")
        # 适应TemporalDynamicLayer的输入要求
        # 假设TemporalDynamicLayer期望的输入形状为 (batch_size, num_nodes, feature_dim)
        # 但因为每个图的节点数可能不同，这里我们假设每个节点独立处理，不使用RNN
        hidden_state = self.temporal_dynamic_layer(x_temporal)

        net_out = self.classifier(hidden_state.squeeze(0))
        print(f"net output shape: {net_out.shape}")
        return F.log_softmax(net_out, dim=1)


class SignedDynamicGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, max_rounds, embedding_dim, hidden_dim):
        super(SignedDynamicGNN, self).__init__()
        self.conv_pos = GCNConv(num_node_features, 16)
        self.conv_neg = GCNConv(num_node_features, 16)
        self.temporal_embedding = TemporalEmbedding(max_rounds, embedding_dim)
        self.temporal_dynamic_layer = TemporalDynamicLayer(embedding_dim +16, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index_pos, edge_index_neg, edge_attr, round_numbers = data.x, data.edge_index_pos, data.edge_index_neg, data.edge_attr, data.max_round
        batch = data.batch  # Tensor indicating the graph each node belongs to

        # Existing operations
        x_pos = F.relu(self.conv_pos(x, edge_index_pos))
        x_neg = F.relu(self.conv_neg(x, edge_index_neg))
        x_combined = x_pos - x_neg

        # Generate round embeddings
        round_embeddings = self.temporal_embedding(round_numbers-1)  # [num_graphs, embedding_dim]

        # Expand round_embeddings to match the number of nodes
        # This uses the 'batch' tensor to index into 'round_embeddings'
        round_embeddings_expanded = round_embeddings[batch]

        print(f"X_combined shape: {x_combined.shape}")
        print(f"Round embeddings shape: {round_embeddings_expanded.shape}")

        # Ensure round_embeddings_expanded matches the shape of x_combined for concatenation
        x_temporal = torch.cat([x_combined, round_embeddings_expanded], dim=1)
        hidden_state = self.temporal_dynamic_layer(x_temporal.unsqueeze(0))
        net_out = self.classifier(hidden_state)
        print(f"net output shape: {net_out.shape}")
        return F.log_softmax(net_out, dim=1)


class SignedDynamicGNN2(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, max_rounds, embedding_dim, hidden_dim):
        super(SignedDynamicGNN2, self).__init__()
        self.conv_pos = GCNConv(num_node_features, 16)
        self.conv_neg = GCNConv(num_node_features, 16)
        self.temporal_embedding = TemporalEmbedding(max_rounds, embedding_dim)
        # Adjusted to combine node features and temporal embeddings before RNN
        self.dynamic_layer = TemporalDynamicLayer(16 * 2 + embedding_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        # Separate processing for positive and negative edges
        x_pos = F.relu(self.conv_pos(data.x, data.edge_index_pos))
        x_neg = F.relu(self.conv_neg(data.x, data.edge_index_neg))
        x_combined = torch.cat([x_pos, x_neg], dim=-1)

        # Temporal embedding for round numbers
        round_embeddings = self.temporal_embedding(data.max_round)

        # Combine structural features with temporal embeddings
        x_temporal = torch.cat([x_combined, round_embeddings], dim=-1)

        # Process combined features dynamically (e.g., through an RNN)
        hidden_state = self.dynamic_layer(x_temporal.unsqueeze(0))

        # Node-level classification
        out = self.classifier(hidden_state)
        return F.log_softmax(out, dim=1)

# todo:1. total 8 player input [8,8] 8 nodes, 8 features, include y as 0; output 3 demension [16,8,1]
# todo:2. set data loader batch size 16, player number 8, node feature 8
# todo:3. remove dummy node result at final end at test stage




epochs_number=20


# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GNNModel(num_node_features=8, num_classes=2).to(device)
# model = CustomGNNModel(num_node_features=8, num_edge_features=2, num_classes=2).to(device)
# model = SignedGCNModel(num_node_features=8, num_classes=2).to(device)
model = SignedDynamicGNN3(num_node_features=8, num_classes=2, max_rounds=8, embedding_dim=16, hidden_dim=32).to(device)
# model = SignedDynamicGNN2(num_node_features=8, num_classes=2, max_rounds=8, embedding_dim=16, hidden_dim=32).to(device)
#set the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
model.train()
# print the model
# print(model)

# for name, param in model.named_parameters():
#     print(f"{name}: {param.size()}")
#     print(param.data)  # Print the weight values

# Train the model
for epoch in range(epochs_number):

    total_loss = 0
    for data in train_loader:
    # for data in datasets:
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

class CustomGNNModel(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(CustomGNNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.dropout(x, p=0.5, train=self.training)
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)

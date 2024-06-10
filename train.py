import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GAE, BatchNorm
import pickle
import numpy as np
import matplotlib.pyplot as plt
from littleballoffur import RandomWalkWithRestartSampler, ForestFireSampler
from torch_geometric.loader import DataLoader, ClusterData, ClusterLoader
from torch_geometric.data import Batch
import networkx as nx
import random
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout_rate):
        super(GCNEncoder, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout_rate = dropout_rate

        # First layer
        self.layers.append(GCNConv(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))

        # Output layer
        self.layers.append(GCNConv(hidden_channels, out_channels))
        self.batch_norms.append(BatchNorm(out_channels))

    def forward(self, x, edge_index):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.batch_norms[i](self.layers[i](x, edge_index)))
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.layers[-1](x, edge_index)
        x = self.batch_norms[-1](x)
        return x

class GAEPipeline:
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers, dropout_rate, sampling_method='method1', preprocessing=True, **sampling_params):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.sampling_method_name = sampling_method
        self.sampling_method = self._get_sampling_method(sampling_method)
        self.preprocessing = preprocessing
        self.sampling_params = sampling_params

        # Create a directory name string that includes all relevant parameters
        params_str = '_'.join([f'{k}_{v}' for k, v in sampling_params.items()])
        self.directory = f"latest_models_normalize/{self.sampling_method_name}_out_{out_channels}_hidden_{hidden_channels}_layers_{num_layers}_dropout_{dropout_rate}_{params_str}"
        
        # Create directories based on hyperparameters
        os.makedirs(self.directory, exist_ok=True)
        os.makedirs(f'{self.directory}/sampled_graphs', exist_ok=True)
        
        self.encoder = GCNEncoder(in_channels, hidden_channels, out_channels, num_layers, dropout_rate).to(self.device)
        self.model = GAE(self.encoder).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def recon_loss(self, predicted_adj, true_adj):
        loss = F.binary_cross_entropy(predicted_adj, true_adj)
        return loss

    def load_graph_from_pickle(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def save_graph_to_pickle(self, graph, filename):
        with open(filename, 'wb') as f:
            pickle.dump(graph, f)

    def _get_sampling_method(self, method_name):
        """Dynamically selects the sampling method."""
        if method_name == 'random_walk':
            return self.random_walk
        elif method_name == 'forest_fire':
            return self.forest_fire
        elif method_name == 'clusterGCN':
            return self.cluster_GCN
        else:
            raise ValueError("Unknown sampling method")

    def convert_node_labels_to_integers(self, graph):
        mapping = {node: i for i, node in enumerate(graph.nodes())}
        graph_int_labels = nx.relabel_nodes(graph, mapping)
        return graph_int_labels

    def custom_collate(self, batch):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch = Batch.from_data_list(batch)
        batch.to(device)
        return batch

    def preprocess_features(self, features, max_length):
        processed_features = []
        for feature in features:
            try:
                processed_features.append(float(feature))
            except ValueError:
                processed_features.append(0.0)  # Using 0.0 as a placeholder
        # Pad features to the max length
        if len(processed_features) < max_length:
           processed_features += [0.0] * (max_length - len(processed_features))
        return processed_features

    def normalize_features(self, features):
        features = np.array(features)
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std[std == 0] = 1
        normalized_features = (features - mean) / std
        return normalized_features.tolist()

    def from_networkx_to_torch_geometric(self, G):
        mapping = {k: i for i, k in enumerate(G.nodes())}
        edges = torch.tensor([list(map(mapping.get, edge)) for edge in G.edges()], dtype=torch.long).t().contiguous()

        if G.nodes():
            sample_features = next(iter(G.nodes(data=True)))[1]
            feature_keys = list(sample_features.keys())
            
            # Determine the maximum length of feature vectors
            max_length = max(len(node_features) for _, node_features in G.nodes(data=True))
        
            features = []
            for _, node_features in G.nodes(data=True):
                node_feature_values = [node_features.get(key, 0) for key in feature_keys]
                processed_features = self.preprocess_features(node_feature_values, max_length)
                features.append(processed_features)
            
            features = self.normalize_features(features)
        else:
            features = [[0]]
    
        x = torch.tensor(features, dtype=torch.float)
        print(f'Feature matrix shape: {x.shape}')
        data = Data(x=x, edge_index=edges)
        return data

    def random_walk(self, graph):
        graph = self.convert_node_labels_to_integers(graph)
        num_nodes = self.sampling_params.pop('num_nodes', 40000)
        model = RandomWalkWithRestartSampler(number_of_nodes=num_nodes, **self.sampling_params)
        new_graph = model.sample(graph)
        self.save_graph_to_pickle(new_graph, f'{self.directory}/sampled_graphs/{self.sampling_method_name}_sampled_graph.pkl')
        return new_graph

    def forest_fire(self, graph):
        graph = self.convert_node_labels_to_integers(graph)
        num_nodes = self.sampling_params.pop('num_nodes', 30000)
        model = ForestFireSampler(number_of_nodes=num_nodes, **self.sampling_params)
        new_graph = model.sample(graph)
        self.save_graph_to_pickle(new_graph, f'{self.directory}/sampled_graphs/{self.sampling_method_name}_sampled_graph.pkl')
        return new_graph

    def cluster_GCN(self, data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(12345)
        cluster_data = ClusterData(data, num_parts=8) 
        loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True)  
        return loader

    def preprocess_graph(self, graph):
        data = self.from_networkx_to_torch_geometric(graph)
        return data

    def plot_learning_curve(self, losses, filename):
        plt.figure()
        plt.plot(losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Learning Curve')
        plt.legend()
        plt.savefig(filename)
        plt.close()

    def train(self, graph, epochs=100, batch_size=16):
        print(f"Training with parameters: out_channels={self.out_channels}, hidden_channels={self.hidden_channels}, num_layers={self.num_layers}, dropout_rate={self.dropout_rate}, sampling_method={self.sampling_method_name}")
        sampled_subgraph = self.sampling_method(graph)
        if self.preprocessing:
            data = self.preprocess_graph(sampled_subgraph)
        loader = DataLoader([data], batch_size=batch_size)
        self.model.train()
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            all_z = []
            for batch_data in loader:
                x = batch_data.x.to(self.device)
                edge_index = batch_data.edge_index.to(self.device)
                self.optimizer.zero_grad()
                z = self.model.encode(x, edge_index)
                loss = self.model.recon_loss(z, edge_index)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                all_z.append(z.detach().cpu()) 
            avg_loss = total_loss / len(loader)
            losses.append(avg_loss)
            print(f'Epoch {epoch+1}, Average Loss: {avg_loss}')
            if epoch == epochs - 1:
               z_to_save = torch.cat(all_z, dim=0)
               torch.save(z_to_save, f'{self.directory}/epoch_{epoch+1}_z_out_channels_{self.out_channels}_hidden_{self.hidden_channels}_layers_{self.num_layers}_dropout_{self.dropout_rate}.pt')
        self.plot_learning_curve(losses, f'{self.directory}/learning_curve_out_channels_{self.out_channels}_hidden_{self.hidden_channels}_layers_{self.num_layers}_dropout_{self.dropout_rate}.png')
        torch.save(self.model, f'{self.directory}/model_state_dict_out_channels_{self.out_channels}_hidden_{self.hidden_channels}_layers_{self.num_layers}_dropout_{self.dropout_rate}.pth')
        print(f'Model saved as {self.directory}/model_state_dict_out_channels_{self.out_channels}_hidden_{self.hidden_channels}_layers_{self.num_layers}_dropout_{self.dropout_rate}.pth')
        return losses

    def train_clusterGCN(self, graph, epochs=500):
        data = self.preprocess_graph(graph)
        loader = self.cluster_GCN(data)
        losses = [] 
        final_embeddings = []
        self.model.train()
        for epoch in range(epochs):  
            total_loss = 0
            epoch_embeddings = [] 
            for batch_data in loader:
                x = batch_data.x.to(self.device)
                edge_index = batch_data.edge_index.to(self.device)
                self.optimizer.zero_grad() 
                z = self.model.encode(x, edge_index)
                loss = self.model.recon_loss(z, edge_index)
                loss.backward()
                self.optimizer.step()
                epoch_embeddings.append(z.detach().cpu().numpy())
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")
            if epoch == epochs - 1:
                final_embeddings = np.concatenate(epoch_embeddings, axis=0)
        torch.save(torch.from_numpy(final_embeddings), f'{self.directory}/embedding_out_channels_{self.out_channels}_hidden_{self.hidden_channels}_layers_{self.num_layers}_dropout_{self.dropout_rate}.pt')
        self.plot_learning_curve(losses, f'{self.directory}/learning_curve_out_channels_{self.out_channels}_hidden_{self.hidden_channels}_layers_{self.num_layers}_dropout_{self.dropout_rate}.png')
        torch.save(self.model, f'{self.directory}/model_state_dict_out_channels_{self.out_channels}_hidden_{self.hidden_channels}_layers_{self.num_layers}_dropout_{self.dropout_rate}.pth')
        print(f'Model saved as {self.directory}/model_state_dict_out_channels_{self.out_channels}_hidden_{self.hidden_channels}_layers_{self.num_layers}_dropout_{self.dropout_rate}.pth')
        return losses

    def train_without_sampling(self, graph, epochs=500, batch_size=32):
        if self.preprocessing:
            data = self.preprocess_graph(graph)
        loader = DataLoader([data], batch_size=batch_size)
        self.model.train()
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            all_z = []
            for batch_data in loader:
                x = batch_data.x.to(self.device)
                edge_index = batch_data.edge_index.to(self.device)
                self.optimizer.zero_grad()
                z = self.model.encode(x, edge_index)
                loss = self.model.recon_loss(z, edge_index)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                all_z.append(z.detach().cpu()) 
            avg_loss = total_loss / len(loader)
            losses.append(avg_loss)
            print(f'Epoch {epoch+1}, Average Loss: {avg_loss}')
            if epoch == epochs - 1:
               z_to_save = torch.cat(all_z, dim=0)
               torch.save(z_to_save, f'{self.directory}/without_sampling_epoch_{epoch+1}_z_out_channels_{self.out_channels}_hidden_{self.hidden_channels}_layers_{self.num_layers}_dropout_{self.dropout_rate}.pt')
        self.plot_learning_curve(losses, f'{self.directory}/without_sampling_learning_curve_out_channels_{self.out_channels}_hidden_{self.hidden_channels}_layers_{self.num_layers}_dropout_{self.dropout_rate}.png')
        torch.save(self.model, f'{self.directory}/without_sampling_model_state_dict_out_channels_{self.out_channels}_hidden_{self.hidden_channels}_layers_{self.num_layers}_dropout_{self.dropout_rate}.pth')
        print(f'Model saved as {self.directory}/without_sampling_model_state_dict_out_channels_{self.out_channels}_hidden_{self.hidden_channels}_layers_{self.num_layers}_dropout_{self.dropout_rate}.pth')
        return losses

def train_with_params(sampling_method, params, in_channels, graph):
    results = []
    for out_channels in params['out_channels']:
        for hidden_channels in filter(lambda x: in_channels <= x <= out_channels, params['hidden_channels']):
            for num_layers in params['num_layers']:
                for dropout_rate in params['dropout_rate']:
                    for num_nodes in params['num_nodes']:
                        if sampling_method == 'random_walk':
                            for restart_prob in params['p']:
                                print(f"Training with params: out_channels={out_channels}, hidden_channels={hidden_channels}, num_layers={num_layers}, dropout_rate={dropout_rate}, num_nodes={num_nodes}, restart_prob={restart_prob}")
                                sampling_params = {'p': restart_prob, 'num_nodes': num_nodes}
                                pipeline = GAEPipeline(
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    hidden_channels=hidden_channels,
                                    num_layers=num_layers,
                                    dropout_rate=dropout_rate,
                                    sampling_method=sampling_method,
                                    **sampling_params
                                )
                                try:
                                    losses = pipeline.train(graph, epochs=500, batch_size=32)
                                    results.append((sampling_method, out_channels, hidden_channels, num_layers, dropout_rate, num_nodes, restart_prob, losses))
                                except Exception as e:
                                    print(f"Error during training with random_walk: {e}")
                        elif sampling_method == 'forest_fire':
                            for p in params['p']:
                                print(f"Training with params: out_channels={out_channels}, hidden_channels={hidden_channels}, num_layers={num_layers}, dropout_rate={dropout_rate}, num_nodes={num_nodes}, p={p}")
                                sampling_params = {'p': p, 'num_nodes': num_nodes}
                                pipeline = GAEPipeline(
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    hidden_channels=hidden_channels,
                                    num_layers=num_layers,
                                    dropout_rate=dropout_rate,
                                    sampling_method=sampling_method,
                                    **sampling_params
                                )
                                try:
                                    losses = pipeline.train(graph, epochs=500, batch_size=32)
                                    results.append((sampling_method, out_channels, hidden_channels, num_layers, dropout_rate, num_nodes, p, losses))
                                except Exception as e:
                                    print(f"Error during training with forest_fire: {e}")
                        elif sampling_method == 'clusterGCN':
                            print(f"Training with params: out_channels={out_channels}, hidden_channels={hidden_channels}, num_layers={num_layers}, dropout_rate={dropout_rate}")
                            pipeline = GAEPipeline(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                hidden_channels=hidden_channels,
                                num_layers=num_layers,
                                dropout_rate=dropout_rate,
                                sampling_method=sampling_method
                            )
                            try:
                                losses = pipeline.train_clusterGCN(graph, epochs=500)
                                results.append((sampling_method, out_channels, hidden_channels, num_layers, dropout_rate, losses))
                            except Exception as e:
                                print(f"Error during training with clusterGCN: {e}")
    return results

# Load the graph
pipeline_st = GAEPipeline(in_channels=10, out_channels=32, hidden_channels=50, num_layers=1, dropout_rate=0, sampling_method='random_walk')
graph = pipeline_st.load_graph_from_pickle('combined_graph_latest.pkl')

# Determine max_length for in_channels
max_length = max(len(node_features) for _, node_features in graph.nodes(data=True))
in_channels = max_length

# Define Parameter Combinations
random_walk_params = {
    'out_channels': [64],
    'hidden_channels': [60],
    'num_layers': [6],
    'dropout_rate': [0.2],
    'p': [0.2],
    'num_nodes': [10000]
}

forest_fire_params = {
    'out_channels': [64],
    'hidden_channels': [60],
    'num_layers': [6],
    'dropout_rate': [0.2],
    'p': [0.6],
    'num_nodes': [10000]
}

cluster_gcn_params = {
    'out_channels': [64],
    'hidden_channels': [60],
    'num_layers': [6],
    'dropout_rate': [0.2],
    'num_nodes': [1000]  # Cluster-GCN does not use num_nodes parameter
}

# # Train with random walk parameters
# random_walk_results = train_with_params('random_walk', random_walk_params, in_channels, graph)

# # Train with forest fire parameters
# forest_fire_results = train_with_params('forest_fire', forest_fire_params, in_channels, graph)

# # Train with cluster-GCN parameters
# cluster_gcn_results = train_with_params('clusterGCN', cluster_gcn_params, in_channels, graph)

# Print results
# print("Random Walk Results:")
# for result in random_walk_results:
#     print(result)

# print("Forest Fire Results:")
# for result in forest_fire_results:
#     print(result)

# print("Cluster-GCN Results:")
# for result in cluster_gcn_results:
#     print(result)

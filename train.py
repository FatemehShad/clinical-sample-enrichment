import os
import argparse
import json
import hashlib
import platform
from importlib.metadata import version
from pathlib import Path
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

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_GRAPH = REPO_ROOT / 'data' / 'combined_graph_latest.pkl'


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


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
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers, dropout_rate, sampling_method='random_walk', preprocessing=True, seed=42, output_dir=None, device=None, feature_keys=None, **sampling_params):
        self.seed = seed
        seed_everything(seed)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.in_channels = in_channels
        self.feature_keys = list(feature_keys) if feature_keys is not None else None
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.sampling_method_name = sampling_method
        self.sampling_method = self._get_sampling_method(sampling_method)
        self.preprocessing = preprocessing
        self.sampling_params = dict(sampling_params)

        # Create a directory name string that includes all relevant parameters
        params_str = '_'.join([f'{k}_{v}' for k, v in sampling_params.items()])
        self.directory = str(Path(output_dir or REPO_ROOT / "outputs") / f"{self.sampling_method_name}_out_{out_channels}_hidden_{hidden_channels}_layers_{num_layers}_dropout_{dropout_rate}_seed_{seed}_{params_str}")
        
        # Create directories based on hyperparameters
        os.makedirs(self.directory, exist_ok=True)
        os.makedirs(f'{self.directory}/sampled_graphs', exist_ok=True)
        
        self.encoder = GCNEncoder(in_channels, hidden_channels, out_channels, num_layers, dropout_rate).to(self.device)
        self.model = GAE(self.encoder).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def recon_loss(self, predicted_adj, true_adj):
        loss = F.binary_cross_entropy(predicted_adj, true_adj)
        return loss

    def load_graph_from_pickle(self, file_path=DEFAULT_GRAPH):
        file_path = Path(file_path)
        if not file_path.is_absolute() and not file_path.exists():
            candidate = REPO_ROOT / file_path
            file_path = candidate if candidate.exists() else REPO_ROOT / 'data' / file_path
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def save_graph_to_pickle(self, graph, filename):
        with open(filename, 'wb') as f:
            pickle.dump(graph, f)

    def _get_sampling_method(self, method_name):
        """Dynamically selects the sampling method."""
        if method_name == 'none':
            return None
        elif method_name == 'random_walk':
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
        device = self.device
        batch = Batch.from_data_list(batch)
        batch.to(device)
        return batch

    def preprocess_features(self, features, max_length):
        processed_features = []
        for feature in features:
            try:
                value = float(feature)
                processed_features.append(value if np.isfinite(value) else 0.0)
            except (ValueError, TypeError):
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
        if not G.number_of_nodes():
            raise ValueError('Cannot train on an empty graph')
        edge_pairs = [list(map(mapping.get, edge)) for edge in G.edges()]
        if not G.is_directed():
            edge_pairs += [[target, source] for source, target in edge_pairs if source != target]
        edges = torch.tensor(edge_pairs, dtype=torch.long).reshape(-1, 2).t().contiguous()

        if G.nodes():
            if self.feature_keys is None:
                self.feature_keys = list(next(iter(G.nodes(data=True)))[1])
            feature_keys = self.feature_keys
            if len(feature_keys) > self.in_channels:
                raise ValueError('Feature schema is wider than in_channels')
            max_length = self.in_channels
        
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
        params = dict(self.sampling_params)
        num_nodes = params.pop('num_nodes', 40000)
        model = RandomWalkWithRestartSampler(number_of_nodes=num_nodes, seed=self.seed, **params)
        new_graph = model.sample(graph)
        self.save_graph_to_pickle(new_graph, f'{self.directory}/sampled_graphs/{self.sampling_method_name}_sampled_graph.pkl')
        return new_graph

    def forest_fire(self, graph):
        graph = self.convert_node_labels_to_integers(graph)
        params = dict(self.sampling_params)
        num_nodes = params.pop('num_nodes', 30000)
        model = ForestFireSampler(number_of_nodes=num_nodes, seed=self.seed, **params)
        new_graph = model.sample(graph)
        self.save_graph_to_pickle(new_graph, f'{self.directory}/sampled_graphs/{self.sampling_method_name}_sampled_graph.pkl')
        return new_graph

    def cluster_GCN(self, data):
        device = self.device
        torch.manual_seed(self.seed)
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
        if self.feature_keys is None:
            self.feature_keys = list(next(iter(graph.nodes(data=True)))[1])
        sampled_subgraph = self.sampling_method(graph)
        data = self.preprocess_graph(sampled_subgraph) if self.preprocessing else sampled_subgraph
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
               self.model.eval()
               with torch.no_grad():
                   z_to_save = torch.cat([self.model.encode(batch.x.to(self.device), batch.edge_index.to(self.device)).cpu()
                                          for batch in loader], dim=0)
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
                self.model.eval()
                final_embeddings = np.empty((data.num_nodes, self.out_channels), dtype=np.float32)
                offset = 0
                with torch.no_grad():
                    for part in ClusterLoader(loader.cluster_data, batch_size=1, shuffle=False):
                        encoded = self.model.encode(part.x.to(self.device), part.edge_index.to(self.device)).cpu().numpy()
                        node_ids = loader.cluster_data.partition.node_perm[offset:offset + part.num_nodes].numpy()
                        final_embeddings[node_ids] = encoded
                        offset += part.num_nodes
        torch.save(torch.from_numpy(final_embeddings), f'{self.directory}/embedding_out_channels_{self.out_channels}_hidden_{self.hidden_channels}_layers_{self.num_layers}_dropout_{self.dropout_rate}.pt')
        self.plot_learning_curve(losses, f'{self.directory}/learning_curve_out_channels_{self.out_channels}_hidden_{self.hidden_channels}_layers_{self.num_layers}_dropout_{self.dropout_rate}.png')
        torch.save(self.model, f'{self.directory}/model_state_dict_out_channels_{self.out_channels}_hidden_{self.hidden_channels}_layers_{self.num_layers}_dropout_{self.dropout_rate}.pth')
        print(f'Model saved as {self.directory}/model_state_dict_out_channels_{self.out_channels}_hidden_{self.hidden_channels}_layers_{self.num_layers}_dropout_{self.dropout_rate}.pth')
        return losses

    def train_without_sampling(self, graph, epochs=500, batch_size=32):
        data = self.preprocess_graph(graph) if self.preprocessing else graph
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
               self.model.eval()
               with torch.no_grad():
                   z_to_save = torch.cat([self.model.encode(batch.x.to(self.device), batch.edge_index.to(self.device)).cpu()
                                          for batch in loader], dim=0)
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

def main(argv=None):
    parser = argparse.ArgumentParser(description='Train a seeded graph autoencoder on bundled data.')
    parser.add_argument('--graph', type=Path, default=DEFAULT_GRAPH)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'outputs')
    parser.add_argument('--sampling', choices=['none', 'random_walk', 'forest_fire', 'clusterGCN'], default='none')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--max-nodes', type=int, default=200, help='Deterministic induced subgraph; 0 uses the full graph.')
    parser.add_argument('--sample-nodes', type=int, default=100)
    parser.add_argument('--hidden-channels', type=int, default=60)
    parser.add_argument('--out-channels', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.max_nodes < 0 or args.num_layers < 2:
        parser.error('epochs must be positive, max-nodes nonnegative, and num-layers at least 2')
    seed_everything(args.seed)
    with open(args.graph, 'rb') as handle:
        graph = pickle.load(handle)
    feature_keys = sorted(set().union(*(features.keys() for _, features in graph.nodes(data=True))))
    if args.max_nodes:
        graph = graph.subgraph(list(graph.nodes())[:args.max_nodes]).copy()
    if not graph.number_of_nodes() or not graph.number_of_edges():
        parser.error('graph must have nodes and edges')
    if args.sampling == 'random_walk':
        graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
    if args.sampling in ('random_walk', 'forest_fire') and not 0 < args.sample_nodes <= graph.number_of_nodes():
        parser.error('sample-nodes must be positive and no larger than the available graph')
    in_channels = len(feature_keys)
    pipeline = GAEPipeline(in_channels=in_channels, out_channels=args.out_channels,
        hidden_channels=args.hidden_channels, num_layers=args.num_layers,
        dropout_rate=args.dropout, sampling_method=args.sampling,
        seed=args.seed, output_dir=args.output_dir, device=args.device, feature_keys=feature_keys,
        **({'num_nodes': args.sample_nodes} if args.sampling in ('random_walk', 'forest_fire') else {}))
    if args.sampling == 'none':
        losses = pipeline.train_without_sampling(graph, epochs=args.epochs)
    elif args.sampling == 'clusterGCN':
        losses = pipeline.train_clusterGCN(graph, epochs=args.epochs)
    else:
        losses = pipeline.train(graph, epochs=args.epochs)
    metadata = {**vars(args), 'graph': str(args.graph.resolve()), 'output_dir': str(args.output_dir.resolve()),
                'nodes': graph.number_of_nodes(), 'edges': graph.number_of_edges(), 'losses': losses, 'feature_keys': feature_keys,
                'torch_version': torch.__version__, 'numpy_version': np.__version__,
                'python_version': platform.python_version(),
                'graph_sha256': hashlib.sha256(args.graph.read_bytes()).hexdigest(),
                'dependencies': {name: version(name) for name in ['torch-geometric', 'networkx', 'scikit-learn', 'littleballoffur']}}
    torch.save({'model_state_dict': pipeline.model.state_dict(),
                'in_channels': in_channels, 'hidden_channels': args.hidden_channels,
                'out_channels': args.out_channels, 'num_layers': args.num_layers,
                'dropout_rate': args.dropout, 'seed': args.seed, 'feature_keys': feature_keys}, Path(pipeline.directory, 'checkpoint.pt'))
    Path(pipeline.directory, 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return losses


if __name__ == '__main__':
    main()

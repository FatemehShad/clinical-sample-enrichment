import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import networkx as nx
import numpy as np
import torch

from train import GAEPipeline, main
from utils_functions import calculate_mutual_info_score, calculate_similarity_matrix


class ReproducibilityTests(unittest.TestCase):
    def graph(self):
        graph = nx.cycle_graph(20)
        for node in graph:
            graph.nodes[node].update(a=float(node), b=float(node % 3))
        return graph

    def test_import_has_no_filesystem_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, '-c', 'import train; import utils_functions'], cwd=tmp,
                                    env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])},
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_seeded_training_and_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            losses, states = [], []
            for run in range(2):
                pipeline = GAEPipeline(2, 4, 8, 2, 0.2, sampling_method='random_walk',
                                       seed=42, output_dir=Path(tmp) / str(run), device='cpu', num_nodes=10)
                before = dict(pipeline.sampling_params)
                first = pipeline.random_walk(self.graph())
                second = pipeline.random_walk(self.graph())
                self.assertEqual(list(first.nodes()), list(second.nodes()))
                self.assertEqual(pipeline.sampling_params, before)
                losses.append(pipeline.train(self.graph(), epochs=2))
                states.append({k: v.clone() for k, v in pipeline.model.state_dict().items()})
            self.assertEqual(losses[0], losses[1])
            self.assertTrue(np.isfinite(losses).all())
            for key in states[0]:
                self.assertTrue(torch.equal(states[0][key], states[1][key]), key)

    def test_bundled_graph_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            losses = main(['--epochs', '1', '--max-nodes', '50', '--num-layers', '2',
                           '--hidden-channels', '16', '--out-channels', '8', '--output-dir', tmp])
            self.assertEqual(len(losses), 1)
            self.assertTrue(np.isfinite(losses).all())
            metadata = json.loads(next(Path(tmp).rglob('run.json')).read_text())
            self.assertEqual(metadata['seed'], 42)
            self.assertEqual(metadata['nodes'], 50)
            checkpoint = torch.load(next(Path(tmp).rglob('checkpoint.pt')), weights_only=True)
            from torch_geometric.nn import GAE
            from train import GCNEncoder
            model = GAE(GCNEncoder(*(checkpoint[key] for key in
                        ('in_channels', 'hidden_channels', 'out_channels', 'num_layers', 'dropout_rate'))))
            model.load_state_dict(checkpoint['model_state_dict'])
            self.assertEqual(len(metadata['graph_sha256']), 64)
            pipeline = GAEPipeline(checkpoint['in_channels'], checkpoint['out_channels'],
                                   checkpoint['hidden_channels'], checkpoint['num_layers'],
                                   checkpoint['dropout_rate'], output_dir=Path(tmp) / 'inference',
                                   feature_keys=checkpoint['feature_keys'], device='cpu')
            graph = pipeline.load_graph_from_pickle()
            graph = graph.subgraph(list(graph.nodes())[:50]).copy()
            data = pipeline.preprocess_graph(graph)
            model.eval()
            with torch.no_grad():
                expected = model.encode(data.x, data.edge_index)
            saved = torch.load(next(Path(tmp).rglob('without_sampling_epoch_*.pt')), weights_only=True)
            torch.testing.assert_close(saved, expected)

    @unittest.skipUnless(importlib.util.find_spec('pyg_lib'), 'optional pyg-lib extension is not installed')
    def test_cluster_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = nx.cycle_graph(80)
            for node in graph:
                graph.nodes[node].update(a=float(node), b=float(node % 3))
            losses, embeddings = [], []
            for run in range(2):
                pipeline = GAEPipeline(2, 4, 8, 2, 0.2, sampling_method='clusterGCN',
                                       seed=42, output_dir=Path(tmp) / str(run), device='cpu')
                losses.append(pipeline.train_clusterGCN(graph, epochs=1))
                saved = torch.load(next(Path(pipeline.directory).glob('embedding_*.pt')), weights_only=True)
                self.assertEqual(tuple(saved.shape), (80, 4))
                self.assertTrue(torch.isfinite(saved).all())
                # Use the partition's explicit original IDs to verify saved row ordering.
                from torch_geometric.loader import ClusterLoader
                data = pipeline.preprocess_graph(graph)
                data.original_id = torch.arange(data.num_nodes)
                parts = pipeline.cluster_GCN(data).cluster_data
                with torch.no_grad():
                    for part in ClusterLoader(parts, batch_size=1, shuffle=False):
                        expected = pipeline.model.encode(part.x, part.edge_index)
                        torch.testing.assert_close(saved[part.original_id], expected)
                embeddings.append(saved)
            self.assertEqual(losses[0], losses[1])
            self.assertTrue(torch.equal(embeddings[0], embeddings[1]))

    def test_feature_schema_survives_missing_attributes(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = GAEPipeline(2, 4, 8, 2, 0.0, seed=42, output_dir=tmp,
                                   device='cpu', feature_keys=['a', 'b'])
            graph = nx.path_graph(3)
            graph.nodes[0]['b'] = 2.0
            graph.nodes[1].update(b=4.0, a=1.0)
            graph.nodes[2]['a'] = 2.0
            data = pipeline.preprocess_graph(graph)
            self.assertEqual(tuple(data.x.shape), (3, 2))
            expected = np.array([[0., 2.], [1., 4.], [2., 0.]])
            expected = (expected - expected.mean(0)) / expected.std(0)
            np.testing.assert_allclose(data.x.numpy(), expected, rtol=1e-6, atol=1e-6)
            self.assertEqual(data.num_edges, 4)

    def test_analysis(self):
        similarity = calculate_similarity_matrix(np.array([[1., 0.], [0., 1.]]))
        np.testing.assert_allclose(similarity, np.eye(2))
        self.assertGreater(calculate_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0]), 0)


if __name__ == '__main__':
    unittest.main()

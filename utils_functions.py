
import networkx as nx
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mutual_info_score
import os
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from torch_geometric.data import Data
import pickle
import gprofiler as gp
from sklearn.preprocessing import normalize
from pyemd import emd
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn_extra.cluster import KMedoids
from scipy.linalg import norm
import math
import textwrap
from matplotlib_venn import venn2

def get_protein_indices_in_pyg(graph, existing_proteins_list):
    """
    This function takes a NetworkX graph and a list of existing proteins,
    and returns the indices of the desired proteins in the PyTorch Geometric tensor.

    Parameters:
    - graph: NetworkX graph
    - existing_proteins_list: List of existing proteins

    Returns:
    - protein_indices_in_pyg: List of indices of desired proteins in PyTorch Geometric tensor
    """
    existing_proteins_set = set(existing_proteins_list)

    mapping = {node: i for i, node in enumerate(graph.nodes())}
    schema_attr = nx.get_node_attributes(graph, "schemaClass")
    name_attr = nx.get_node_attributes(graph, "name")

    proteins = []
    processed_proteins = set()

    # Iterate over each node and its schema attribute
    for node, schema in schema_attr.items():
        if schema == 'EntityWithAccessionedSequence':
            name_string = name_attr.get(node, "")
            if name_string:
                name_list = [name.strip(' "[]') for name in name_string.split(',')]
                matched_names = set(name_list) & existing_proteins_set
                if matched_names:
                    new_matches = matched_names - processed_proteins
                    if new_matches:
                        proteins.append(node)
                        processed_proteins.update(new_matches)

    # Convert node identifiers to indices in the PyTorch Geometric tensor
    protein_indices_in_pyg = [mapping[node] for node in proteins if node in mapping]

    return protein_indices_in_pyg


def map_tensor_indices_to_names(indices, pipeline, graph_path='combined_graph_latest.pkl'):
    # Load the graph
    graph = pipeline.load_graph_from_pickle(graph_path)
    
    # Create a mapping from node to index and reverse mapping from index to node
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    reverse_mapping = {i: node for node, i in mapping.items()}
    
    # Get node attributes for name
    name_attr = nx.get_node_attributes(graph, "name")
    
    # Find the names for the given indices
    names = []
    for index in indices:
        node = reverse_mapping.get(index)
        if node:
            name_string = name_attr.get(node, "")
            name_list = [name.strip(' "[]') for name in name_string.split(',')]
            if name_list:
                last_name = name_list[-1]  # Choose the last name in the list
                names.append(last_name)
    
    return names


def map_names_to_tensor_indices(names, pipeline, existing_proteins_list, graph_path='combined_graph_latest.pkl'):
    # Load the graph
    graph = pipeline.load_graph_from_pickle(graph_path)
    
    # Convert the existing proteins list to a set
    existing_proteins_set = set(existing_proteins_list)
    
    # Create a mapping from node to index
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    
    # Get node attributes for schemaClass and name
    schema_attr = nx.get_node_attributes(graph, "schemaClass")
    name_attr = nx.get_node_attributes(graph, "name")
    
    # Initialize containers for proteins and processed proteins
    proteins = []
    processed_proteins = set()
    
    # Iterate over each node and its schema attribute
    for node, schema in schema_attr.items():
        if schema == 'EntityWithAccessionedSequence':
            name_string = name_attr.get(node, "")
            if name_string:
                name_list = [name.strip(' "[]') for name in name_string.split(',')]
                matched_names = set(name_list) & existing_proteins_set
                if matched_names:
                    new_matches = matched_names - processed_proteins
                    if new_matches:
                        proteins.append(node)
                        processed_proteins.update(new_matches)
    
    # Convert node identifiers to indices in the PyTorch Geometric tensor
    protein_indices_in_pyg = [mapping[node] for node in proteins if node in mapping]
    # Find the indices for the given names
    name_indices = []
    for name in names:
        name_index = None
        for node in proteins:
            name_string = name_attr.get(node, "")
            name_list = [name.strip(' "[]') for name in name_string.split(',')]
            if name_string and name in name_list:
                name_index = mapping.get(node)
                break
        name_indices.append(name_index)
    
    return name_indices


def select_features_and_predict(train_data_path, target, processed_proteins, num_best_features):
    # Load data
    train = pd.read_csv(train_data_path, index_col=0)

    # Set up features
    features = train.columns.tolist()
    features.remove(target)
    matched_set = processed_proteins.intersection(features)
    features = list(matched_set)
    print("Matched items:", len(features))

    best = []
    all_features = features[:]  

    loo = LeaveOneOut()

    # Feature selection loop
    while len(best) < num_best_features:
        max_acc = 0
        remaining_features = list(set(features) - set(best))
        for new_column in remaining_features:
            accuracies = []
            for train_index, test_index in loo.split(train):
                loo_train, loo_test = train.iloc[train_index], train.iloc[test_index]
                model = LogisticRegression()
                if best:
                    model.fit(loo_train[best + [new_column]], loo_train[target])
                    target_predicted = model.predict(loo_test[best + [new_column]])
                else:
                    model.fit(loo_train[[new_column]], loo_train[target])
                    target_predicted = model.predict(loo_test[[new_column]])
                acc = metrics.accuracy_score(loo_test[target], target_predicted)
                accuracies.append(acc)
            avg_acc = np.mean(accuracies)
            if avg_acc > max_acc:
                max_acc = avg_acc
                max_column = new_column

        best.append(max_column)
        features.remove(max_column)
        print('Best columns:', best)
        print('Average LOO-CV Accuracy:', max_acc)

    return best


def calculate_similarity_matrix(embeddings):

    # Calculate the cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

def calculate_mutual_info_score(labels1, labels2):
    mi_score = mutual_info_score(labels1, labels2)

def create_directories(directories):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Load models from saved files and generate embeddings
def generate_embeddings_from_models(parent_dir, graph_data, pipeline):
    models = []
    
    # Iterate through each subdirectory in the parent directory
    for subdir in os.listdir(parent_dir):
        subdir_path = os.path.join(parent_dir, subdir)
        
        if os.path.isdir(subdir_path):
            # Iterate through files in each subdirectory
            for file in os.listdir(subdir_path):
                if file.endswith('.pth'):
                    model_path = os.path.join(subdir_path, file)
                    # print(model_path)
                    
                    # Load the model
                    model = torch.load(model_path, map_location=torch.device('cpu'))
                    model.eval()
                    
                    with torch.no_grad():
                        device = next(model.parameters()).device
                        embeddings = model.encode(graph_data.x.to(device), graph_data.edge_index.to(device)).cpu().numpy()
                        models.append((model_path, embeddings)) 
    return models

# def perform_enrichment_analysis(protein_clusters, pipeline, n_clusters, output_dir, clustering_method, organism='mmusculus', top_n=5):
#     gp_client = gp.GProfiler(return_dataframe=True)
#     os.makedirs(output_dir, exist_ok=True)

#     enrichment_results = []
#     for cluster_id in range(n_clusters):
#         proteins_in_cluster = protein_clusters[protein_clusters['Cluster'] == cluster_id]['ProteinID'].tolist()
#         prot_names = map_tensor_indices_to_names(proteins_in_cluster, pipeline)
        
#         if not prot_names:
#             continue
        
#         results = gp_client.profile(organism=organism, query=prot_names)
#         results['Cluster'] = cluster_id
#         available_sources = results['source'].unique()
#         print(f"Available sources for cluster {cluster_id}: {available_sources}")
#         # # Filter results to only include Reactome source
#         # results = results[results['source'] == 'REAC']
#         enrichment_results.append(results)

#     if enrichment_results:
#         enrichment_results_df = pd.concat(enrichment_results, ignore_index=True)
#         enrichment_results_df.to_csv(os.path.join(output_dir, f'enrichment_results_{clustering_method}.csv'), index=False)

#         significant_results = enrichment_results_df[enrichment_results_df['p_value'] < 0.05]
#         significant_results.to_csv(os.path.join(output_dir, f'significant_results_{clustering_method}.csv'), index=False)

#         top_enrichments = significant_results.groupby('Cluster').apply(lambda x: x.nsmallest(top_n, 'p_value')).reset_index(drop=True)
#         top_enrichments['log_p_value'] = -np.log10(top_enrichments['p_value'])

#         plt.figure(figsize=(12, 8))
#         sns.barplot(data=top_enrichments, x='log_p_value', y='name', hue='Cluster', dodge=True, palette="Set2")
#         plt.xlabel('-log10 P-value', fontsize=18)
#         plt.ylabel('Pathway', fontsize=18)
#         # plt.title(f'Top Enriched Pathways per Cluster ({clustering_method})', fontsize=25)
#         plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=18)
#         plt.xticks(fontsize=18)
#         plt.yticks(fontsize=18)
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, f'top_enriched_pathways_barplot_{clustering_method}.eps'), dpi=300)
#         plt.close()

#         heatmap_data = top_enrichments.pivot('name', 'Cluster', 'p_value').fillna(1)
#         plt.figure(figsize=(12, 8))
#         sns.heatmap(-np.log10(heatmap_data), cmap='viridis', annot=True, annot_kws={"size": 18}, cbar_kws={'label': '-log10 p-value'})
#         plt.xlabel('Cluster', fontsize=18)
#         plt.ylabel('Pathway', fontsize=18)
#         # plt.title(f'Heatmap of Enriched Pathways (-log10 p-value) ({clustering_method})', fontsize=25)
#         plt.xticks(fontsize=18)
#         plt.yticks(fontsize=18)
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, f'enriched_pathways_heatmap_{clustering_method}.eps'), dpi=300)
#         plt.close()

#     else:
#         print(f"No enrichment results for clustering method {clustering_method}")








# def perform_enrichment_analysis(protein_clusters, pipeline, n_clusters, output_dir, clustering_method, organism='mmusculus', top_n=10):
#     gp_client = gp.GProfiler(return_dataframe=True)
#     os.makedirs(output_dir, exist_ok=True)

#     enrichment_results = []
#     for cluster_id in range(n_clusters):
#         proteins_in_cluster = protein_clusters[protein_clusters['Cluster'] == cluster_id]['ProteinID'].tolist()
#         prot_names = map_tensor_indices_to_names(proteins_in_cluster, pipeline)
        
#         if not prot_names:
#             continue
        
#         results = gp_client.profile(organism=organism, query=prot_names)
#         results['Cluster'] = cluster_id
#         available_sources = results['source'].unique()
#         print(f"Available sources for cluster {cluster_id}: {available_sources}")
#         enrichment_results.append(results)

#     if enrichment_results:
#         enrichment_results_df = pd.concat(enrichment_results, ignore_index=True)
#         enrichment_results_df.to_csv(os.path.join(output_dir, f'enrichment_results_{clustering_method}.csv'), index=False)

#         significant_results = enrichment_results_df[enrichment_results_df['p_value'] < 0.05]
#         significant_results.to_csv(os.path.join(output_dir, f'significant_results_{clustering_method}.csv'), index=False)

#         # Compute -log10(p-value)
#         significant_results['log_p_value'] = -np.log10(significant_results['p_value'])

#         # Get the top N terms per cluster
#         top_enrichments = significant_results.groupby('Cluster').apply(lambda x: x.nsmallest(top_n, 'p_value')).reset_index(drop=True)
        
#         # Create a heatmap for top N terms per cluster
#         top_terms = top_enrichments.pivot_table(index='name', columns='Cluster', values='log_p_value', fill_value=0)

#         plt.figure(figsize=(12, 8))  # Increase figure size
#         sns.heatmap(top_terms, cmap='viridis', annot=True, fmt='.2f', cbar_kws={'label': '-log10(p-value)'}, annot_kws={"size": 8})
#         plt.xlabel('Cluster')
#         plt.ylabel('Terms')
#         plt.title('Heatmap of Enrichment Scores (-log10(p-value))')
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, f'enriched_pathways_heatmap_{clustering_method}.png'), dpi=300)
#         plt.close()

#         # Bar Plot for All Clusters
#         enrichment_results_df['-log10(p_value)'] = -np.log10(enrichment_results_df['p_value'])
#         data = enrichment_results_df[['source', 'name', '-log10(p_value)', 'Cluster']]

#         sns.set(style="whitegrid")
#         plt.figure(figsize=(14, 7))
#         sns.barplot(x='source', y='-log10(p_value)', hue='Cluster', data=data, ci=None, palette="Set2")
#         plt.xticks(rotation=45, ha='right')
#         plt.xlabel('Source')
#         plt.ylabel('-log10(p-adj)')
#         plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.title('g:Profiler Results for All Clusters')
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, f'gprofiler_bar_plot_all_clusters.eps'), dpi=300)
#         plt.close()

#         # Bar Plot for Top Enrichment Results per Cluster
#         plt.figure(figsize=(14, 7))
#         sns.barplot(x='log_p_value', y='name', hue='Cluster', data=top_enrichments, dodge=True, palette="Set2")
#         plt.xlabel('-log10 P-value', fontsize=12)
#         plt.ylabel('Pathway', fontsize=12)
#         plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
#         plt.xticks(fontsize=12)
#         plt.yticks(fontsize=12)
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, f'top_enriched_pathways_barplot_{clustering_method}.png'), dpi=300)
#         plt.close()

#         # Dot Plot
#         plt.figure(figsize=(14, 7))
#         sns.scatterplot(data=enrichment_results_df, x='source', y='-log10(p_value)', hue='Cluster', size='intersection_size', sizes=(20, 200), alpha=0.6)
#         plt.xticks(rotation=45, ha='right')
#         plt.xlabel('Source')
#         plt.ylabel('-log10(p-adj)')
#         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.title('Dot Plot of Enrichment Results')
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, f'gprofiler_dot_plot_all_clusters.eps'), dpi=300)
#         plt.close()

#         # Violin Plot
#         plt.figure(figsize=(14, 7))
#         sns.violinplot(data=enrichment_results_df, x='source', y='-log10(p_value)', hue='Cluster', split=True)
#         plt.xticks(rotation=45, ha='right')
#         plt.xlabel('Source')
#         plt.ylabel('-log10(p-adj)')
#         plt.title('Violin Plot of Enrichment Results')
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, f'gprofiler_violin_plot_all_clusters.eps'), dpi=300)
#         plt.close()

#         # Box Plot
#         plt.figure(figsize=(14, 7))
#         sns.boxplot(data=enrichment_results_df, x='source', y='-log10(p_value)', hue='Cluster')
#         plt.xticks(rotation=45, ha='right')
#         plt.xlabel('Source')
#         plt.ylabel('-log10(p-adj)')
#         plt.title('Box Plot of Enrichment Results')
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, f'gprofiler_box_plot_all_clusters.eps'), dpi=300)
#         plt.close()

#         # Bubble Plot
#         plt.figure(figsize=(14, 7))
#         sns.scatterplot(data=enrichment_results_df, x='source', y='-log10(p_value)', hue='Cluster', size='intersection_size', sizes=(20, 200), alpha=0.6)
#         plt.xticks(rotation=45, ha='right')
#         plt.xlabel('Source')
#         plt.ylabel('-log10(p-adj)')
#         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.title('Bubble Plot of Enrichment Results')
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, f'gprofiler_bubble_plot_all_clusters.eps'), dpi=300)
#         plt.close()

#         # Combined Manhattan Plot for All Clusters using sns.relplot
#         g = sns.relplot(
#             data=significant_results,
#             x='source', y='log_p_value', hue='Cluster', style='Cluster', size='log_p_value',
#             sizes=(100, 1000), alpha=0.6, palette='muted', kind='scatter', height=8, aspect=8, linewidth=0
#         )
#         g.set_xticklabels(rotation=45, ha='right')
#         g.set_axis_labels('Source', '-log10(p-adj)')
#         g._legend.set_bbox_to_anchor((1.05, 1))
#         g.fig.suptitle('Combined Manhattan Plot for All Clusters', y=1.03)
#         g.tight_layout()
#         g.savefig(os.path.join(output_dir, f'combined_manhattan_plot_all_clusters.eps'), dpi=300)

#     else:
#         print(f"No enrichment results for clustering method {clustering_method}")

# Novel Assosiations plot

def perform_enrichment_analysis(protein_clusters, pipeline, n_clusters, output_dir, clustering_method, organism='mmusculus', top_n=10):
    gp_client = gp.GProfiler(return_dataframe=True)
    enrichment_results = []
    
    for cluster_id in range(n_clusters):
        proteins_in_cluster = protein_clusters[protein_clusters['Cluster'] == cluster_id]['ProteinID'].tolist()
        prot_names = map_tensor_indices_to_names(proteins_in_cluster, pipeline)
        
        if not prot_names:
            continue
        
        results = gp_client.profile(organism=organism, query=prot_names)
        results['Cluster'] = cluster_id
        available_sources = results['source'].unique()
        print(f"Available sources for cluster {cluster_id}: {available_sources}")
        enrichment_results.append(results)

    if enrichment_results:
        enrichment_results_df = pd.concat(enrichment_results, ignore_index=True)
        enrichment_results_df.to_csv(os.path.join(output_dir, f'enrichment_results_{clustering_method}.csv'), index=False)

        significant_results = enrichment_results_df[enrichment_results_df['p_value'] < 0.05]
        significant_results.to_csv(os.path.join(output_dir, f'significant_results_{clustering_method}.csv'), index=False)

        # Compute -log10(p-value)
        significant_results['log_p_value'] = -np.log10(significant_results['p_value'])

        # Get the top N terms per cluster
        top_enrichments = significant_results.groupby('Cluster').apply(lambda x: x.nsmallest(top_n, 'p_value')).reset_index(drop=True)
        
        # Print the top 10 most enriched terms for each cluster
        for cluster_id in range(n_clusters):
            cluster_top_terms = top_enrichments[top_enrichments['Cluster'] == cluster_id]
            print(f"Top {top_n} enriched terms for Cluster {cluster_id}:")
            print(cluster_top_terms[['name', 'source', 'p_value', 'term_size']])
            print("\n")
        
        # Create a heatmap for top N terms per cluster
        top_terms = top_enrichments.pivot_table(index='name', columns='Cluster', values='log_p_value', fill_value=0)

        plt.figure(figsize=(12, 4))  # Adjust figure size
        ax = sns.heatmap(top_terms, cmap='viridis', annot=True, fmt='.2f', cbar_kws={'label': '-log10(p-value)'}, annot_kws={"size": 6, "weight": "bold"})
        
        ax.set_aspect(aspect=0.2)  # Adjust the aspect ratio (smaller value makes the heatmap bars wider)
        
        ax.set_xlabel('Cluster', fontsize=12)  # Increase font size for xlabel
        ax.set_ylabel('Terms', fontsize=12)    # Increase font size for ylabel
        ax.yaxis.label.set_size(12)
        ax.yaxis.set_tick_params(labelsize=12)  # Increase font size and pad to separate the labels
        
        # Increase font size of color bar label
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('-log10(p-value)', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'enriched_pathways_heatmap_{clustering_method}.eps'), dpi=300)
        plt.close()

        # Combined Manhattan Plot for All Clusters
        clusters = significant_results['Cluster'].unique()
        num_clusters = len(clusters)
        num_cols = 4  # Adjust the number of columns as needed
        num_rows = math.ceil(num_clusters / num_cols)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(30, num_rows * 8), sharey=True)
        axes = axes.flatten()

        for i, cluster_id in enumerate(clusters):
            cluster_data = significant_results[significant_results['Cluster'] == cluster_id]

            if not cluster_data.empty:
                cluster_data['-log10(p_value)'] = -np.log10(cluster_data['p_value'])
                data = cluster_data[['source', 'name', '-log10(p_value)', 'term_size']]

                # Assign colors based on 'source'
                color_mapping = {
                    'GO:CC': 'green', 'GO:BP': 'orange', 'GO:MF': 'red', 'TF': 'magenta',
                    'REAC': 'blue', 'KEGG': 'purple', 'HP': 'brown', 'WP': 'cyan', 'CORUM': 'black'
                }
                data['color'] = data['source'].map(color_mapping)

                # Sort the data within each category based on term_size
                data = data.sort_values(by=['source', 'term_size'], ascending=[True, False]).reset_index(drop=True)

                # Create numeric categories for x-axis with increased spacing
                categories = data['source'].unique()
                category_map = {category: i * 100 for i, category in enumerate(categories)}  # Increased spacing
                data['category_num'] = data['source'].map(category_map)

                # Add jitter to the x-axis within each category for better dispersion
                data['jitter'] = data.groupby('source').cumcount().apply(lambda x: np.random.uniform(-40, 40))

                # Adjust x-coordinate with jitter
                data['adjusted_x'] = data['category_num'] + data['jitter']

                # Scale the term_size for better visualization
                size_scaling_factor = 500 / data['term_size'].max()  # Reduced scaling factor for smaller bubbles
                data['scaled_size'] = data['term_size'] * size_scaling_factor

                # Apply a minimum size threshold
                min_size = 20  # Reduce the minimum size threshold
                data['scaled_size'] = data['scaled_size'].apply(lambda x: max(x, min_size))

                ax = axes[i]

                # Create the scatter plot
                ax.scatter(
                    x=data['adjusted_x'],
                    y=data['-log10(p_value)'],
                    s=data['scaled_size'],
                    c=data['color'],
                    alpha=0.6,
                    edgecolors='w',
                    linewidth=0.7
                )

                # Improve the plot aesthetics
                ax.set_xticks([category_map[cat] for cat in categories])
                ax.set_xticklabels([f'{cat} ({len(data[data["source"] == cat])})' for cat in categories], rotation=45, ha='right', fontsize=18)
                ax.set_xlabel('Source', fontsize=18)
                ax.set_ylabel('-log10(p-value)', fontsize=18)
                ax.set_ylim(0, 200)  # Adjust the y-axis limit if needed
                ax.set_title(f'Cluster {cluster_id}', fontsize=20)

        # Remove unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'gprofiler_manhattan_plot_all_clusters_{clustering_method}.png'), dpi=300)
        plt.close()

        # Discover Novel Associations
        novel_associations = {}
        for cluster_id in range(n_clusters):
            cluster_data = significant_results[significant_results['Cluster'] == cluster_id]
            all_data = significant_results[significant_results['Cluster'] != cluster_id]
            unique_terms = set(cluster_data['name']) - set(all_data['name'])
            novel_associations[cluster_id] = cluster_data[cluster_data['name'].isin(unique_terms)]
        
        # Print the top 10 most enriched novel associations for each cluster
        for cluster_id, terms in novel_associations.items():
            if not terms.empty:
                top_novel_terms = terms.nsmallest(top_n, 'p_value')
                print(f"Top {top_n} novel associations in Cluster {cluster_id}:")
                print(top_novel_terms[['name', 'source', 'p_value', 'term_size']])
                top_novel_terms.to_csv(os.path.join(output_dir, f'novel_associations_cluster_{cluster_id}.csv'), index=False)
        
                # Plot smaller heatmaps for the top 10 novel associations
                sns.set(font_scale=0.6)  # Adjust the font scale for smaller plots
        
                heatmap_data = top_novel_terms.pivot_table(index='name', columns='Cluster', values='log_p_value', fill_value=0)
        
                plt.figure(figsize=(9, 4))  # Smaller figure size
                ax = sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt='.2f', cbar_kws={'label': '-log10(p-value)'}, annot_kws={"size": 10, "weight": "bold"}) 
                
                ax.set_aspect(aspect=0.5)  # Adjust the aspect ratio (smaller value makes the heatmap bars wider)
                
                ax.set_xlabel(f'Cluster {cluster_id}', fontsize=12)  # Adjust font size for xlabel
                ax.set_ylabel('Terms', fontsize=12)    # Adjust font size for ylabel
                # ax.set_title(f'Heatmap of Top {top_n} Novel Associations in Cluster {cluster_id}', fontsize=10)  # Adjust font size for title
                ax.set_xticklabels([])
                # Adjust y-axis to handle long term names and separate the labels
                ax.yaxis.label.set_size(12)
                ax.yaxis.set_tick_params(labelsize=12)  # Increase font size and pad to separate the labels
                
                # Increase font size of color bar label
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=10)
                cbar.set_label('-log10(p-value)', fontsize=10)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'novel_associations_heatmap_cluster_{cluster_id}.eps'), dpi=300)
                plt.close()


            else:
                print(f"No enrichment results for clustering method {clustering_method}")







def plot_clustering_scores(protein_embedding, model_result_dir, model_name):
    silhouette_scores = {'KMeans': [], 'Agglomerative': []}
    calinski_scores = {'KMeans': [], 'Agglomerative': []}
    davies_scores = {'KMeans': [], 'Agglomerative': []}
    cluster_range = range(2, 11)
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=2)
        kmeans_labels = kmeans.fit_predict(protein_embedding)
        silhouette_scores['KMeans'].append(silhouette_score(protein_embedding, kmeans_labels))
        calinski_scores['KMeans'].append(calinski_harabasz_score(protein_embedding, kmeans_labels))
        davies_scores['KMeans'].append(davies_bouldin_score(protein_embedding, kmeans_labels))
        
        agglo = AgglomerativeClustering(n_clusters=n_clusters)
        agglo_labels = agglo.fit_predict(protein_embedding)
        silhouette_scores['Agglomerative'].append(silhouette_score(protein_embedding, agglo_labels))
        calinski_scores['Agglomerative'].append(calinski_harabasz_score(protein_embedding, agglo_labels))
        davies_scores['Agglomerative'].append(davies_bouldin_score(protein_embedding, agglo_labels))
        
    # Plot Silhouette Scores
    plt.figure(figsize=(10, 6))
    for method, scores in silhouette_scores.items():
        plt.plot(cluster_range, scores, label=method)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title(f'Silhouette Scores for Different Clustering Methods: {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_result_dir, 'silhouette_scores.png'))
    plt.close()
    
    # Plot Calinski-Harabasz Scores
    plt.figure(figsize=(10, 6))
    for method, scores in calinski_scores.items():
        plt.plot(cluster_range, scores, label=method)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Calinski-Harabasz Score')
    plt.title(f'Calinski-Harabasz Scores for Different Clustering Methods: {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_result_dir, 'calinski_harabasz_scores.png'))
    plt.close()
    
    # Plot Davies-Bouldin Scores
    plt.figure(figsize=(10, 6))
    for method, scores in davies_scores.items():
        plt.plot(cluster_range, scores, label=method)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Bouldin Score')
    plt.title(f'Davies-Bouldin Scores for Different Clustering Methods: {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_result_dir, 'davies_bouldin_scores.png'))
    plt.close()



def evaluate_embeddings(models, n_clusters, protein_indices_in_pyg, pipeline, clustering_algorithm):
    metrics = []
    embeddings_list = []
    model_names = []
    for file_name, embedding in models:
        model_dir = os.path.dirname(file_name)
        model_result_dir = model_dir
        os.makedirs(model_result_dir, exist_ok=True)

        protein_embedding = load_protein_embeddings(embedding, protein_indices_in_pyg)
        # embeddings_list.append(protein_embedding_normalized)
        model_names.append(file_name)
        protein_embedding_normalized = normalize(protein_embedding, norm='l2', axis=1)
        embeddings_list.append(protein_embedding_normalized)
        print(protein_embedding_normalized.shape)
        plot_clustering_scores(protein_embedding_normalized, model_result_dir, file_name)
        kmeans = KMeans(n_clusters=n_clusters, random_state=2)
        kmeans_labels = kmeans.fit_predict(protein_embedding_normalized)

        agglo = AgglomerativeClustering(n_clusters=n_clusters)
        agglo_labels = agglo.fit_predict(protein_embedding_normalized)

        kmedoids = KMedoids(n_clusters=n_clusters)
        kmedoids_labels = kmedoids.fit_predict(protein_embedding_normalized)

        mutual_info_kmeans_agglo = mutual_info_score(kmeans_labels, agglo_labels)
        mutual_info_kmeans_kmedoids = mutual_info_score(kmeans_labels, kmedoids_labels)
        mutual_info_agglo_kmedoids = mutual_info_score(agglo_labels, kmedoids_labels)

        protein_clusters_kmeans = pd.DataFrame({'ProteinID': protein_indices_in_pyg, 'Cluster': kmeans_labels})
        protein_clusters_agglo = pd.DataFrame({'ProteinID': protein_indices_in_pyg, 'Cluster': agglo_labels})
        # protein_clusters_kmedoids = pd.DataFrame({'ProteinID': protein_indices_in_pyg, 'Cluster': kmedoids_labels})

        perform_enrichment_analysis(protein_clusters_kmeans, pipeline, n_clusters, model_result_dir, 'kmeans')
        perform_enrichment_analysis(protein_clusters_agglo, pipeline, n_clusters, model_result_dir, 'agglomerative')
        # perform_enrichment_analysis(protein_clusters_kmedoids, pipeline, n_clusters, model_result_dir, 'kmedoids')

        silhouette_kmeans = silhouette_score(protein_embedding_normalized, kmeans_labels)
        calinski_kmeans = calinski_harabasz_score(protein_embedding_normalized, kmeans_labels)
        davies_kmeans = davies_bouldin_score(protein_embedding_normalized, kmeans_labels)

        silhouette_agglo = silhouette_score(protein_embedding_normalized, agglo_labels)
        calinski_agglo = calinski_harabasz_score(protein_embedding_normalized, agglo_labels)
        davies_agglo = davies_bouldin_score(protein_embedding_normalized, agglo_labels)

        # silhouette_kmedoids = silhouette_score(protein_embedding_normalized, kmedoids_labels)
        # calinski_kmedoids = calinski_harabasz_score(protein_embedding_normalized, kmedoids_labels)
        # davies_kmedoids = davies_bouldin_score(protein_embedding_normalized, kmedoids_labels)
        silhouette_kmedoids = None
        calinski_kmedoids = None
        davies_kmedoids = None
        
        metrics.append({
            'model': file_name,
            'silhouette_score_kmeans': silhouette_kmeans,
            'calinski_harabasz_score_kmeans': calinski_kmeans,
            'davies_bouldin_score_kmeans': davies_kmeans,
            'silhouette_score_agglo': silhouette_agglo,
            'calinski_harabasz_score_agglo': calinski_agglo,
            'davies_bouldin_score_agglo': davies_agglo,
            'silhouette_score_kmedoids': silhouette_kmedoids,
            'calinski_harabasz_score_kmedoids': calinski_kmedoids,
            'davies_bouldin_score_kmedoids': davies_kmedoids,
            'mutual_info_kmeans_agglo': mutual_info_kmeans_agglo,
            'mutual_info_kmeans_kmedoids': mutual_info_kmeans_kmedoids,
            'mutual_info_agglo_kmedoids': mutual_info_agglo_kmedoids
        })

        metrics_file = os.path.join(model_result_dir, 'clustering_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f'Model: {file_name}\n')
            f.write(f'KMeans Silhouette Score: {silhouette_kmeans}\n')
            f.write(f'KMeans Calinski-Harabasz Score: {calinski_kmeans}\n')
            f.write(f'KMeans Davies-Bouldin Score: {davies_kmeans}\n')
            f.write(f'Agglomerative Silhouette Score: {silhouette_agglo}\n')
            f.write(f'Agglomerative Calinski-Harabasz Score: {calinski_agglo}\n')
            f.write(f'Agglomerative Davies-Bouldin Score: {davies_agglo}\n')
            f.write(f'KMedoids Silhouette Score: {silhouette_kmedoids}\n')
            f.write(f'KMedoids Calinski-Harabasz Score: {calinski_kmedoids}\n')
            f.write(f'KMedoids Davies-Bouldin Score: {davies_kmedoids}\n')
            f.write(f'Mutual Information Score (KMeans-Agglomerative): {mutual_info_kmeans_agglo}\n')
            f.write(f'Mutual Information Score (KMeans-KMedoids): {mutual_info_kmeans_kmedoids}\n')
            f.write(f'Mutual Information Score (Agglomerative-KMedoids): {mutual_info_agglo_kmedoids}\n')

        tsne = TSNE(n_components=2, random_state=2)
        tsne_results = tsne.fit_transform(protein_embedding_normalized)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=tsne_results[:, 0], y=tsne_results[:, 1],
            hue=kmeans_labels,
            palette=sns.color_palette("colorblind", n_clusters),
            legend='full',
            alpha=0.6,
            s=100
        )
        
        # plt.title(f'TSNE Visualization for Model: {file_name} (KMeans)')
        font_size=22
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.legend(title='Cluster', title_fontsize=font_size, fontsize=font_size, loc='best')
        
        plt.savefig(os.path.join(model_result_dir, 'tsne_visualization_kmeans.eps'))
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=tsne_results[:, 0], y=tsne_results[:, 1],
            hue=agglo_labels,
            palette=sns.color_palette('hsv', n_clusters),
            legend='full',
            alpha=0.6
        )
        # plt.title(f'TSNE Visualization for Model: {file_name} (Agglomerative)')
        plt.savefig(os.path.join(model_result_dir, 'tsne_visualization_agglo.eps'))
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=tsne_results[:, 0], y=tsne_results[:, 1],
            hue=kmedoids_labels,
            palette=sns.color_palette('hsv', n_clusters),
            legend='full',
            alpha=0.6
        )
        # plt.title(f'TSNE Visualization for Model: {file_name} (KMedoids)')
        plt.savefig(os.path.join(model_result_dir, 'tsne_visualization_kmedoids.png'))
        plt.close()

    overall_metrics_df = pd.DataFrame(metrics)
    overall_metrics_df.to_csv(os.path.join(model_result_dir, 'overall_clustering_metrics.csv'), index=False)

    emd_metrics = []
    frobenius_norm_metrics = []
    n = len(embeddings_list)
    sim_matrix = [None] * n
    for i in range(len(embeddings_list)):
        sim_matrix[i] = calculate_similarity_matrix(embeddings_list[i])
        for j in range(i + 1, len(embeddings_list)):
            sim_matrix[j] = calculate_similarity_matrix(embeddings_list[j])
            emd_value = calculate_emd(sim_matrix[i], sim_matrix[j])
            frobenius_norm_value = norm(sim_matrix[i] - sim_matrix[j], 'fro')
            emd_metrics.append({
                'model1': model_names[i],
                'model2': model_names[j],
                'emd': emd_value
            })
            frobenius_norm_metrics.append({
                'model1': model_names[i],
                'model2': model_names[j],
                'frobenius_norm': frobenius_norm_value
            })
            
    emd_metrics_df = pd.DataFrame(emd_metrics)
    frobenius_norm_metrics_df = pd.DataFrame(frobenius_norm_metrics)

    emd_metrics_df.to_csv(os.path.join(model_result_dir, 'emd_metrics.csv'), index=False)
    frobenius_norm_metrics_df.to_csv(os.path.join(model_result_dir, 'frobenius_norm_metrics.csv'), index=False)



    return metrics, emd_metrics, frobenius_norm_metrics







def calculate_emd(hist1, hist2):
    """Calculates the Earth Mover's Distance (EMD) between two histograms."""
    # Normalize histograms
    # hist1 = hist1.astype(np.float64)
    # hist2 = hist2.astype(np.float64)
    
    # hist1 /= hist1.sum()
    # hist2 /= hist2.sum()
    
    # Calculate EMD using Wasserstein distance
    emd_value = wasserstein_distance(hist1.flatten(), hist2.flatten())
    return emd_value



def load_protein_embeddings(embedding, protein_indices):

    protein_embeddings = embedding[protein_indices]
    return protein_embeddings
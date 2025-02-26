import json
import torch.nn as nn
import os
from matplotlib import pyplot as plt
import pandas as pd
import torch
import itertools
import dgl
import numpy as np
import scipy.sparse as sp
from dgl.dataloading import GraphDataLoader
from .models import GATModel, GCNModel, GINModel, GraphSAGE, MLPPredictor, FocalLoss
from .utils import (choose_model, plot_training_validation_metrics, plot_roc_pr_curves, plot_roc_curves, plot_pr_curves, compute_hits_k, compute_auc, compute_f1, compute_focalloss,
                    compute_accuracy, compute_precision, compute_recall, compute_map,
                    compute_focalloss_with_symmetrical_confidence, compute_auc_with_symmetrical_confidence,
                    compute_f1_with_symmetrical_confidence, compute_accuracy_with_symmetrical_confidence,
                    compute_precision_with_symmetrical_confidence, compute_recall_with_symmetrical_confidence,
                    compute_map_with_symmetrical_confidence)
from scipy.stats import sem
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import networkx as nx
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict, deque


##-------------------------------------------------------------------------------------


############################################################################################################

# Random Negative Sampling - Both Visible Node
def random_negative_sampling_both_visible(G_dgl, num_samples):
    # Ensure all_nodes is a CPU tensor before converting to a NumPy array
    all_nodes = G_dgl.nodes().cpu().numpy()
    
    neg_u, neg_v = [], []
    
    while len(neg_u) < num_samples:
        u_node = np.random.choice(all_nodes)
        v_node = np.random.choice(all_nodes)

        if not G_dgl.has_edges_between(u_node, v_node):  # Ensure they are not connected
            neg_u.append(u_node)
            neg_v.append(v_node)

    return np.array(neg_u), np.array(neg_v)

# Random Negative Sampling - One Visible Node
def random_negative_sampling_one_visible(G_dgl, num_samples):
    all_nodes = G_dgl.nodes().cpu().numpy()  # Move nodes to CPU and convert to numpy
    neg_u, neg_v = [], []

    while len(neg_u) < num_samples:
        u_node = np.random.choice(all_nodes)
        v_node = np.random.choice(all_nodes)

        if not G_dgl.has_edges_between(u_node, v_node):  # Ensure they are not connected
            neg_u.append(u_node)
            neg_v.append(v_node)

    return np.array(neg_u), np.array(neg_v)

# Random Negative Sampling - Neither Visible
def random_negative_sampling_neither_visible(G_dgl, num_samples):
    all_nodes = G_dgl.nodes().cpu().numpy()  # Move nodes to CPU and convert to numpy
    neg_u, neg_v = [], []

    while len(neg_u) < num_samples:
        u_node = np.random.choice(all_nodes)
        v_node = np.random.choice(all_nodes)

        if not G_dgl.has_edges_between(u_node, v_node):  # Ensure neither node is visible
            neg_u.append(u_node)
            neg_v.append(v_node)

    return np.array(neg_u), np.array(neg_v)

def dfs_negative_sampling_both_visible(G_dgl, num_samples):
    ##all_nodes = G_dgl.nodes()
    all_nodes = G_dgl.nodes().cpu().numpy()  # Move to CPU and convert to NumPy array
    visited = set()
    neg_u, neg_v = [], []
    
    def dfs(node):
        if len(neg_u) >= num_samples:
            return
        visited.add(node)
        neighbors = set(G_dgl.successors(node))
        for neighbor in neighbors:
            if len(neg_u) >= num_samples:
                return
            for candidate in all_nodes:
                if candidate not in neighbors and candidate != node and candidate not in visited:
                    # Both nodes are visible (in the DFS traversal path)
                    neg_u.append(node)
                    neg_v.append(candidate)
    
    for node in all_nodes:
        if len(neg_u) >= num_samples:
            break
        if node not in visited:
            dfs(node)
    
    return np.array(neg_u), np.array(neg_v)

# DFS Negative Sampling - One Visible Node
def dfs_negative_sampling_one_visible(G_dgl, num_samples):
    all_nodes = G_dgl.nodes().cpu().numpy()
    visited = set()
    neg_u, neg_v = [], []

    def dfs(node):
        if len(neg_u) >= num_samples:
            return
        visited.add(node)
        neighbors = set(G_dgl.successors(node).cpu().numpy())
        for neighbor in all_nodes:
            if neighbor not in neighbors and neighbor not in visited:
                neg_u.append(node)
                neg_v.append(neighbor)
                if len(neg_u) >= num_samples:
                    return
        for neighbor in neighbors:
            if neighbor not in visited:
                dfs(neighbor)

    for node in all_nodes:
        if len(neg_u) >= num_samples:
            break
        if node not in visited:
            dfs(node)

    return np.array(neg_u), np.array(neg_v)

# DFS Negative Sampling - Neither Visible
def dfs_negative_sampling_neither_visible(G_dgl, num_samples):
    all_nodes = G_dgl.nodes().cpu().numpy()
    neg_u, neg_v = [], []
    visited_pairs = set()

    def dfs_component(node, visited):
        stack = [node]
        component = set()
        while stack:
            current_node = stack.pop()
            if current_node not in visited:
                visited.add(current_node)
                component.add(current_node)
                neighbors = G_dgl.successors(current_node).cpu().numpy()
                stack.extend(neighbors)
        return component

    components = []
    unvisited_nodes = set(all_nodes)

    while unvisited_nodes:
        start_node = next(iter(unvisited_nodes))
        component = dfs_component(start_node, visited=set())
        components.append(component)
        unvisited_nodes -= component

    while len(neg_u) < num_samples:
        component_a, component_b = np.random.choice(components, size=2, replace=False)
        u_node = np.random.choice(list(component_a))
        v_node = np.random.choice(list(component_b))

        if (u_node, v_node) not in visited_pairs and not G_dgl.has_edges_between(u_node, v_node):
            neg_u.append(u_node)
            neg_v.append(v_node)
            visited_pairs.add((u_node, v_node))

    return np.array(neg_u), np.array(neg_v)

# BFS Negative Sampling - Neither Visible
def bfs_negative_sampling_neither_visible(G_dgl, num_samples):
    all_nodes = G_dgl.nodes().cpu().numpy()
    neg_u, neg_v = [], []
    visited_pairs = set()

    def bfs_component(start_node):
        visited = set()
        queue = deque([start_node])
        component = set()
        while queue:
            current_node = queue.popleft()
            if current_node not in visited:
                visited.add(current_node)
                component.add(current_node)
                neighbors = G_dgl.successors(current_node).cpu().numpy()
                queue.extend(neighbors)
        return component

    components = []
    unvisited_nodes = set(all_nodes)

    while unvisited_nodes:
        start_node = next(iter(unvisited_nodes))
        component = bfs_component(start_node)
        components.append(component)
        unvisited_nodes -= component

    while len(neg_u) < num_samples:
        component_a, component_b = np.random.choice(components, size=2, replace=False)
        u_node = np.random.choice(list(component_a))
        v_node = np.random.choice(list(component_b))

        if (u_node, v_node) not in visited_pairs and not G_dgl.has_edges_between(u_node, v_node):
            neg_u.append(u_node)
            neg_v.append(v_node)
            visited_pairs.add((u_node, v_node))

    return np.array(neg_u), np.array(neg_v)

# BFS Negative Sampling - Both Visible
def bfs_negative_sampling_both_visible(G_dgl, num_samples):
    all_nodes = G_dgl.nodes().cpu().numpy()
    visited = set()
    neg_u, neg_v = [], []

    def bfs(start_node):
        queue = deque([start_node])
        while queue and len(neg_u) < num_samples:
            node = queue.popleft()
            visited.add(node)
            neighbors = set(G_dgl.successors(node).cpu().numpy())

            for neighbor in neighbors:
                if len(neg_u) >= num_samples:
                    break
                for candidate in all_nodes:
                    if candidate not in neighbors and candidate != node and candidate not in visited:
                        neg_u.append(node)
                        neg_v.append(candidate)

            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)

    for node in all_nodes:
        if len(neg_u) >= num_samples:
            break
        if node not in visited:
            bfs(node)

    return np.array(neg_u), np.array(neg_v)

def bfs_negative_sampling_one_visible(G_dgl, num_samples):
    """
    BFS-based negative sampling where at least one node in a sampled pair is visible.

    Args:
        G_dgl (dgl.DGLGraph): Input DGL graph.
        num_samples (int): Number of negative samples to generate.

    Returns:
        torch.Tensor, torch.Tensor: Negative source nodes, Negative destination nodes.
    """
    if not G_dgl.is_homogeneous:
        raise ValueError("This function supports homogeneous graphs only.")
    
    # Ensure the graph is on GPU if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G_dgl = G_dgl.to(device)

    all_nodes = G_dgl.nodes()
    neg_u, neg_v = [], []
    visited = set()

    def bfs(start_node):
        """Perform BFS starting from a given node."""
        queue = deque([start_node])
        while queue and len(neg_u) < num_samples:
            node = queue.popleft()
            visited.add(node.item())

            # Get the neighbors of the current node
            neighbors = set(G_dgl.successors(node).cpu().numpy())
            all_nodes_list = all_nodes.cpu().numpy()

            # Sample a negative destination node for the current source
            for candidate in all_nodes_list:
                if candidate not in neighbors and candidate != node.item():
                    neg_u.append(node.item())
                    neg_v.append(candidate)

                if len(neg_u) >= num_samples:
                    break

            # Add unvisited neighbors to the BFS queue
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(torch.tensor(neighbor, device=device))

    for node in all_nodes:
        if len(neg_u) >= num_samples:
            break
        if node not in visited:
            bfs(node)

    return np.array(neg_u), np.array(neg_v)

###############################################################################################################################
##def train_and_evaluate_FocalLoss(args, G_dgl, node_features): ##, interaction_type):
def train_and_evaluate(args, G_dgl, node_features): ##, interaction_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ##print(f"Using device: {device}")

    u, v = G_dgl.edges()
    eids = np.arange(G_dgl.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.2)
    val_size = int(len(eids) * 0.2)
    train_size = G_dgl.number_of_edges() - test_size - val_size

    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    val_pos_u, val_pos_v = u[eids[test_size:test_size + val_size]], v[eids[test_size:test_size + val_size]]
    train_pos_u, train_pos_v = u[eids[test_size + val_size:]], v[eids[test_size + val_size:]]

    ##adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    '''def to_numpy(tensor):
            return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()

    ##adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj = sp.coo_matrix(
        (np.ones(len(u)), (to_numpy(u), to_numpy(v))),
        shape=(G_dgl.number_of_nodes(), G_dgl.number_of_nodes())
    )
        
    ##adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(G_dgl.number_of_nodes(), G_dgl.number_of_nodes()))
    adj_neg = 1 - adj.todense() - np.eye(G_dgl.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)'''

    sampling_methods = {
        "dfs_one_visible": dfs_negative_sampling_one_visible,
        "dfs_neither_visible": dfs_negative_sampling_neither_visible,
        "dfs_both_visible": dfs_negative_sampling_both_visible,
        "bfs_one_visible": bfs_negative_sampling_one_visible,
        "bfs_neither_visible": bfs_negative_sampling_neither_visible,
        "bfs_both_visible": bfs_negative_sampling_both_visible,
        "random_one_visible": random_negative_sampling_one_visible,
        "random_neither_visible": random_negative_sampling_neither_visible,
        "random_both_visible": random_negative_sampling_both_visible
    }

    if args.sampling_method not in sampling_methods:
        raise ValueError(f"Unknown sampling method: {args.sampling_method}")
    
    print(f"Using sampling method: {args.sampling_method}")
    neg_u, neg_v = sampling_methods[args.sampling_method](G_dgl, G_dgl.number_of_edges())


    '''def to_numpy(tensor):
        return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()

    ##adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj = sp.coo_matrix(
        (np.ones(len(u)), (to_numpy(u), to_numpy(v))),
        shape=(G_dgl.number_of_nodes(), G_dgl.number_of_nodes())
    )
    
    ##adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(G_dgl.number_of_nodes(), G_dgl.number_of_nodes()))
    adj_neg = 1 - adj.todense() - np.eye(G_dgl.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)'''


    # Dropout values to validate
    dropout_values = np.arange(0.5, 0.6, 0.1)

    for dropout in dropout_values:
        print(f"Validating with Dropout: {dropout}")
            
        neg_eids = np.random.choice(len(neg_u), G_dgl.number_of_edges())
        test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
        val_neg_u, val_neg_v = neg_u[neg_eids[test_size:test_size + val_size]], neg_v[neg_eids[test_size:test_size + val_size]]
        train_neg_u, train_neg_v = neg_u[neg_eids[test_size + val_size:]], neg_v[neg_eids[test_size + val_size:]]

        print(f"  Sample Counts:")
        print(f"  Train Positive: {len(train_pos_u)}, Train Negative: {len(train_neg_u)}")
        print(f"  Validation Positive: {len(val_pos_u)}, Validation Negative: {len(val_neg_u)}")
        print(f"  Test Positive: {len(test_pos_u)}, Test Negative: {len(test_neg_u)}")

        train_g = dgl.remove_edges(G_dgl, eids[:test_size + val_size])

        kf = KFold(n_splits=5, shuffle=True, random_state=66)
        output_path = './link_prediction_gat/results/STRING'
        os.makedirs(output_path, exist_ok=True)
        
        fold_results = []
        all_fold_results = pd.DataFrame()
        
        train_accuracies = []
        val_accuracies = []
        train_losses = []
        val_losses = []
        # Initialize an empty list to store results for all folds
        lowest_loss_results = []


        for fold, (train_idx, test_idx) in enumerate(kf.split(eids)):
            print(f'Fold {fold + 1}')
                
            def create_graph(u, v, num_nodes):
                assert len(u) == len(v), "Source and destination nodes must have the same length"
                return dgl.graph((u, v), num_nodes=num_nodes).to(device)  # Move graph to device


            train_pos_g = create_graph(train_pos_u, train_pos_v, G_dgl.number_of_nodes())
            train_neg_g = create_graph(train_neg_u, train_neg_v, G_dgl.number_of_nodes())
            val_pos_g = create_graph(val_pos_u, val_pos_v, G_dgl.number_of_nodes())
            val_neg_g = create_graph(val_neg_u, val_neg_v, G_dgl.number_of_nodes())
            test_pos_g = create_graph(test_pos_u, test_pos_v, G_dgl.number_of_nodes())
            test_neg_g = create_graph(test_neg_u, test_neg_v, G_dgl.number_of_nodes())


            '''model = GINModel(
                ##in_feats=16,
                in_feats=node_features.shape[1],  # Input feature size
                hidden_feats=args.hidden_feats, 
                out_feats=args.out_feats,         # Output feature size
                num_layers=args.num_layers,       # Number of GIN layers
                feat_drop=args.feat_drop,
                activation=nn.LeakyReLU(0.1),  # Using LeakyReLU
                do_train=True
            ).to(device)'''  
            model = choose_model(
                model_type=args.model_type,  # Ensure args.model_type is set correctly
                in_feats=node_features.size(1),
                hidden_feats=args.hidden_feats,
                out_feats=args.out_feats,
                num_layers=args.num_layers
            ).to(device)
            '''model = GraphSAGE(
                in_feats=node_features.size(1), 
                hidden_feats=args.hidden_feats, 
                out_feats=args.out_feats, 
                num_layers=args.num_layers
            ).to(device)'''
            
            '''model = GATModel(
                node_features.shape[1],
                out_feats=args.out_feats,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                feat_drop=0,
                attn_drop=0,
                ##dropout=dropout,
                do_train=True
            ).to(device)'''
            
            '''model = GCNModel(
                node_features.shape[1],
                out_feats=args.out_feats,
                num_layers=args.num_layers,
                do_train=True
            ).to(device)'''
            

            fold_train_accuracies = []
            fold_val_accuracies = []
            fold_train_losses = []
            fold_val_losses = []


            pred = MLPPredictor(args.input_size, args.hidden_size).to(device)  # Move predictor to device
            ##criterion = nn.BCEWithLogitsLoss(reduction='none')
            criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean').to(device)  # Move loss to device
            optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)
            scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)

            best_train_loss = float('inf')  # Track the best training loss
            best_val_loss = float('inf')  # Track the best validation loss
            best_epoch = -1  # Track the epoch with the best validation loss

            for e in tqdm(range(args.epochs)):
                model.train()
                h = model(train_g, train_g.ndata['feat'].to(device))  # Move features to device
                pos_score = pred(train_pos_g, h)
                neg_score = pred(train_neg_g, h)

                pos_labels = torch.ones_like(pos_score).to(device)
                neg_labels = torch.zeros_like(neg_score).to(device)

                all_scores = torch.cat([pos_score, neg_score])
                all_labels = torch.cat([pos_labels, neg_labels])

                loss = criterion(all_scores, all_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                fold_train_losses.append(loss.item())
                
                if loss.item() < best_train_loss:
                    best_train_loss = loss.item()
                    best_epoch_train = e
                    
                with torch.no_grad():
                    model.eval()
                    h_val = model(G_dgl, G_dgl.ndata['feat'])
                    val_pos_score = pred(val_pos_g, h_val)
                    val_neg_score = pred(val_neg_g, h_val)

                    val_all_scores = torch.cat([val_pos_score, val_neg_score])
                    val_all_labels = torch.cat([torch.ones_like(val_pos_score), torch.zeros_like(val_neg_score)])
                    
                    val_loss = criterion(val_all_scores, val_all_labels)
                    fold_val_losses.append(val_loss.item())
                    
                    val_acc = ((val_pos_score > 0.5).sum().item() + (val_neg_score <= 0.5).sum().item()) / (len(val_pos_score) + len(val_neg_score))
                    fold_val_accuracies.append(val_acc)

                    train_acc = ((pos_score > 0.5).sum().item() + (neg_score <= 0.5).sum().item()) / (len(pos_score) + len(neg_score))
                    fold_train_accuracies.append(train_acc)
                
                # Check if this epoch's validation loss is the lowest
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_epoch = e

                if e % 5 == 0:
                    print(f'Epoch {e} | Loss: {loss.item()} | Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f}')

            train_accuracies.append(fold_train_accuracies)
            val_accuracies.append(fold_val_accuracies)
            train_losses.append(fold_train_losses)
            val_losses.append(fold_val_losses)


            with torch.no_grad():
                model.eval()

                h_val = model(G_dgl, G_dgl.ndata['feat'])
                val_pos_score = pred(val_pos_g, h_val)
                val_neg_score = pred(val_neg_g, h_val)
                
                val_pos_score = torch.sigmoid(val_pos_score)
                val_neg_score = torch.sigmoid(val_neg_score)
                val_auc, val_auc_err = compute_auc_with_symmetrical_confidence(val_pos_score, val_neg_score)
                val_f1, val_f1_err = compute_f1_with_symmetrical_confidence(val_pos_score, val_neg_score)
                val_focal_loss, val_focal_loss_err = compute_focalloss_with_symmetrical_confidence(val_pos_score, val_neg_score)
                val_precision, val_precision_err = compute_precision_with_symmetrical_confidence(val_pos_score, val_neg_score)
                val_recall, val_recall_err = compute_recall_with_symmetrical_confidence(val_pos_score, val_neg_score)
                val_hits_k = compute_hits_k(val_pos_score, val_neg_score, k=10)
                val_map, val_map_err = compute_map_with_symmetrical_confidence(val_pos_score, val_neg_score)
                val_accuracy, val_accuracy_err = compute_accuracy_with_symmetrical_confidence(val_pos_score, val_neg_score)

                val_metrics = (
                    f'Val AUC: {val_auc:.4f} ± {val_auc_err:.4f} | Val F1: {val_f1:.4f} ± {val_f1_err:.4f} | '
                    f'Val FocalLoss: {val_focal_loss:.4f} ± {val_focal_loss_err:.4f} | Val Accuracy: {val_accuracy:.4f} ± {val_accuracy_err:.4f} | '
                    f'Val Precision: {val_precision:.4f} ± {val_precision_err:.4f} | Val Recall: {val_recall:.4f} ± {val_recall_err:.4f} | '
                    f'Val Hits@10: {val_hits_k:.4f} | Val MAP: {val_map:.4f} ± {val_map_err:.4f}'
                )
                print(val_metrics)

                # Save the val metrics to a .txt file
                output_path_val_metrics = f'SAGE_STRING_ptmod_FocalLoss_{args.sampling_method}_val_drop{dropout}_head{args.num_heads}_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.txt'
            
                with open(os.path.join(output_path, output_path_val_metrics), 'a') as f:
                    f.write(f'Fold {fold + 1}:\n')
                    f.write(val_metrics + '\n\n')

                h_test = model(G_dgl, G_dgl.ndata['feat'])
                test_pos_score = pred(test_pos_g, h_test)
                test_neg_score = pred(test_neg_g, h_test)
                
                # Apply sigmoid to test scores
                test_pos_score = torch.sigmoid(test_pos_score)
                test_neg_score = torch.sigmoid(test_neg_score)
                test_auc, test_auc_err = compute_auc_with_symmetrical_confidence(test_pos_score, test_neg_score)
                test_f1, test_f1_err = compute_f1_with_symmetrical_confidence(test_pos_score, test_neg_score)
                test_focal_loss, test_focal_loss_err = compute_focalloss_with_symmetrical_confidence(test_pos_score, test_neg_score)
                test_precision, test_precision_err = compute_precision_with_symmetrical_confidence(test_pos_score, test_neg_score)
                test_recall, test_recall_err = compute_recall_with_symmetrical_confidence(test_pos_score, test_neg_score)
                test_hits_k = compute_hits_k(test_pos_score, test_neg_score, k=10)
                test_map, test_map_err = compute_map_with_symmetrical_confidence(test_pos_score, test_neg_score)
                test_accuracy, test_accuracy_err = compute_accuracy_with_symmetrical_confidence(test_pos_score, test_neg_score)

                test_metrics = (
                    f'Test AUC: {test_auc:.4f} ± {test_auc_err:.4f} | Test F1: {test_f1:.4f} ± {test_f1_err:.4f} | '
                    f'Test FocalLoss: {test_focal_loss:.4f} ± {test_focal_loss_err:.4f} | Test Accuracy: {test_accuracy:.4f} ± {test_accuracy_err:.4f} | '
                    f'Test Precision: {test_precision:.4f} ± {test_precision_err:.4f} | Test Recall: {test_recall:.4f} ± {test_recall_err:.4f} | '
                    f'Test Hits@10: {test_hits_k:.4f} | Test MAP: {test_map:.4f} ± {test_map_err:.4f}'
                )
                print(test_metrics)

                # Save the test metrics to a .txt file
                output_path_test_metrics = f'SAGE_STRING_ptmod_FocalLoss_{args.sampling_method}_test_drop{dropout}_head{args.num_heads}_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.txt'

                with open(os.path.join(output_path, output_path_test_metrics), 'a') as f:
                    f.write(f'Fold {fold + 1}:\n')
                    f.write(test_metrics + '\n\n')

                true_labels = torch.cat([torch.ones(len(test_pos_score)), torch.zeros(len(test_neg_score))])
                predicted_scores = torch.cat([test_pos_score, test_neg_score]).cpu().numpy()

                fold_results.append((true_labels.cpu().numpy(), predicted_scores))

                fold_result_data = pd.DataFrame({
                    'Fold': [fold + 1],
                    'Test AUC': [test_auc.cpu().item() if torch.is_tensor(test_auc) else test_auc],
                    'Test AUC Err': [test_auc_err.cpu().item() if torch.is_tensor(test_auc_err) else test_auc_err],
                    'Test F1 Score': [test_f1.cpu().item() if torch.is_tensor(test_f1) else test_f1],
                    'Test F1 Score Err': [test_f1_err.cpu().item() if torch.is_tensor(test_f1_err) else test_f1_err],
                    'Test Precision': [test_precision.cpu().item() if torch.is_tensor(test_precision) else test_precision],
                    'Test Precision Err': [test_precision_err.cpu().item() if torch.is_tensor(test_precision_err) else test_precision_err],
                    'Test Recall': [test_recall.cpu().item() if torch.is_tensor(test_recall) else test_recall],
                    'Test Recall Err': [test_recall_err.cpu().item() if torch.is_tensor(test_recall_err) else test_recall_err],
                    'Test Hit': [test_hits_k.cpu().item() if torch.is_tensor(test_hits_k) else test_hits_k],
                    'Test mAP': [test_map.cpu().item() if torch.is_tensor(test_map) else test_map],
                    'Test mAP Err': [test_map_err.cpu().item() if torch.is_tensor(test_map_err) else test_map_err],
                    'Test FocalLoss': [test_focal_loss.cpu().item() if torch.is_tensor(test_focal_loss) else test_focal_loss],
                    'Test FocalLoss Err': [test_focal_loss_err.cpu().item() if torch.is_tensor(test_focal_loss_err) else test_focal_loss_err],
                    'Test Accuracy': [test_accuracy.cpu().item() if torch.is_tensor(test_accuracy) else test_accuracy],
                    'Test Accuracy Err': [test_accuracy_err.cpu().item() if torch.is_tensor(test_accuracy_err) else test_accuracy_err],
                })


                all_fold_results = pd.concat([all_fold_results, fold_result_data], ignore_index=True)

    
                if e % 5 == 0:
                    print(f'Epoch {e} | Loss: {loss.item()} | Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f}')


            '''train_accuracies.append(fold_train_accuracies)
            val_accuracies.append(fold_val_accuracies)
            train_losses.append(fold_train_losses)
            val_losses.append(fold_val_losses)'''

            # Store the results for this fold
            lowest_loss_results.append({
                'fold': fold + 1,
                'best_epoch_train': best_epoch_train,
                'lowest_train_loss': best_train_loss,
                'best_epoch': best_epoch,
                'lowest_val_loss': best_val_loss

            })

            all_fold_results_filename_loss = f'SAGE_STRING_ptmod_FocalLoss_{args.sampling_method}_lowest_loss_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.json'
            results_file = os.path.join(output_path, all_fold_results_filename_loss)


            with open(results_file, 'w') as f:
                json.dump(lowest_loss_results, f, indent=4)

            print(f"Results saved to {results_file}")   
            
            all_fold_results_filename = f'SAGE_STRING_ptmod_FocalLoss_{args.sampling_method}_all_folds_results_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.csv'
            all_fold_results.to_csv(os.path.join(output_path, all_fold_results_filename), index=False)

            # Average metrics over folds
            avg_train_accuracies = np.mean(train_accuracies, axis=0)
            avg_val_accuracies = np.mean(val_accuracies, axis=0)
            avg_train_losses = np.mean(train_losses, axis=0)
            avg_val_losses = np.mean(val_losses, axis=0)

            # Plot and save ROC and PR curves for all folds
            ## plot_roc_pr_curves(fold_results, os.path.join(output_path, f'GraphSAGE_STRING_950_{args.sampling_method}_roc_pr_curves_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.png')) 
            ## True, (0.1, 0.2, 0.9, 1), (0.9, 1.0, 0.9, 1))
            ## (0.05, 0.15, 0.9, 1), (0.85, 0.95, 0.9, 1))
            ## plot_roc_pr_curves(fold_results, os.path.join(output_path, f'GraphSAGE_STRING_950_{args.sampling_method}_roc_pr_curves_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.png'), True, (0.05, 0.15, 0.9, 1), (0.85, 0.95, 0.9, 1))
            ## (0, 0.1, 0.9, 1), (0.9, 1, 0.9, 1))
            ## GIN (0.05, 0.15, 0.9, 1), (0.85, 0.95, 0.9, 1))
            ## plot_roc_pr_curves(fold_results, os.path.join(output_path, f'GraphSAGE_STRING_950_{args.sampling_method}_roc_pr_curves_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.png'), True, (0.1, 0.2, 0.7, 0.8), (0.8, 0.9, 0.7, 0.8))
            ## plot_roc_pr_curves(fold_results, os.path.join(output_path, f'GraphSAGE_STRING_950_{args.sampling_method}_roc_pr_curves_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.png'), True, (0, 0.1, 0.9, 1), (0.9, 1, 0.9, 1))
            ## (0.02, 0.12, 0.88, 0.98), (0.9, 1, 0.9, 1))
            plot_roc_pr_curves(fold_results, os.path.join(output_path, f'SAGE_STRING_ptmod_FocalLoss_{args.sampling_method}_roc_pr_curves_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.png'), True, (0.09, 0.19, 0.88, 0.98), (0.9, 1, 0.9, 1)) ##(0.1, 0.2, 0.88, 0.98), (0.9, 1, 0.9, 1))

            # Plot and save training and validation metrics
            output_path_train = os.path.join(output_path, f'SAGE_STRING_ptmod_FocalLoss_{args.sampling_method}_f1_curve_head{args.num_heads}_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.png')

            plot_training_validation_metrics(
                train_accuracies, avg_train_accuracies,
                val_accuracies, avg_val_accuracies,
                train_losses, avg_train_losses,
                val_losses, avg_val_losses,
                output_path_train, args
            )

            # Process top predictions

            print('test_size==========\n', len(test_pos_u))

def train_and_evaluate_BCEWithLogitsLoss(args, G_dgl, node_features): ##, interaction_type):
##def train_and_evaluate(args, G_dgl, node_features): ##, interaction_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ##print(f"Using device: {device}")

    u, v = G_dgl.edges()
    eids = np.arange(G_dgl.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.2)
    val_size = int(len(eids) * 0.2)
    train_size = G_dgl.number_of_edges() - test_size - val_size

    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    val_pos_u, val_pos_v = u[eids[test_size:test_size + val_size]], v[eids[test_size:test_size + val_size]]
    train_pos_u, train_pos_v = u[eids[test_size + val_size:]], v[eids[test_size + val_size:]]

    sampling_methods = {
        "dfs_one_visible": dfs_negative_sampling_one_visible,
        "dfs_neither_visible": dfs_negative_sampling_neither_visible,
        "dfs_both_visible": dfs_negative_sampling_both_visible,
        "bfs_one_visible": bfs_negative_sampling_one_visible,
        "bfs_neither_visible": bfs_negative_sampling_neither_visible,
        "bfs_both_visible": bfs_negative_sampling_both_visible,
        "random_one_visible": random_negative_sampling_one_visible,
        "random_neither_visible": random_negative_sampling_neither_visible,
        "random_both_visible": random_negative_sampling_both_visible
    }

    if args.sampling_method not in sampling_methods:
        raise ValueError(f"Unknown sampling method: {args.sampling_method}")
    
    print(f"Using sampling method: {args.sampling_method}")
    neg_u, neg_v = sampling_methods[args.sampling_method](G_dgl, G_dgl.number_of_edges())

    # Dropout values to validate
    dropout_values = np.arange(0.5, 0.6, 0.1)

    for dropout in dropout_values:
        print(f"Validating with Dropout: {dropout}")
            
        neg_eids = np.random.choice(len(neg_u), G_dgl.number_of_edges())
        test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
        val_neg_u, val_neg_v = neg_u[neg_eids[test_size:test_size + val_size]], neg_v[neg_eids[test_size:test_size + val_size]]
        train_neg_u, train_neg_v = neg_u[neg_eids[test_size + val_size:]], neg_v[neg_eids[test_size + val_size:]]

        print(f"  Sample Counts:")
        print(f"  Train Positive: {len(train_pos_u)}, Train Negative: {len(train_neg_u)}")
        print(f"  Validation Positive: {len(val_pos_u)}, Validation Negative: {len(val_neg_u)}")
        print(f"  Test Positive: {len(test_pos_u)}, Test Negative: {len(test_neg_u)}")

        train_g = dgl.remove_edges(G_dgl, eids[:test_size + val_size])

        kf = KFold(n_splits=5, shuffle=True, random_state=66)
        output_path = './link_prediction_gat/results/SHS27k'
        os.makedirs(output_path, exist_ok=True)
        
        fold_results = []
        all_fold_results = pd.DataFrame()
        
        train_accuracies = []
        val_accuracies = []
        train_losses = []
        val_losses = []


        for fold, (train_idx, test_idx) in enumerate(kf.split(eids)):
            print(f'Fold {fold + 1}')
                
            def create_graph(u, v, num_nodes):
                assert len(u) == len(v), "Source and destination nodes must have the same length"
                return dgl.graph((u, v), num_nodes=num_nodes).to(device)  # Move graph to device


            train_pos_g = create_graph(train_pos_u, train_pos_v, G_dgl.number_of_nodes())
            train_neg_g = create_graph(train_neg_u, train_neg_v, G_dgl.number_of_nodes())
            val_pos_g = create_graph(val_pos_u, val_pos_v, G_dgl.number_of_nodes())
            val_neg_g = create_graph(val_neg_u, val_neg_v, G_dgl.number_of_nodes())
            test_pos_g = create_graph(test_pos_u, test_pos_v, G_dgl.number_of_nodes())
            test_neg_g = create_graph(test_neg_u, test_neg_v, G_dgl.number_of_nodes())


            model = GINModel(
                ##in_feats=16,
                in_feats=node_features.shape[1],  # Input feature size
                hidden_feats=args.hidden_feats, 
                out_feats=args.out_feats,         # Output feature size
                num_layers=args.num_layers,       # Number of GIN layers
                feat_drop=args.feat_drop,
                activation=nn.LeakyReLU(0.1),  # Using LeakyReLU
                do_train=True
            ).to(device)  # Move model to device

            '''model = GraphSAGE(
                in_feats=node_features.size(1), 
                hidden_feats=args.hidden_feats, 
                out_feats=args.out_feats, 
                num_layers=args.num_layers
            ).to(device)'''
            
            '''model = GATModel(
                node_features.shape[1],
                out_feats=args.out_feats,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                feat_drop=0,
                attn_drop=0,
                ##dropout=dropout,
                do_train=True
            ).to(device)'''
            
            '''model = GCNModel(
                node_features.shape[1],
                out_feats=args.out_feats,
                num_layers=args.num_layers,
                do_train=True
            ).to(device)'''

 
            fold_train_accuracies = []
            fold_val_accuracies = []
            fold_train_losses = []
            fold_val_losses = []


            pred = MLPPredictor(args.input_size, args.hidden_size).to(device)  # Move predictor to device
            criterion = nn.BCEWithLogitsLoss(reduction='none')
            ##criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean').to(device)  # Move loss to device
            optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)
            scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)

            for e in tqdm(range(args.epochs)):
                model.train()
                h = model(train_g, train_g.ndata['feat'].to(device))  # Move features to device
                pos_score = pred(train_pos_g, h)  # Logits
                neg_score = pred(train_neg_g, h)  # Logits

                pos_labels = torch.ones_like(pos_score).to(device)
                neg_labels = torch.zeros_like(neg_score).to(device)

                all_scores = torch.cat([pos_score, neg_score])  # Raw logits
                all_labels = torch.cat([pos_labels, neg_labels])

                loss = criterion(all_scores, all_labels)  # BCEWithLogitsLoss computes sigmoid internally
                loss = loss.mean()  # Aggregate to a scalar for backward pass

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                fold_train_losses.append(loss.item())

                with torch.no_grad():
                    model.eval()
                    h_val = model(G_dgl, G_dgl.ndata['feat'])
                    val_pos_score = pred(val_pos_g, h_val)
                    val_neg_score = pred(val_neg_g, h_val)
                    
                    val_all_scores = torch.cat([val_pos_score, val_neg_score])
                    val_all_labels = torch.cat([torch.ones_like(val_pos_score), torch.zeros_like(val_neg_score)])
                    
                    ##val_loss = criterion(val_all_scores, val_all_labels)
                    val_loss = criterion(val_all_scores, val_all_labels).mean()
                    fold_val_losses.append(val_loss.item())
                    
                    val_acc = ((val_pos_score > 0.5).sum().item() + (val_neg_score <= 0.5).sum().item()) / (len(val_pos_score) + len(val_neg_score))
                    fold_val_accuracies.append(val_acc)

                    train_acc = ((pos_score > 0.5).sum().item() + (neg_score <= 0.5).sum().item()) / (len(pos_score) + len(neg_score))
                    fold_train_accuracies.append(train_acc)
                
                if e % 5 == 0:
                    print(f'Epoch {e} | Loss: {loss.item()} | Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f}')
            
            train_accuracies.append(fold_train_accuracies)
            val_accuracies.append(fold_val_accuracies)
            train_losses.append(fold_train_losses)
            val_losses.append(fold_val_losses)


            with torch.no_grad():
                model.eval()

                h_val = model(G_dgl, G_dgl.ndata['feat'])
                val_pos_score = pred(val_pos_g, h_val)
                val_neg_score = pred(val_neg_g, h_val)
                
                val_pos_score = torch.sigmoid(val_pos_score)
                val_neg_score = torch.sigmoid(val_neg_score)
                val_auc, val_auc_err = compute_auc_with_symmetrical_confidence(val_pos_score, val_neg_score)
                val_f1, val_f1_err = compute_f1_with_symmetrical_confidence(val_pos_score, val_neg_score)
                val_focal_loss, val_focal_loss_err = compute_focalloss_with_symmetrical_confidence(val_pos_score, val_neg_score)
                val_precision, val_precision_err = compute_precision_with_symmetrical_confidence(val_pos_score, val_neg_score)
                val_recall, val_recall_err = compute_recall_with_symmetrical_confidence(val_pos_score, val_neg_score)
                val_hits_k = compute_hits_k(val_pos_score, val_neg_score, k=10)
                val_map, val_map_err = compute_map_with_symmetrical_confidence(val_pos_score, val_neg_score)
                val_accuracy, val_accuracy_err = compute_accuracy_with_symmetrical_confidence(val_pos_score, val_neg_score)

                val_metrics = (
                    f'Val AUC: {val_auc:.4f} ± {val_auc_err:.4f} | Val F1: {val_f1:.4f} ± {val_f1_err:.4f} | '
                    f'Val FocalLoss: {val_focal_loss:.4f} ± {val_focal_loss_err:.4f} | Val Accuracy: {val_accuracy:.4f} ± {val_accuracy_err:.4f} | '
                    f'Val Precision: {val_precision:.4f} ± {val_precision_err:.4f} | Val Recall: {val_recall:.4f} ± {val_recall_err:.4f} | '
                    f'Val Hits@10: {val_hits_k:.4f} | Val MAP: {val_map:.4f} ± {val_map_err:.4f}'
                )
                print(val_metrics)

                # Save the val metrics to a .txt file
                output_path_val_metrics = f'GraphSAGE_SHS27k_BCEWithLogitsLoss_{args.sampling_method}_val_drop{dropout}_head{args.num_heads}_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.txt'
            
                with open(os.path.join(output_path, output_path_val_metrics), 'a') as f:
                    f.write(f'Fold {fold + 1}:\n')
                    f.write(val_metrics + '\n\n')

                h_test = model(G_dgl, G_dgl.ndata['feat'])
                test_pos_score = pred(test_pos_g, h_test)
                test_neg_score = pred(test_neg_g, h_test)
                
                # Apply sigmoid to test scores
                test_pos_score = torch.sigmoid(test_pos_score)
                test_neg_score = torch.sigmoid(test_neg_score)
                test_auc, test_auc_err = compute_auc_with_symmetrical_confidence(test_pos_score, test_neg_score)
                test_f1, test_f1_err = compute_f1_with_symmetrical_confidence(test_pos_score, test_neg_score)
                test_focal_loss, test_focal_loss_err = compute_focalloss_with_symmetrical_confidence(test_pos_score, test_neg_score)
                test_precision, test_precision_err = compute_precision_with_symmetrical_confidence(test_pos_score, test_neg_score)
                test_recall, test_recall_err = compute_recall_with_symmetrical_confidence(test_pos_score, test_neg_score)
                test_hits_k = compute_hits_k(test_pos_score, test_neg_score, k=10)
                test_map, test_map_err = compute_map_with_symmetrical_confidence(test_pos_score, test_neg_score)
                test_accuracy, test_accuracy_err = compute_accuracy_with_symmetrical_confidence(test_pos_score, test_neg_score)

                test_metrics = (
                    f'Test AUC: {test_auc:.4f} ± {test_auc_err:.4f} | Test F1: {test_f1:.4f} ± {test_f1_err:.4f} | '
                    f'Test FocalLoss: {test_focal_loss:.4f} ± {test_focal_loss_err:.4f} | Test Accuracy: {test_accuracy:.4f} ± {test_accuracy_err:.4f} | '
                    f'Test Precision: {test_precision:.4f} ± {test_precision_err:.4f} | Test Recall: {test_recall:.4f} ± {test_recall_err:.4f} | '
                    f'Test Hits@10: {test_hits_k:.4f} | Test MAP: {test_map:.4f} ± {test_map_err:.4f}'
                )
                print(test_metrics)

                # Save the test metrics to a .txt file
                output_path_test_metrics = f'GraphSAGE_SHS27k_BCEWithLogitsLoss_{args.sampling_method}_test_drop{dropout}_head{args.num_heads}_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.txt'

                with open(os.path.join(output_path, output_path_test_metrics), 'a') as f:
                    f.write(f'Fold {fold + 1}:\n')
                    f.write(test_metrics + '\n\n')

                true_labels = torch.cat([torch.ones(len(test_pos_score)), torch.zeros(len(test_neg_score))])
                predicted_scores = torch.cat([test_pos_score, test_neg_score]).cpu().numpy()

                fold_results.append((true_labels.cpu().numpy(), predicted_scores))

                fold_result_data = pd.DataFrame({
                    'Fold': [fold + 1],
                    'Test AUC': [test_auc.cpu().item() if torch.is_tensor(test_auc) else test_auc],
                    'Test AUC Err': [test_auc_err.cpu().item() if torch.is_tensor(test_auc_err) else test_auc_err],
                    'Test F1 Score': [test_f1.cpu().item() if torch.is_tensor(test_f1) else test_f1],
                    'Test F1 Score Err': [test_f1_err.cpu().item() if torch.is_tensor(test_f1_err) else test_f1_err],
                    'Test Precision': [test_precision.cpu().item() if torch.is_tensor(test_precision) else test_precision],
                    'Test Precision Err': [test_precision_err.cpu().item() if torch.is_tensor(test_precision_err) else test_precision_err],
                    'Test Recall': [test_recall.cpu().item() if torch.is_tensor(test_recall) else test_recall],
                    'Test Recall Err': [test_recall_err.cpu().item() if torch.is_tensor(test_recall_err) else test_recall_err],
                    'Test Hit': [test_hits_k.cpu().item() if torch.is_tensor(test_hits_k) else test_hits_k],
                    'Test mAP': [test_map.cpu().item() if torch.is_tensor(test_map) else test_map],
                    'Test mAP Err': [test_map_err.cpu().item() if torch.is_tensor(test_map_err) else test_map_err],
                    'Test FocalLoss': [test_focal_loss.cpu().item() if torch.is_tensor(test_focal_loss) else test_focal_loss],
                    'Test FocalLoss Err': [test_focal_loss_err.cpu().item() if torch.is_tensor(test_focal_loss_err) else test_focal_loss_err],
                    'Test Accuracy': [test_accuracy.cpu().item() if torch.is_tensor(test_accuracy) else test_accuracy],
                    'Test Accuracy Err': [test_accuracy_err.cpu().item() if torch.is_tensor(test_accuracy_err) else test_accuracy_err],
                })


                all_fold_results = pd.concat([all_fold_results, fold_result_data], ignore_index=True)

            
            all_fold_results_filename = f'GraphSAGE_SHS27k_BCEWithLogitsLoss_{args.sampling_method}_all_folds_results_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.csv'
            all_fold_results.to_csv(os.path.join(output_path, all_fold_results_filename), index=False)

            # Average metrics over folds
            avg_train_accuracies = np.mean(train_accuracies, axis=0)
            avg_val_accuracies = np.mean(val_accuracies, axis=0)
            avg_train_losses = np.mean(train_losses, axis=0)
            avg_val_losses = np.mean(val_losses, axis=0)

            # Plot and save ROC and PR curves for all folds
            ## plot_roc_pr_curves(fold_results, os.path.join(output_path, f'GraphSAGE_STRING_950_{args.sampling_method}_roc_pr_curves_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.png')) 
            ## True, (0.1, 0.2, 0.9, 1), (0.9, 1.0, 0.9, 1))
            ## (0.05, 0.15, 0.9, 1), (0.85, 0.95, 0.9, 1))
            ## plot_roc_pr_curves(fold_results, os.path.join(output_path, f'GraphSAGE_STRING_950_{args.sampling_method}_roc_pr_curves_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.png'), True, (0.05, 0.15, 0.9, 1), (0.85, 0.95, 0.9, 1))
            ## (0, 0.1, 0.9, 1), (0.9, 1, 0.9, 1))
            ## GIN (0.05, 0.15, 0.9, 1), (0.85, 0.95, 0.9, 1))
            ## plot_roc_pr_curves(fold_results, os.path.join(output_path, f'GraphSAGE_STRING_950_{args.sampling_method}_roc_pr_curves_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.png'), True, (0.1, 0.2, 0.7, 0.8), (0.8, 0.9, 0.7, 0.8))
            ## plot_roc_pr_curves(fold_results, os.path.join(output_path, f'GraphSAGE_STRING_950_{args.sampling_method}_roc_pr_curves_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.png'), True, (0, 0.1, 0.9, 1), (0.9, 1, 0.9, 1))
            plot_roc_pr_curves(fold_results, os.path.join(output_path, f'GraphSAGE_SHS27k_BCEWithLogitsLoss_{args.sampling_method}_roc_pr_curves_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.png'), True, (0.1, 0.2, 0.7, 0.8), (0.8, 0.9, 0.7, 0.8))

            # Plot and save training and validation metrics
            output_path_train = os.path.join(output_path, f'GraphSAGE_SHS27k_BCEWithLogitsLoss_{args.sampling_method}_f1_curve_head{args.num_heads}_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.png')

            plot_training_validation_metrics(
                train_accuracies, avg_train_accuracies,
                val_accuracies, avg_val_accuracies,
                train_losses, avg_train_losses,
                val_losses, avg_val_losses,
                output_path_train, args
            )

            # Process top predictions

            print('test_size==========\n', len(test_pos_u))

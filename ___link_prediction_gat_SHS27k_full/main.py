import argparse
import torch
from src.data_loader import load_graph_data
from src.train import train_and_evaluate

if __name__ == "__main__":
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Argument parser setup
    parser = argparse.ArgumentParser(description="MLP Predictor")
    parser.add_argument(
        "--sampling_method",
        type=str,
        choices=[
            "dfs_both_visible", "dfs_neither_visible", "dfs_one_visible",
            "bfs_both_visible", "bfs_neither_visible", "bfs_one_visible",
            "random_both_visible", "random_neither_visible", "random_one_visible"
        ],
        default="random_both_visible",
        help="Negative sampling method to use: dfs, bfs, or random"
    )
    parser.add_argument('--in-feats', type=int, default=256, help='Dimension of the first layer')
    parser.add_argument('--hidden-feats', type=int, default=256, help='Dimension of the hidden layer')
    parser.add_argument('--out-feats', type=int, default=32, help='Dimension of the final layer')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--input-size', type=int, default=2, help='Input size for the first linear layer')
    parser.add_argument('--hidden-size', type=int, default=16, help='Hidden size for the first linear layer')
    parser.add_argument('--feat-drop', type=float, default=0.5, help='Feature dropout rate')
    parser.add_argument('--attn-drop', type=float, default=0.5, help='Attention dropout rate')
    parser.add_argument('--model_type', type=str, choices=['GraphSAGE', 'GAT', 'GCN', 'GIN', 'Chebnet'], required=True)
    args = parser.parse_args()

    # Define protein interaction types to process
    ## interaction_types = ["ptmod", "catalysis", "ptmod", "reaction", "inhibition", "activation", "expression"]
    ##interaction_types = ["reaction"]

    ##graph_data_path = f'gat/data/protein.actions.SHS27k/protein.actions.SHS27k_full.json'
    graph_data_path = f'gat/data/protein.actions.SHS27k/protein.actions.SHS27k_expression.json'
    ##graph_data_path = f'gat/data/protein.actions.SHS148k/protein.actions.SHS148k_950.json'
    ##graph_data_path = f'gat/data/protein.actions.v10.5/9606_v10.5_950_ptmod.json'
    G_dgl, node_features, node_id_to_name = load_graph_data(graph_data_path, args.out_feats)

    # Move data to the GPU
    G_dgl = G_dgl.to(device)
    node_features = node_features.to(device)

    # Train and evaluate for each interaction type
    ##print(f"Training and evaluating for interaction type: {interaction_type}")
    train_and_evaluate(args, G_dgl, node_features) ##, interaction_type)  # Pass each interaction type individually

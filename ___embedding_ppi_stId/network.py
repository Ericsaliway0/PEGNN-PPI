import pandas as pd
import networkx as nx
from collections import defaultdict

class Network:

    def __init__(self, interaction_data_path):
        # Initialize an empty directed graph
        self.graph_nx = nx.DiGraph()
        
        # Load interaction data from CSV
        self.interaction_data = self.load_interaction_data(interaction_data_path)
        self.build_graph()

    def load_interaction_data(self, path):
        # Load protein interaction data from CSV
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()  # Strip any leading/trailing whitespace from column names
        return df

    def build_graph(self):
        # Build the graph by adding edges and attributes from the interaction data
        for _, row in self.interaction_data.iterrows():
            protein1 = row['protein1']
            protein2 = row['protein2']
            stId = row['stId']
            name = row['name']
            shared_partners = row['shared_partners']
            shared_count = row['shared_partners_count']
            p_value = row['p-value']
            significance = row['significance']
            
            # Add edge between proteins
            self.graph_nx.add_edge(protein1, protein2)
            
            # Store protein info in a dictionary format
            self.graph_nx.nodes[protein1]['stId'] = stId
            self.graph_nx.nodes[protein1]['name'] = name
            self.graph_nx.nodes[protein1]['significance'] = significance
            self.graph_nx.nodes[protein1]['shared_partners'] = shared_partners
            self.graph_nx.nodes[protein1]['shared_count'] = shared_count
            self.graph_nx.nodes[protein1]['p_value'] = p_value

    def get_protein_info(self, protein):
        # Retrieve the stored protein info
        if protein in self.graph_nx:
            return {
                'stId': self.graph_nx.nodes[protein]['stId'],
                'name': self.graph_nx.nodes[protein]['name'],
                'significance': self.graph_nx.nodes[protein]['significance'],
                'shared_partners': self.graph_nx.nodes[protein]['shared_partners'],
                'shared_count': self.graph_nx.nodes[protein]['shared_count'],
                'p_value': self.graph_nx.nodes[protein]['p_value']
            }
        else:
            return None

    def display_graph(self):
        # Display graph with node attributes (for debugging purposes)
        for node in self.graph_nx.nodes:
            info = self.get_protein_info(node)
            print(f"Protein: {node}, StId: {info['stId']}, Name: {info['name']}, Significance: {info['significance']}, Shared Count: {info['shared_count']}, P-value: {info['p_value']}")

    def save_name_to_id(self):
        # Save a mapping of protein names to IDs (stId)
        name_to_id = {node: self.graph_nx.nodes[node]['stId'] for node in self.graph_nx.nodes}
        file_path = 'name_to_id.txt'
        with open(file_path, 'w') as f:
            for name, stid in name_to_id.items():
                f.write(f"{name}: {stid}\n")

    def save_sorted_stids(self):
        # Save a sorted list of protein IDs (stIds)
        file_path = 'sorted_stids.txt'
        stids = sorted([self.graph_nx.nodes[node]['stId'] for node in self.graph_nx.nodes])
        with open(file_path, 'w') as f:
            for stid in stids:
                f.write(f"{stid}\n")

# Usage example
'''interaction_data_path = 'gat/data/protein_interaction_p_value_results_with_fdr.csv'
network = Network(interaction_data_path)
network.display_graph()
network.save_name_to_id()
network.save_sorted_stids()'''

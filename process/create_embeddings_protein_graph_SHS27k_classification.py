import csv
import json
import os

'''
source_csv_path = 'gat/data/protein.actions.SHS148k/protein_embeddings_lr0.0001_dim256_lay2_epo20_final_SHS148k_950.csv'
target_csv_path = 'gat/data/protein.actions.SHS148k/protein_embeddings_lr0.0001_dim256_lay2_epo20_final_SHS148k_950.csv'
relation_csv_path = 'gat/data/protein.actions.SHS148k/protein.actions.SHS148k_950.csv'
output_json_path = 'gat/data/protein.actions.SHS148k/protein.actions.SHS148k_950.json'
csv_file_path = 'gat/data/protein.actions.SHS148k/interaction_stats.csv'
'''
# File path

# File paths
source_csv_path = 'gat/data/protein.actions.SHS27k/protein_embeddings_lr0.01_dim256_lay2_epo40_final_SHS27k.csv'
target_csv_path = 'gat/data/protein.actions.SHS27k/protein_embeddings_lr0.01_dim256_lay2_epo40_final_SHS27k.csv'
relation_csv_path = 'gat/data/protein.actions.SHS27k/protein.actions.SHS27k_850.csv'
output_json_path = 'gat/data/protein.actions.SHS27k/protein.actions.SHS27k_850.json'
csv_file_path = 'gat/data/protein.actions.SHS27k/interaction_stats.csv'

interaction_types = ["ptmod", "catalysis", "reaction", "inhibition", "activation", "expression"]

# Ensure interaction stats CSV exists with a header
file_exists = os.path.isfile(csv_file_path)
if not file_exists:
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['num_nodes', 'num_edges']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

# Function to read embeddings from a CSV file
def read_embeddings(file_path):
    embeddings = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Skip the header
        for row in reader:
            name = row[0]
            embedding = list(map(float, row[1:]))
            embeddings[name] = embedding
    return embeddings

# Read source and target embeddings
source_embeddings = read_embeddings(source_csv_path)
target_embeddings = read_embeddings(target_csv_path)

# Read relationships and count nodes and edges
nodes = set()
edges = 0
relationships_to_include = []

with open(relation_csv_path, newline='') as csvfile:
    ##reader = csv.DictReader(csvfile, delimiter='\t')  # Assuming tab-delimited file
    reader = csv.DictReader(csvfile)
    for row in reader:
        source_stId = row['item_id_a']
        target_stId = row['item_id_b']
        relation_type = row['mode'] if row['mode'] in interaction_types else "unknown"
        nodes.add(source_stId)
        nodes.add(target_stId)
        relationships_to_include.append((source_stId, target_stId, relation_type))
        edges += 1

# Print node and edge counts
print(f"Number of nodes: {len(nodes)}")
print(f"Number of edges: {edges}")

# Create the JSON structure
relationships = []
for source_stId, target_stId, relation_type in relationships_to_include:
    if source_stId in source_embeddings and target_stId in target_embeddings:
        relationship = {
            "source": {
                "properties": {
                    "name": source_stId,
                    "embedding": source_embeddings[source_stId]
                }
            },
            "relation": {
                "type": relation_type  # Dynamically set relation type
            },
            "target": {
                "properties": {
                    "name": target_stId,
                    "embedding": target_embeddings[target_stId]
                }
            }
        }
        relationships.append(relationship)

# Save to JSON file
with open(output_json_path, 'w') as json_file:
    json.dump(relationships, json_file, indent=2)

print(f"JSON file saved to {output_json_path}")

# Save node and edge counts to CSV file
with open(csv_file_path, 'a', newline='') as csvfile:
    fieldnames = ['num_nodes', 'num_edges']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    # Write data
    writer.writerow({
        'num_nodes': len(nodes),
        'num_edges': edges
    })

print(f"CSV file updated with interaction data.")

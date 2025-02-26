import os
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# Function to compute p-value using Fisher's Exact Test
def compute_p_value_fisher_exact(N, K, M, n):
    contingency_table = [[n, K - n], [M - n, N - K - M + n]]
    _, p_value = fisher_exact(contingency_table, alternative='two-sided')
    return p_value

# Load the data
file_path = 'gat/data/protein.actions.SHS27k.inhibition.csv'
data = pd.read_csv(file_path)

# Ensure output directory exists
output_dir = 'gat/data'
os.makedirs(output_dir, exist_ok=True)

# Process the entire dataset
unique_proteins = pd.unique(data[['mode', 'action']].values.ravel())
N = len(unique_proteins)

protein_interaction_group = data.groupby('mode')['action'].nunique().reset_index()
protein_interaction_group.columns = ['protein', 'num_interactions']

protein_interaction_dict = protein_interaction_group.set_index('protein').to_dict()['num_interactions']

results = []
p_values = []

proteins = data['mode'].unique()

with tqdm(total=len(proteins) * (len(proteins) - 1) // 2, desc=f'Processing all data') as pbar:
    for i, proteini in enumerate(proteins):
        for j, proteinj in enumerate(proteins):
            if i < j:
                K = protein_interaction_dict.get(proteini, 0)
                M = protein_interaction_dict.get(proteinj, 0)

                partners_i = set(data[data['mode'] == proteini]['action'])
                partners_j = set(data[data['mode'] == proteinj]['action'])
                shared_partners = partners_i & partners_j
                n = len(shared_partners)

                p_value = compute_p_value_fisher_exact(N, K, M, n) if n > 0 else 1.0
                p_values.append(p_value)

                results.append({
                    'mode': proteini,
                    'stId': proteini,
                    'name': proteini,
                    'action': proteinj,
                    'is_directional': is_directional,
                    'shared_partners': ', '.join(shared_partners),
                    'shared_partners_count': n,
                    'p-value': p_value
                })
                pbar.update(1)

# Perform FDR correction
_, adjusted_p_values, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

# Add adjusted p-values and significance to results
for i, result in enumerate(results):
    result['adjusted_p-value'] = adjusted_p_values[i]
    result['significance'] = 'significant' if adjusted_p_values[i] < 0.05 else 'non-significant'

# Save results to a CSV
output_results_path = os.path.join(output_dir, 'inhibition_protein_interaction_p_value_results_with_fdr_SHS27k.csv')
results_df = pd.DataFrame(results)
results_df.to_csv(output_results_path, index=False)
print(f"Saved results with p-values to {output_results_path}")

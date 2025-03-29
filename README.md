## Learning Protein-Protein Interaction Networks with Graph Neural Networks and Pretrained Embeddings

This repository contains the code for our paper,
"Learning Protein-Protein Interaction Networks with Graph Neural Networks and Pretrained Embeddings,"
submitted to the **13th International Conference on Intelligent Biology and Medicine (ICIBM 2025)**,  
which will take place **August 10-12, 2025, in Columbus, OH, USA**.  

You can learn more about the conference here:  
[ICIBM 2025](https://icibm2025.iaibm.org/)

![Alt text](images/__overview_framework.png)



## Data Source

The dataset is obtained from the following sources:

- **[STRING](https://string-db.org/cgi/download?sessionId=b7WYyccF6G1p)**  
- **[SHS27k](https://pubmed.ncbi.nlm.nih.gov/31510705/)**  
- **[SHS148k](https://pubmed.ncbi.nlm.nih.gov/31510705/)** 

These databases provide curated and integrated protein-protein interaction (PPI) and pathway data for bioinformatics research.

## Setup and Get Started

1. Create Conda environment:
   - `conda create --name gnn python=3.11.3`

2. Activate the Conda environment:
   - `conda activate gnn`

3. Install PyTorch:
   - `conda install pytorch torchvision torchaudio -c pytorch`

4. Install DGL:
   - `conda install -c dglteam dgl`

5. To train the model, run the following command:
   - `python ___link_prediction_gat_SHS27k_full/main.py --in-feats 256 --out-feats 128 --num-heads 2 --num-layers 2 --lr 0.001 --input-size 512 --hidden-size 16 --sampling_method random_neither_visible --model_type GAT --epochs 501`


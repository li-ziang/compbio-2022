#!/usr/bin/env python
# coding: utf-8

# # Introduction to scETM
# 
# In this introductory tutorial, we will analyze a published mouse & human pancreas dataset (GSE84133) using scETM. We will first train and evaluate scETM on the mouse pancreas (MP) data, then transfer the results on the human pancreas (HP) data, and finally train a pathway-informed scETM (p-scETM) on the HP data.
# 
# ### Prepare data
# 
# scETM uses AnnData objects from the anndata package to represent single-cell datasets. AnnData is a versatile format compatible with multiple single-cell frameworks, e.g. scanpy and scvi-tools.

# In[1]:


import anndata as ad
import pandas as pd
import scanpy as sc
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scETM import scETM, UnsupervisedTrainer, evaluate, prepare_for_transfer
sc.set_figure_params(dpi=120, dpi_save=250, fontsize=10, figsize=(10, 10), facecolor="white")


# In[2]:


# Construct mouse pancreas AnnData object
import scanpy 
mp = scanpy.read_h5ad('/data/bingliang/bio/dataset/uc/data.h5ad')
mp


# In[3]:


mp.to_df().to_numpy()


# Note that the cell type information is in the "assigned_cluster" column, and the batch information is in the "batch_indices" column. The column names will be useful when we train and evaluate the model, so keep them in mind!
# 
# ### Train scETM
# 
# We then instantiate an scETM model, and use an UnsupervisedTrainer to train it.
# 
# Note that scETM requires about 6k steps to converge (observe the test NLL to confirm that), so for the MP dataset whose size is smaller than the training minibatch size (which means training for an epoch requires only one training step), it is recommended to train for at least 6k epochs.

# In[4]:


mp_model = scETM(mp.n_vars, mp.obs.batch_indices.nunique(), n_topics=5, trainable_gene_emb_dim=64, enable_ce_loss=True)
trainer = UnsupervisedTrainer(mp_model, mp, test_ratio=0.1)

trainer.train(n_epochs = 2000, eval_every = 500, eval_kwargs = dict(cell_type_col = 'assigned_cluster'), save_model_ckpt = True)


# In[12]:


def visualization(mp, alpha=None, rho=None, labels=None):
    size = 1000
    df = pd.DataFrame()
    if mp is not None:
        rna_embeddings = mp[:size, :].to_df().to_numpy()
        atac_embeddings = mp[-size:, :].to_df().to_numpy()
        print(rna_embeddings.shape)
        print(atac_embeddings.shape)
        embeddings =  np.concatenate((rna_embeddings, atac_embeddings))
        prefix = 'data'
        if labels is None:
            data_label = np.array(["health", "disease"])
            df['data'] = np.repeat(data_label, [rna_embeddings.shape[0], atac_embeddings.shape[0]], axis=0)
        else:
            df['data'] = labels
    elif alpha is not None and rho is not None:
        topic_embs = alpha[:size, :]
        gene_embs = rho[:size, :]
        print(topic_embs.shape)
        print(gene_embs.shape)
        embeddings = np.concatenate((topic_embs, gene_embs))
        prefix = 'embs'
        if labels is None:
            data_label = np.array(["topics", "genes"])
            df['data'] = np.repeat(data_label, [topic_embs.shape[0], gene_embs.shape[0]], axis=0)
        else:
            df['data'] = labels
    else:
        raise ValueError("mp or alpha and rho need to be specified.")
    print(embeddings.shape)
    tsne_results = TSNE(perplexity=30, n_iter = 1000).fit_transform(embeddings)
    tsne_results.shape
    
    df['tSNE1'] = tsne_results[:,0]
    df['tSNE2'] = tsne_results[:,1]

    
    plt.figure(figsize=(10,10))
    sns.scatterplot(
        x = "tSNE1", y = "tSNE2",
        hue = "data",
        palette = sns.color_palette("tab10", 2),
        data = df,
        legend = "full",
        alpha = 0.3
    )
    plt.savefig(f'/data/bingliang/bio/scETM/image/{prefix}_tsne.png')
    plt.show()
    plt.close()


# ### Get scETM output embeddings
# 
# The topic and trainale gene embeddings are stored in `model.alpha` and `model.rho_trainable`, respectively.
# 
# The cell embeddings are not stored in the model since we use amortized inference. Instead, they can be inferred from the data points using the encoder of the model using `model.get_cell_embeddings_and_nll(mp)`. After calling this function, you can find the cell embeddings in `mp.obs['delta']`.

# In[6]:


visualization(mp)


# In[7]:


# mp_model.get_cell_embeddings_and_nll(mp)
mp_model.get_all_embeddings_and_nll(mp)
alpha = mp.uns['alpha']
rho = mp.varm['rho']
visualization(None, alpha=alpha, rho=rho)


# ### Evaluate learned embeddings
# 
# As you can see in the training log, evaluation metrics are already printed every `eval_every` epochs. To explicitly evaluate the learned embedding, use the `evaluate` function provided by scETM. Note that you can use `evaluate` on any AnnData object that stores an embedding in its .obsm\[embedding_key\] or .X attribute.
# 
# The `evaluate` function looks for the `embedding_key` (which defaults to "delta") in adata.obsm, evaluates its ARI with cell type and batch, NMI with cell type, batch mixing entropy and kBET, then plots the embedding. Use `return_fig=True` to get the plotted figure, or specify `plot_dir` to save the figure to a file.
# 
# Note that we specify the `cell_type_col` and the `batch_col` arguments to let scETM find our cell type annotation and batch information for each cell.

# In[8]:


result = evaluate(mp, resolutions = [0.1, 0.15, 0.2, 0.3, 0.4, 0.8], return_fig=True, cell_type_col="assigned_cluster", batch_col="batch_indices")


# In[9]:


gene_names = mp.var_names.tolist()
gene_names_dict = {name: i for i, name in enumerate(gene_names)}
gene_names_dict


# In[26]:


targets = set(["IL23R", "NOD2", "TNF", "IL1A", "IL1B", "IL10", "PTPN2", "IRF5", "ABCB1", "IL6", "HLA-DRB1"])
less_targets = set(["IL37", "SIRT1", "IL23R", "PSTPIP1", "NEDD4L", "TPMT", "KRT8", "RLTPR", "TTC7A", "FCGR2A", "NOD2", "IL23R", "PTPN2", "CCL20", "TNF", "IL1B", "ICAM1", "IRF5", "IL10", "STAT3", "IL10RA", "IL10RB", "PTPN2", "IL23R", "IRF5", "ABCB1", "TNF", "IL6", "IL10", "NOD2", "IL1B", "NOD2", "IL6", "IL10", "TNF", "IL23R", "IL1B", "TLR4", "HLA-DRB1", "ABCB1", "CXCL8", "HLA-DRB1", "TNFSF15", "GPR35", "STMN3", "SATB2", "INSL6", "PROCR", "ZBTB46", "MIEN1"])
for t in targets.union(less_targets):
    if t not in gene_names_dict:
        print(t, t in gene_names_dict)
print([gene_names_dict[t] for  t in targets.union(less_targets)])


# In[38]:


size = 500
print(alpha.shape, rho.shape)
idx = np.random.choice(rho.shape[0], size=(size,), replace=False)
# print(idx)
idx_set = set(idx)
# embs_ad.obs['assigned_cluster'] = np.repeat(np.array(["topics", "genes"]), [alpha.shape[0], sampled_rho.shape[0]], axis=0)

missing_genes = [gene_names_dict[t] for t in targets.union(less_targets) if gene_names_dict[t] not in idx_set]
# print(len(missing_genes))
# print(missing_genes)

idx = np.concatenate([idx, missing_genes])
idx_set = set(idx)
idx_dict = {index: i for i, index in enumerate(idx)}

# print(idx[-50:])

sampled_rho = rho[idx, :]
embs_ad = ad.AnnData(X = np.concatenate([alpha, sampled_rho]))
data_labels = np.array(["topics", "known target genes", "other genes"])
labels = np.repeat(data_labels, [alpha.shape[0], 0, sampled_rho.shape[0]], axis=0)
cnt = 0
for i, index in enumerate(idx):
    if gene_names[index] in less_targets:
        labels[i] = "less known target genes"
    if gene_names[index] in targets:
#         print(i)
        labels[i] = "known target genes"
        cnt += 1

embs_ad.obs['assigned_cluster'] = labels
# print("cnt =", cnt)
# embs_ad.obs['assigned_cluster'].to_numpy()
result = evaluate(embs_ad, embedding_key='X', resolutions=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8], return_fig=True, cell_type_col="assigned_cluster", batch_col=None)


# In[71]:


def average_distances(targets_idx):
    n = len(targets_idx)
    dists = np.zeros((n, n))
    for i, index in enumerate(targets_idx):
        dists[i, :] = np.sqrt(np.sum((rho[index, :] - rho[targets_idx, :]) ** 2, axis=1))
    dist = np.sum(np.triu(dists)) / (n * (n + 1) / 2)
    return dist

all_targets = targets.union(less_targets)
n_targets = len(all_targets)

all_targets_idx = [gene_names_dict[t] for t in all_targets]
random_targets_idx = np.random.choice(rho.shape[0], size=(n_targets,), replace=False)

target_dist = average_distances(all_targets_idx)
random_dist = average_distances(random_targets_idx)

print(target_dist, random_dist)


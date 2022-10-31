import sys
import torch
import os
from datetime import datetime

from config import Config
from util.trainingprocess_stage1 import TrainingProcessStage1
from util.trainingprocess_stage3 import TrainingProcessStage3
from util.dataloader_stage1 import PrepareDataloader
import wandb
from util.knn import KNN
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def visualization(exp_name):
    rna_embeddings = np.loadtxt(f'./output/{exp_name}/citeseq_control_rna_embeddings.txt')
    atac_embeddings = np.loadtxt(f'./output/{exp_name}/asapseq_control_atac_embeddings.txt')
    print(rna_embeddings.shape)
    print(atac_embeddings.shape)
    embeddings =  np.concatenate((rna_embeddings, atac_embeddings))
    print(embeddings.shape)
    tsne_results = TSNE(perplexity=30, n_iter = 1000).fit_transform(embeddings)
    tsne_results.shape
    df = pd.DataFrame()
    df['tSNE1'] = tsne_results[:,0]
    df['tSNE2'] = tsne_results[:,1]
    
    rna_labels = np.loadtxt('./data/citeseq_control_cellTypes.txt')
    atac_labels = np.loadtxt('./data/asapseq_control_cellTypes.txt')
    atac_predictions = np.loadtxt(f'./output/{exp_name}/asapseq_control_atac_knn_predictions.txt')
    labels =  np.concatenate((rna_labels, atac_predictions))
    label_to_idx = pd.read_csv('./data/label_to_idx.txt', sep = '\t', header = None)
    label_to_idx.shape
    label_dic = []
    for i in range(label_to_idx.shape[0]):
        label_dic = np.append(label_dic, label_to_idx[0][i][:-2])
    
    data_label = np.array(["CITE-seq", "ASAP-seq"])
    df['data'] = np.repeat(data_label, [rna_embeddings.shape[0], atac_embeddings.shape[0]], axis=0)
    df['predicted'] = label_dic[labels.astype(int)]
    
    plt.figure(figsize=(10,10))
    sns.scatterplot(
        x = "tSNE1", y = "tSNE2",
        hue = "data",
        palette = sns.color_palette("tab10", 2),
        data = df,
        legend = "full",
        alpha = 0.3
    )
    plt.savefig(f'./output/{exp_name}/classification.png')
    plt.close()
    
    
    plt.figure(figsize=(10,10))
    sns.scatterplot(
        x = "tSNE1", y = "tSNE2",
        hue = "predicted",
        palette = sns.color_palette("Set2", 7),
        data = df,
        legend = "full",
        alpha = 0.3
    )
    plt.savefig(f'./output/{exp_name}/embedding.png')
    plt.close()


def visualization_data(exp_name):
    config = Config()  
    add_exp_config(exp_name, config)
    train_rna_loaders, _, train_atac_loaders, _, _ = PrepareDataloader(config).getloader()
    rna_embeddings = np.array([train_rna_loaders[0].dataset[i][0] for i in range(len(train_rna_loaders[0].dataset))])[:,0,:]
    atac_embeddings = np.array([train_atac_loaders[0].dataset[i] for i in range(len(train_atac_loaders[0].dataset))])[:,0,:]
    embeddings =  np.concatenate((rna_embeddings, atac_embeddings))
    print(embeddings.shape)
    tsne_results = TSNE(perplexity=30, n_iter = 1000).fit_transform(embeddings)
    tsne_results.shape
    df = pd.DataFrame()
    df['tSNE1'] = tsne_results[:,0]
    df['tSNE2'] = tsne_results[:,1]
    
    rna_labels = np.loadtxt('./data/citeseq_control_cellTypes.txt')
    atac_labels = np.loadtxt('./data/asapseq_control_cellTypes.txt')
    labels =  np.concatenate((rna_labels, atac_labels))
    label_to_idx = pd.read_csv('./data/label_to_idx.txt', sep = '\t', header = None)
    label_to_idx.shape
    label_dic = []
    for i in range(label_to_idx.shape[0]):
        label_dic = np.append(label_dic, label_to_idx[0][i][:-2])
    
    data_label = np.array(["CITE-seq", "ASAP-seq"])
    df['data'] = np.repeat(data_label, [rna_embeddings.shape[0], atac_embeddings.shape[0]], axis=0)
    df['predicted'] = label_dic[labels.astype(int)]
    
    plt.figure(figsize=(10,10))
    sns.scatterplot(
        x = "tSNE1", y = "tSNE2",
        hue = "data",
        palette = sns.color_palette("tab10", 2),
        data = df,
        legend = "full",
        alpha = 0.3
    )
    plt.savefig(f'./output/{exp_name}/classification.png')
    plt.close()
    
    
    plt.figure(figsize=(10,10))
    sns.scatterplot(
        x = "tSNE1", y = "tSNE2",
        hue = "predicted",
        palette = sns.color_palette("Set2", 7),
        data = df,
        legend = "full",
        alpha = 0.3
    )
    plt.savefig(f'./output/{exp_name}/embedding.png')
    plt.close()

def add_exp_config(name, config):
    config.epochs_stage1 = 10
    config.epochs_stage3 = 10
    
    if name == 'test':
        config.epochs_stage1 = 1
        config.epochs_stage3 = 1
    
    if name == 'origin':
        config.knn_metrics = 'uniform'

    if name == 'ce_only':
        config.reduction_loss = 0.0
    
    if name == 'exp_align':
        config.encoder_act = True
        config.sim_loss = 1.0
        config.rna_sim_loss = 1.0
        config.space_alignment = True
        config.knn_neighbor = 20
        config.reduction_boundedness = 0.0
        config.reduction_entanglement = 0.0
        config.reduction_separation = 0.0
        
    if name == 'final':
        config.encoder_act = True
        config.sim_loss = 1.0
        config.rna_sim_loss = 0.0
        config.space_alignment = True
        config.knn_neighbor = 20
        
    if name == 'mlp':
        config.encoder_act = True
    
    if name == 'without_reduction_loss':
        config.reduction_boundedness = 0.0
        config.reduction_entanglement = 0.0
        config.reduction_separation = 0.0

    if name == 'with_reduction_boundedness':
        config.reduction_boundedness = 1.0
        config.reduction_entanglement = 0.0
        config.reduction_separation = 0.0
    
    if name == 'with_reduction_entanglement':
        config.reduction_boundedness = 0.0
        config.reduction_entanglement = 1.0
        config.reduction_separation = 0.0
    
    if name == 'with_reduction_separation':
        config.reduction_boundedness = 0.0
        config.reduction_entanglement = 0.0
        config.reduction_separation = 1.0
    
    # if name.startswith('embed_dropout'):
    #     ratio = float(name.split('_')[-1])
    #     print(ratio)
    #     config.encoder_act = True
    #     config.encoder_drop_rate = ratio
    #     config.optim_adam = False
    
    # if name == 'rna_reduction':
    #     config.encoder_act = True
    #     config.drop_rate = 0
    #     config.optim_adam = False
    #     config.rna_reduction_loss = 0.0
        
    # if name == 'atac_reduction':
    #     config.encoder_act = True
    #     config.drop_rate = 0
    #     config.optim_adam = False
    #     config.atac_reduction_loss = 0.0
    
    # if name == 'rna_atac_reduction':
    #     config.encoder_act = True
    #     config.drop_rate = 0
    #     config.optim_adam = False
    #     config.rna_reduction_loss = 1.0
    #     config.atac_reduction_loss = 1.0
    
    # if name == 'reduction_log_separation':
    #     config.reduction_log_separation = True
    
    # if name == 'reduction_log_separation_mlp':
    #     config.reduction_log_separation = True
    #     config.encoder_act = True
    
    # if name == 'without_sim_loss':
    #     config.sim_loss = 0.0
    
    # if name == 'with_rna_sim_loss':
    #     config.sim_loss = 0.0
    #     config.rna_sim_loss = 1.0
    
    # if name == 'space_alignment':
    #     config.space_alignment = True
    #     config.rna_sim_loss = 0.0
    #     config.sim_loss = 0.0
    #     config.epochs_stage1 = 20
    
    # if name == 'space_alignment_mlp':
    #     config.space_alignment = True
    #     config.rna_sim_loss = 1.0
    #     config.sim_loss = 1.0
    #     config.encoder_act = True
    #     config.epochs_stage1 = 40
    
    # if name == 'space_alignment_mlp_test':
    #     config.space_alignment = True
    #     config.rna_sim_loss = 1.0
    #     config.sim_loss = 0
    #     config.encoder_act = True
    #     config.epochs_stage1 = 40
    #     config.reduction_boundedness = 0.0
    #     config.reduction_entanglement = 0.0
    #     config.reduction_separation = 0.0
    
    # if name == 'final2':
    #     config.encoder_act = True
    #     config.sim_loss = 1.0
    #     config.rna_sim_loss = 0.0
    #     config.space_alignment = True
    #     config.knn_neighbor = 20
    #     config.reduction_boundedness = 0.0
    #     config.reduction_entanglement = 0.0
    #     config.reduction_separation = 0.0
    
    # if name.startswith('final_'):
    #     config.knn_neighbor = int(name.split('_')[-1])
    #     config.sim_loss = 0.0
    #     config.space_alignment = True
    #     config.encoder_act = True
        
    config.exp_name = name
    return
        
    

def main(name):    
    # hardware constraint for speed test
    torch.set_num_threads(1)

    os.environ['OMP_NUM_THREADS'] = '1'
    
    # initialization 
    config = Config()  
    add_exp_config(name, config)
    print(config.exp_name)
    torch.manual_seed(config.seed)
    print('Start time: ', datetime.now().strftime('%H:%M:%S'))

    wandb.init(project="BioScJointFinal", name=name, config=config)
    # stage1 training
    print('Training start [Stage1]')
    model_stage1= TrainingProcessStage1(config)    
    for epoch in range(config.epochs_stage1):
        print('Epoch:', epoch)
        model_stage1.train(epoch)
    
    print('Write embeddings')
    model_stage1.write_embeddings(config.exp_name)
    print('Stage 1 finished: ', datetime.now().strftime('%H:%M:%S'))
    
    # KNN
    print('KNN')
    KNN(config, neighbors = config.knn_neighbor, knn_rna_samples=20000, stage1=True)
    print('KNN finished: ', datetime.now().strftime('%H:%M:%S'))
    
    
    # stage3 training
    print('Training start [Stage3]')
    model_stage3 = TrainingProcessStage3(config)    
    for epoch in range(config.epochs_stage3):
       print('Epoch:', epoch)
       model_stage3.train(epoch)
        
    print('Write embeddings [Stage3]')
    model_stage3.write_embeddings(config.exp_name)
    print('Stage 3 finished: ', datetime.now().strftime('%H:%M:%S'))
    
    # KNN
    print('KNN stage3')
    KNN(config, neighbors = config.knn_neighbor, knn_rna_samples=20000, stage1=False)
    print('KNN finished: ', datetime.now().strftime('%H:%M:%S'))
    
if __name__ == "__main__":
    name = sys.argv[-1]
    #visualization_data(name)
    main(name)
    #visualization(name)

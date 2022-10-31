import torch
import os

class Config(object):
    def __init__(self):
        DB = '10x'
        self.use_cuda = True
        self.threads = 1

        if not self.use_cuda:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')
        
        if DB == '10x':
            self.number_of_class = 7 # Number of cell types in CITE-seq data
            self.input_size = 17668 # Number of common genes and proteins between CITE-seq data and ASAP-seq
            self.rna_paths = ['data/citeseq_control_rna.npz'] # RNA gene expression from CITE-seq data
            self.rna_labels = ['data/citeseq_control_cellTypes.txt'] # CITE-seq data cell type labels (coverted to numeric) 
            self.atac_paths = ['data/asapseq_control_atac.npz'] # ATAC gene activity matrix from ASAP-seq data
            self.atac_labels = ['data/asapseq_control_cellTypes.txt'] # ASAP-seq data cell type labels (coverted to numeric) 
            self.rna_protein_paths = ['data/citeseq_control_adt.npz'] # Protein expression from CITE-seq data
            self.atac_protein_paths = ['data/asapseq_control_adt.npz'] # Protein expression from ASAP-seq data

            # Training config            
            self.batch_size = 256
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 20
            self.epochs_stage3 = 20
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 1
            self.num_threads = 1
            self.seed = 1
            self.with_crossentorpy = True
            self.checkpoint = ''   
            
            # Tuning config 
            self.exp_name = 'test'
            self.encoder_act = False
            self.encoder_drop_rate = 0
            self.optim_adam = False
            self.reduction_loss = 1.0
            self.rna_reduction_loss = 1.0
            self.atac_reduction_loss = 1.0
            self.reduction_entanglement = 1.0
            self.reduction_separation = 1.0
            self.reduction_boundedness = 1.0
            self.reduction_log_separation = False
            self.sim_loss = 1.0
            self.rna_sim_loss = 0.0
            self.space_alignment = False
            self.knn_neighbor = 30
            self.knn_metrics = 'distance'
            
        
        elif DB == "MOp":
            self.number_of_class = 21
            self.input_size = 18603
            self.rna_paths = ['data_MOp/YaoEtAl_RNA_snRNA_10X_v3_B_exprs.npz',\
                                'data_MOp/YaoEtAl_RNA_snRNA_10X_v3_A_exprs.npz',\
                                'data_MOp/YaoEtAl_RNA_snRNA_10X_v2_exprs.npz',\
                                'data_MOp/YaoEtAl_RNA_snRNA_SMARTer_exprs.npz',\
                                'data_MOp/YaoEtAl_RNA_scRNA_10X_v3_exprs.npz',\
                                'data_MOp/YaoEtAl_RNA_scRNA_10X_v2_exprs.npz',\
                                'data_MOp/YaoEtAl_RNA_scRNA_SMARTer_exprs.npz']
            self.rna_labels = ['data_MOp/YaoEtAl_RNA_snRNA_10X_v3_B_cellTypes.txt',\
                                'data_MOp/YaoEtAl_RNA_snRNA_10X_v3_A_cellTypes.txt',\
                                'data_MOp/YaoEtAl_RNA_snRNA_10X_v2_cellTypes.txt',\
                                'data_MOp/YaoEtAl_RNA_snRNA_SMARTer_cellTypes.txt',\
                                'data_MOp/YaoEtAl_RNA_scRNA_10X_v3_cellTypes.txt',\
                                'data_MOp/YaoEtAl_RNA_scRNA_10X_v2_cellTypes.txt',\
                                'data_MOp/YaoEtAl_RNA_scRNA_SMARTer_cellTypes.txt']
            self.atac_paths = ['data_MOp/YaoEtAl_ATAC_exprs.npz',\
                                'data_MOp/YaoEtAl_snmC_exprs.npz']
            self.atac_labels = ['data_MOp/YaoEtAl_ATAC_cellTypes.txt',\
                                'data_MOp/YaoEtAl_snmC_cellTypes.txt']
            self.rna_protein_paths = []
            self.atac_protein_paths = []
            
            # Training config            
            self.batch_size = 256
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.001
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 10
            self.epochs_stage3 = 10
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 20
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = '' 
            
        elif DB == "db4_control":
            self.number_of_class = 7 # Number of cell types in CITE-seq data
            self.input_size = 17668 # Number of common genes and proteins between CITE-seq data and ASAP-seq
            self.rna_paths = ['data/citeseq_control_rna.npz'] # RNA gene expression from CITE-seq data
            self.rna_labels = ['data/citeseq_control_cellTypes.txt'] # CITE-seq data cell type labels (coverted to numeric) 
            self.atac_paths = ['data/asapseq_control_atac.npz'] # ATAC gene activity matrix from ASAP-seq data
            self.atac_labels = ['data/asapseq_control_cellTypes.txt'] # ASAP-seq data cell type labels (coverted to numeric) 
            self.rna_protein_paths = ['data/citeseq_control_adt.npz'] # Protein expression from CITE-seq data
            self.atac_protein_paths = ['data/asapseq_control_adt.npz'] # Protein expression from ASAP-seq data
            
            # Training config            
            self.batch_size = 256
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 20
            self.epochs_stage3 = 20
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 1
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = '' 



            

        




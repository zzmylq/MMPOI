import logging
import logging
import os
import pathlib
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import time
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataloader import load_graph_adj_mtx, load_graph_node_features, load_graph_node_features_of_geo
from model import GCN, NodeAttnMap, UserEmbeddings, Time2Vec, CategoryEmbeddings, FuseEmbeddings, TransformerModel, OnlyTimeOrCatEmbeddings, FuseMultimodalEmbeddings
from param_parser import parameter_parser
from utils import increment_path, calculate_laplacian_matrix, zipdir, top_k_acc_last_timestep, \
    mAP_metric_last_timestep,  ndcg_last_timestep, MRR_metric_last_timestep, maksed_mse_loss

from sentence_transformers import SentenceTransformer

import geohash

import torch.nn.functional as F
from collections import OrderedDict

import random
args = parameter_parser()
seed = args.seed

random.seed(seed)

np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

dataset_name = 'NYC'

def train(args):
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, sep='-')
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    # Setup logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(args.save_dir, f"log_training.txt"),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    # Save run settings
    logging.info(args)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # Save python code
    zipf = zipfile.ZipFile(os.path.join(args.save_dir, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
    zipf.close()

    # %% ====================== Load data ======================
    # Read check-in train data
    train_df = pd.read_csv(args.data_train) 
    val_df = pd.read_csv(args.data_val)     

    # Build POI graph (built from train_df)
    print('Loading POI graph...')
    raw_A = load_graph_adj_mtx(args.data_adj_mtx)   
    raw_X = load_graph_node_features(args.data_node_feats,  
                                     args.feature1, 
                                     args.feature2, 
                                     args.feature3, 
                                     args.feature4) 

    if args.geo_graph_enabled:
        geo_raw_A = load_graph_adj_mtx(args.geo_adj_mtx)  
        geo_raw_X = load_graph_node_features_of_geo(args.geo_node_feats,
                                         "checkin_cnt",  
                                         "geohash")

    logging.info(
        f"raw_X.shape: {raw_X.shape}; "  # (4980,4)
        f"Four features: {args.feature1}, {args.feature2}, {args.feature3}, {args.feature4}.")
    logging.info(f"raw_A.shape: {raw_A.shape}; Edge from row_index to col_index with weight (frequency).")
    num_pois = raw_X.shape[0]   # 4980
    if args.geo_graph_enabled:
        num_geos = geo_raw_X.shape[0]

    # One-hot encoding poi categories
    logging.info('One-hot encoding poi categories id')
    one_hot_encoder = OneHotEncoder()
    cat_list = list(raw_X[:, 1])  
    for i in range(len(cat_list)):
        if str(cat_list[i]).lower() == 'nan':
            cat_list[i] = 'unknown'

    if args.geo_graph_enabled:
        geo_list = list(geo_raw_X[:, 1])
        geo_one_hot_encoder = OneHotEncoder()
        geo_one_hot_encoder.fit(list(map(lambda x: [x], geo_list)))  
        geo_one_hot_rlt = geo_one_hot_encoder.transform(list(map(lambda x: [x], geo_list))).toarray()  

    if args.cat_encoder == "OneHot":
        one_hot_encoder.fit(list(map(lambda x: [x], cat_list))) 
        one_hot_rlt = one_hot_encoder.transform(list(map(lambda x: [x], cat_list))).toarray()   
    elif args.cat_encoder == "SentenceTransformer":
        model_sentence = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=args.device)
        embeddings = model_sentence.encode(cat_list)
        one_hot_rlt = embeddings

    
    logging.info(f"use_time: {args.use_time}")
    logging.info(f"use_cat: {args.use_cat}")
    logging.info(f"cat_type: {args.cat_type}")
    logging.info(f"cat_freeze: {args.cat_freeze}")

    if args.geo_graph_enabled:
       with open(os.path.join(args.save_dir, 'geo-one-hot-encoder.pkl'), "wb") as f:
            pickle.dump(geo_one_hot_encoder, f)

    if args.cat_encoder == "OneHot":
        # logging.info(f'POI categories: {list(one_hot_encoder.categories_[0])}')
        # Save ont-hot encoder
        with open(os.path.join(args.save_dir, 'one-hot-encoder.pkl'), "wb") as f:
            pickle.dump(one_hot_encoder, f)

    # Normalization
    print('Laplician matrix...')    
    A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')
    if args.geo_graph_enabled:
        A_geo = calculate_laplacian_matrix(geo_raw_A, mat_type='hat_rw_normd_lap_mat')
   
    nodes_df = pd.read_csv(args.data_node_feats, delimiter="\t")    
    poi_ids = list(nodes_df['node_name/poi_id'].tolist())   
    poi_id2idx_dict = OrderedDict(zip(poi_ids, range(len(poi_ids))))

    # Cat id to index
    cat_ids = cat_list 
    cat_id2idx_dict = OrderedDict(zip(cat_ids, range(len(cat_ids))))

    # Poi idx to cat idx
    poi_idx2cat_idx_dict = {}  
    for i, row in nodes_df.iterrows():
        if not isinstance(row[args.feature2], str):
            row[args.feature2] = 'unknown'
        poi_idx2cat_idx_dict[poi_id2idx_dict[row['node_name/poi_id']]] = \
            cat_id2idx_dict[row[args.feature2]]

    def geohash_encode(lat, lon, precision=6):
        """Encode latitude and longitude into geohash"""
        return geohash.encode(lat, lon, precision=precision)

    nodes_df['geohash'] = nodes_df.apply(lambda row: geohash_encode(row['latitude'], row['longitude']), axis=1)

    if args.geo_graph_enabled:
        # Geo id to index
        geo_nodes_df = pd.read_csv(args.geo_node_feats, delimiter="\t")  
        geo_ids = list(geo_nodes_df['geohash'].tolist())
        geo_id2idx_dict = OrderedDict(zip(geo_ids, range(len(geo_ids))))

        # Poi idx to geo idx
        poi_idx2geo_idx_dict = {}
        for i, row in nodes_df.iterrows():
            poi_idx2geo_idx_dict[poi_id2idx_dict[row['node_name/poi_id']]] = \
                geo_id2idx_dict[row['geohash']]

        num_geos = len(geo_ids)
        geo_embed_size = geo_one_hot_rlt.shape[-1]
        X_geo = geo_one_hot_rlt
        logging.info(f"After one hot encoding poi geo, X_geo.shape: {X_geo.shape}")

    num_cats = len(cat_ids)
 
    X = one_hot_rlt     
    logging.info(f"After one hot encoding poi cat, X.shape: {X.shape}")

    def load_multimodal_data(filename):
        filename_saved = filename[0:-5] + '.pkl'
        if os.path.exists(filename_saved):
            with open(filename_saved, 'rb') as f:
                data = pickle.load(f)
            return data
        with open(filename, 'r') as f:
            data = f.readlines()
            data = [eval(line) for line in data]
            data = {k: v for d in data for k, v in d.items()}
            padding = [0.0 for i in range(args.NLP_embedding_dim)]
            for k, v in data.items():
                if len(v) == 786:
                    data[k] = padding
        with open(filename_saved, 'wb') as f:
            pickle.dump(data, f)
        return data
    if args.multimodal_enabled:
        image_embeddings_filename = f'/home/CaiZhuoXiao/MMPOI-main/dataset/NYC/modal_image_embedding.json'
        review_embeddings_filename = f'/home/CaiZhuoXiao/MMPOI-main/dataset/NYC/modal_review_embedding.json'
        if dataset_name in ['Alaska', 'Hawaii']:
            meta_embeddings_filename = f'/home/CaiZhuoXiao/MMPOI-main/dataset/NYC/modal_meta_embedding.json'
            review_summary_embeddings_filename = f'/home/CaiZhuoXiao/MMPOI-main/dataset/NYC/modal_review_summary_embedding.json'
        elif dataset_name in ['NYC', 'TKY', 'GB']:
            meta_embeddings_filename = f'/home/CaiZhuoXiao/MMPOI-main/dataset/NYC/modal_meta_embedding.json'
        print('loading multimodal data...')
        image_dict = load_multimodal_data(image_embeddings_filename)
        review_dict = load_multimodal_data(review_embeddings_filename)
        if dataset_name in ['Alaska', 'Hawaii']:
            meta_dict = load_multimodal_data(meta_embeddings_filename)
            review_summary_dict = load_multimodal_data(review_summary_embeddings_filename)
        elif dataset_name in ['NYC', 'TKY', 'GB']:
            meta_dict = load_multimodal_data(meta_embeddings_filename)
        X_image = np.array([image_dict[poi_id] for poi_id in poi_ids])    # poi_ids是所有poi的id列表，顺序是A邻接矩阵的行列顺序
        X_review = np.array([review_dict[poi_id] for poi_id in poi_ids])
        if dataset_name in ['Alaska', 'Hawaii']:
            X_meta = np.array([meta_dict[poi_id] for poi_id in poi_ids])
            X_review_summary = np.array([review_summary_dict[poi_id] for poi_id in poi_ids])
            logging.info(f"After loading multimodal data, X_meta.shape: {X_meta.shape}")
            logging.info(f"After loading multimodal data, X_review_summary.shape: {X_review_summary.shape}")
        elif dataset_name in ['NYC', 'TKY', 'GB']:
            X_meta = np.array([meta_dict[poi_id] for poi_id in poi_ids])
            X_meta = X_meta.squeeze(1)
            logging.info(f"After loading multimodal data, X_meta.shape: {X_meta.shape}")
        logging.info(f"After loading multimodal data, X_image.shape: {X_image.shape}")
        logging.info(f"After loading multimodal data, X_review.shape: {X_review.shape}")


    user_ids = sorted(list(set(train_df['user_id'].tolist())))
    user_ids = list(map(str, user_ids))

    user_id2idx_dict = OrderedDict(zip(user_ids, range(len(user_ids))))

    # %% ====================== Define Dataset ======================
    class TrajectoryDatasetTrain(Dataset):
        def __init__(self, train_df):
            self.df = train_df
            self.traj_seqs = []  
            self.input_seqs = []
            self.label_seqs = []

            traj_data_dict = {traj_id: sub_df for traj_id, sub_df in train_df.groupby('trajectory_id')}


            traj_ids = sorted(train_df['trajectory_id'].unique().tolist())
            for traj_id in tqdm(traj_ids):
                traj_df = traj_data_dict[traj_id]
               
                poi_ids = traj_df['POI_id'].to_list()  
                poi_idxs = [poi_id2idx_dict[each] for each in poi_ids] 

                input_seq = []  
                label_seq = []  

                if args.use_time:
                    time_feature = traj_df[args.time_feature].to_list() 
                    for i in range(len(poi_idxs) - 1):
                        input_seq.append((poi_idxs[i], time_feature[i]))
                        label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))
                else:
                    for i in range(len(poi_idxs) - 1):
                        input_seq.append((poi_idxs[i],))
                        label_seq.append((poi_idxs[i + 1],))

                if len(input_seq) < args.short_traj_thres:  
                    continue

                self.traj_seqs.append(traj_id)   
                self.input_seqs.append(input_seq)   
                self.label_seqs.append(label_seq)   

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    class TrajectoryDatasetVal(Dataset):
        def __init__(self, df):
            self.df = df
            self.traj_seqs = []
            self.input_seqs = []
            self.label_seqs = []

            traj_data_dict = {traj_id: sub_df for traj_id, sub_df in df.groupby('trajectory_id')}

            traj_ids = sorted(df['trajectory_id'].unique().tolist())
            for traj_id in tqdm(traj_ids):
                user_id = traj_id.split('_')[0]

                # Ignore user if not in training set
                if user_id not in user_id2idx_dict.keys():
                    continue

                # Ger POIs idx in this trajectory
                traj_df = traj_data_dict[traj_id]
                poi_ids = traj_df['POI_id'].to_list() 
                poi_idxs = []  

                for each in poi_ids:   
                    if each in poi_id2idx_dict.keys():
                        poi_idxs.append(poi_id2idx_dict[each])
                    else:
                        # Ignore poi if not in training set
                        continue

                input_seq = []
                label_seq = []   

                if args.use_time:
                    time_feature = traj_df[args.time_feature].to_list()  
                    for i in range(len(poi_idxs) - 1):
                        input_seq.append((poi_idxs[i], time_feature[i]))
                        label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))
                else:
                    for i in range(len(poi_idxs) - 1):
                        input_seq.append((poi_idxs[i],))
                        label_seq.append((poi_idxs[i + 1],))

                # Ignore seq if too short
                if len(input_seq) < args.short_traj_thres:
                    continue

                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)
                self.traj_seqs.append(traj_id)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    # %% ====================== Define dataloader ======================
    print('Prepare dataloader...')
    train_dataset = TrajectoryDatasetTrain(train_df)
    val_dataset = TrajectoryDatasetVal(val_df)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=False, drop_last=False,
                              pin_memory=True, num_workers=args.workers,
                              collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch,
                            shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers,
                            collate_fn=lambda x: x)

    # %% ====================== Build Models ======================
    # Model1: POI embedding model
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X) 
        A = torch.from_numpy(A)
    X = X.to(device=args.device, dtype=torch.float)
    A = A.to(device=args.device, dtype=torch.float)

    if args.multimodal_enabled:
        X_image = torch.from_numpy(X_image)
        X_image = X_image.to(device=args.device, dtype=torch.float)
        X_review = torch.from_numpy(X_review)
        X_review = X_review.to(device=args.device, dtype=torch.float)
        if dataset_name in ['Alaska', 'Hawaii']:
            X_meta = torch.from_numpy(X_meta)
            X_meta = X_meta.to(device=args.device, dtype=torch.float)
            X_review_summary = torch.from_numpy(X_review_summary)
            X_review_summary = X_review_summary.to(device=args.device, dtype=torch.float)
        elif dataset_name in ['NYC', 'TKY', 'GB']:
            X_meta = torch.from_numpy(X_meta)
            X_meta = X_meta.to(device=args.device, dtype=torch.float)
        args.gcn_image_nfeat = X_image.shape[1]
        image_embed_model = GCN(ninput=args.gcn_image_nfeat,
                              nhid=args.gcn_nhid,  
                              noutput=args.multimodal_embed_dim,  
                              dropout=args.gcn_dropout)
        args.gcn_review_nfeat = X_review.shape[1]
        review_embed_model = GCN(ninput=args.gcn_review_nfeat,
                                 nhid=args.gcn_nhid,  
                                 noutput=args.multimodal_embed_dim,  
                                 dropout=args.gcn_dropout)

        if dataset_name in ['Alaska', 'Hawaii']:
            args.gcn_meta_nfeat = X_meta.shape[1]
            meta_embed_model = GCN(ninput=args.gcn_meta_nfeat,
                                  nhid=args.gcn_nhid,  
                                  noutput=args.multimodal_embed_dim,  
                                  dropout=args.gcn_dropout)

            args.gcn_review_summary_nfeat = X_review_summary.shape[1]
            review_summary_embed_model = GCN(ninput=args.gcn_review_summary_nfeat,
                                  nhid=args.gcn_nhid,  
                                  noutput=args.multimodal_embed_dim,  
                                  dropout=args.gcn_dropout)
        elif dataset_name in ['NYC', 'TKY', 'GB']:
            args.gcn_meta_nfeat = X_meta.shape[1]
            meta_embed_model = GCN(ninput=args.gcn_meta_nfeat,
                                   nhid=args.gcn_nhid,  
                                   noutput=args.multimodal_embed_dim, 
                                   dropout=args.gcn_dropout)

    args.gcn_nfeat = X.shape[1]  
    poi_embed_model = GCN(ninput=args.gcn_nfeat, 
                          nhid=args.gcn_nhid,   
                          noutput=args.poi_embed_dim,   
                          dropout=args.gcn_dropout) 

    # Model1.5: Geo embedding model
    if args.geo_graph_enabled:
        if isinstance(X_geo, np.ndarray):
            X_geo = torch.from_numpy(X_geo)  
            A_geo = torch.from_numpy(A_geo)  
        X_geo = X_geo.to(device=args.device, dtype=torch.float)
        A_geo = A_geo.to(device=args.device, dtype=torch.float)
        args.gcn_geo_nfeat = X_geo.shape[1]  
        geo_embed_model = GCN(ninput=args.gcn_geo_nfeat,
                                nhid=args.gcn_nhid,   
                                noutput=args.geo_embed_dim,   
                                dropout=args.gcn_dropout)


    # %% Model2: User embedding model, nn.embedding
    num_users = len(user_id2idx_dict)
    user_embed_model = UserEmbeddings(num_users, args.user_embed_dim)

    # %% Model3: Time Model
    if args.use_time:
        time_embed_model = Time2Vec('sin', out_dim=args.time_embed_dim)

    # %% Model4: Category embedding model
    if args.use_cat:  # one_hot_rlt
        cat_embed_model = CategoryEmbeddings(num_cats, args.cat_embed_dim, one_hot_rlt)

    # %% Model5: Embedding fusion models
    if dataset_name in ['Alaska', 'Hawaii']:
        embed_fuse_multimodal_model = FuseMultimodalEmbeddings(args.multimodal_embed_dim, 4)
    elif dataset_name in ['NYC', 'TKY', 'GB']:
        embed_fuse_multimodal_model = FuseMultimodalEmbeddings(args.multimodal_embed_dim, 3)
    else:
        embed_fuse_multimodal_model = FuseMultimodalEmbeddings(args.multimodal_embed_dim, 2)

    embed_fuse_model1 = FuseEmbeddings(args.user_embed_dim, args.poi_embed_dim)

    if args.use_time and args.geo_graph_enabled:
        embed_fuse_model2 = FuseEmbeddings(args.time_embed_dim, args.geo_embed_dim)
    elif args.use_time and not args.geo_graph_enabled:
        embed_fuse_model2 = OnlyTimeOrCatEmbeddings(args.time_embed_dim)


    # %% Model6: Sequence model
    if args.use_time and args.geo_graph_enabled:
        if dataset_name in ['Alaska', 'Hawaii']:
            args.seq_input_embed = args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim + args.geo_embed_dim
        elif dataset_name in ['NYC', 'TKY', 'GB']:
            args.seq_input_embed = args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim + args.geo_embed_dim+ args.multimodal_embed_dim*3
        else:
            args.seq_input_embed = args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim + args.geo_embed_dim+ args.multimodal_embed_dim*3
    else:
        if dataset_name in ['Alaska', 'Hawaii']:
            args.seq_input_embed = args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim
        elif dataset_name in ['NYC', 'TKY', 'GB']:
            args.seq_input_embed = args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim+ args.multimodal_embed_dim*3
        else:
            args.seq_input_embed = args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim+ args.multimodal_embed_dim*3
    if args.geo_graph_enabled:
        seq_model = TransformerModel(num_pois,
                                 num_cats,
                                 num_geos,
                                 args.seq_input_embed,
                                 args.transformer_nhead,
                                 args.transformer_nhid,
                                 args.transformer_nlayers,
                                 args.use_time,
                                 args.use_cat,
                                 args.geo_graph_enabled,
                                 dropout=args.transformer_dropout)
    else:
        seq_model = TransformerModel(num_pois,
                                     num_cats,
                                     1000,
                                     args.seq_input_embed,
                                     args.transformer_nhead,
                                     args.transformer_nhid,
                                     args.transformer_nlayers,
                                     args.use_time,
                                     args.use_cat,
                                     args.geo_graph_enabled,
                                     dropout=args.transformer_dropout)

    if args.multi_loss_weight:
        task_weight1 = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        task_weight2 = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        task_weight3 = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        task_weight4 = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        if args.task_weight_negative:
            precision1 = torch.exp(-task_weight1)
            precision2 = torch.exp(-task_weight2)
            precision3 = torch.exp(-task_weight3)
            precision4 = torch.exp(-task_weight4)
        else:
            precision1 = torch.exp(task_weight1)
            precision2 = torch.exp(task_weight2)
            precision3 = torch.exp(task_weight3)
            precision4 = torch.exp(task_weight4)
        l2_regularization = torch.tensor(0., requires_grad=True)
        val_epochs_task_weight1_list = []
        val_epochs_task_weight2_list = []
        val_epochs_task_weight3_list = []
        val_epochs_task_weight4_list = []


    # Define overall loss and optimizer
    if args.fusion_method == 'fusion' or args.fusion_method == 'concat':
        if args.use_time and args.geo_graph_enabled:
            if args.multi_loss_weight:
                if dataset_name in ['Alaska', 'Hawaii']:
                    optimizer = optim.Adam(params=list(poi_embed_model.parameters()) +
                                              # list(node_attn_model.parameters()) +
                                              list(geo_embed_model.parameters()) +
                                              list(user_embed_model.parameters()) +
                                              list(image_embed_model.parameters()) +
                                              list(meta_embed_model.parameters()) +
                                              list(review_embed_model.parameters()) +
                                              list(review_summary_embed_model.parameters()) +
                                              list(time_embed_model.parameters()) +
                                              list(embed_fuse_multimodal_model.parameters()) +
                                              list(embed_fuse_model1.parameters()) +
                                              list(embed_fuse_model2.parameters()) +
                                              list(seq_model.parameters()) +
                                                [task_weight1, task_weight2, task_weight3, task_weight4],
                                       lr=args.lr,
                                       weight_decay=args.weight_decay)
                elif dataset_name in ['NYC', 'TKY', 'GB']:
                    optimizer = optim.Adam(params=list(poi_embed_model.parameters()) +
                                                  # list(node_attn_model.parameters()) +
                                                  list(geo_embed_model.parameters()) +
                                                  list(user_embed_model.parameters()) +
                                                  list(image_embed_model.parameters()) +
                                                  list(meta_embed_model.parameters()) +
                                                  list(review_embed_model.parameters()) +
                                                  list(time_embed_model.parameters()) +
                                                  list(embed_fuse_multimodal_model.parameters()) +
                                                  list(embed_fuse_model1.parameters()) +
                                                  list(embed_fuse_model2.parameters()) +
                                                  list(seq_model.parameters()) +
                                                  [task_weight1, task_weight2, task_weight3, task_weight4],
                                           lr=args.lr,
                                           weight_decay=args.weight_decay)
                else:
                    optimizer = optim.Adam(params=list(poi_embed_model.parameters()) +
                                                  # list(node_attn_model.parameters()) +
                                                  list(geo_embed_model.parameters()) +
                                                  list(user_embed_model.parameters()) +
                                                  list(image_embed_model.parameters()) +
                                                  list(review_embed_model.parameters()) +
                                                  list(time_embed_model.parameters()) +
                                                  list(embed_fuse_multimodal_model.parameters()) +
                                                  list(embed_fuse_model1.parameters()) +
                                                  list(embed_fuse_model2.parameters()) +
                                                  list(seq_model.parameters()) +
                                                  [task_weight1, task_weight2, task_weight3, task_weight4],
                                           lr=args.lr,
                                           weight_decay=args.weight_decay)
            else:
                optimizer = optim.Adam(params=list(poi_embed_model.parameters()) +
                                              # list(node_attn_model.parameters()) +
                                              list(geo_embed_model.parameters()) +
                                              list(user_embed_model.parameters()) +
                                              list(image_embed_model.parameters()) +
                                              list(meta_embed_model.parameters()) +
                                              list(review_embed_model.parameters()) +
                                              list(review_summary_embed_model.parameters()) +
                                              list(time_embed_model.parameters()) +
                                              list(embed_fuse_multimodal_model.parameters()) +
                                              list(embed_fuse_model1.parameters()) +
                                              list(embed_fuse_model2.parameters()) +
                                              list(seq_model.parameters()),
                                       lr=args.lr,
                                       weight_decay=args.weight_decay)
        else:
            optimizer = optim.Adam(params=list(poi_embed_model.parameters()) +
                                          # list(node_attn_model.parameters()) +
                                          list(user_embed_model.parameters()) +
                                          list(time_embed_model.parameters()) +
                                          list(embed_fuse_model1.parameters()) +
                                          list(embed_fuse_model2.parameters()) +
                                          list(seq_model.parameters()),
                                   lr=args.lr,
                                   weight_decay=args.weight_decay)
    else:
        if args.use_time and args.geo_graph_enabled:
            optimizer = optim.Adam(params=list(poi_embed_model.parameters()) +
                                          # list(node_attn_model.parameters()) +
                                          list(geo_embed_model.parameters()) +
                                          list(user_embed_model.parameters()) +
                                          list(time_embed_model.parameters()) +
                                          list(seq_model.parameters()),
                                   lr=args.lr,
                                   weight_decay=args.weight_decay)
        else:
            optimizer = optim.Adam(params=list(poi_embed_model.parameters()) +
                                          # list(node_attn_model.parameters()) +
                                          list(user_embed_model.parameters()) +
                                          list(time_embed_model.parameters()) +
                                          list(seq_model.parameters()),
                                   lr=args.lr,
                                   weight_decay=args.weight_decay)

    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_geo = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding

    if args.reg_enabled:
        params = optimizer.param_groups[0]['params']
        for param in params:
            param_copy = param.clone().to(args.device).detach()
            l2_regularization = torch.add(l2_regularization, torch.norm(param_copy, p=2))
        l2_regularization = l2_regularization.to(args.device)

    if args.use_time:
        criterion_time = maksed_mse_loss

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, factor=args.lr_scheduler_factor) 

    # %% Tool functions for training
    def input_traj_to_embeddings(sample, poi_embeddings):
        # Parse sample
        traj_id = sample[0]
        input_seq = [each[0] for each in sample[1]]
        if args.use_time:
            input_seq_time = [each[1] for each in sample[1]]


        # User to embedding
        user_id = traj_id.split('_')[0]
        user_idx = user_id2idx_dict[user_id]
        input = torch.LongTensor([user_idx]).to(device=args.device)
        user_embedding = user_embed_model(input)
        user_embedding = torch.squeeze(user_embedding)
        # POI to embedding and fuse embeddings
        input_seq_embed = []
        for idx in range(len(input_seq)):
            poi_embedding = poi_embeddings[input_seq[idx]]
            poi_embedding = torch.squeeze(poi_embedding).to(device=args.device)

            # Time to vector
            if args.use_time:
                time_embedding = time_embed_model(
                    torch.tensor([input_seq_time[idx]], dtype=torch.float).to(device=args.device))
                time_embedding = torch.squeeze(time_embedding).to(device=args.device)

            # Fuse user+poi embeds
            if args.fusion_method == 'fusion':
                fused_embedding1 = embed_fuse_model1(user_embedding, poi_embedding)

                if args.use_time and args.use_cat:
                    fused_embedding2 = embed_fuse_model2(time_embedding, cat_embedding)
                elif args.use_time and not args.use_cat:
                    fused_embedding2 = embed_fuse_model2(time_embedding)
                elif not args.use_time and args.use_cat:
                    fused_embedding2 = embed_fuse_model2(cat_embedding)
            else:
                fused_embedding1 = torch.cat((user_embedding, poi_embedding), dim=-1)

                if args.use_time and args.use_cat:
                    fused_embedding2 = torch.cat((time_embedding, cat_embedding), dim=-1)
                elif args.use_time and not args.use_cat:
                    fused_embedding2 = time_embedding
                elif not args.use_time and args.use_cat:
                    fused_embedding2 = cat_embedding

            # Concat time, cat after user+poi
            if args.use_time or args.use_cat:
                concat_embedding = torch.cat((fused_embedding1, fused_embedding2), dim=-1)
            else:
                concat_embedding = fused_embedding1

            # Save final embed
            input_seq_embed.append(concat_embedding)

        return input_seq_embed

    def input_traj_to_embeddings_with_geo(sample, poi_embeddings, geo_embeddings, 
    image_embeddings, meta_embeddings, review_embeddings, review_summary_embeddings
    ):  
        # Parse sample
        traj_id = sample[0]
        input_seq = [each[0] for each in sample[1]]
        if args.use_time:
            input_seq_time = [each[1] for each in sample[1]]

        input_seq_geo = [poi_idx2geo_idx_dict[each] for each in input_seq]

        # User to embedding
        user_id = traj_id.split('_')[0]
        user_idx = user_id2idx_dict[user_id]
        input = torch.LongTensor([user_idx]).to(device=args.device)
        user_embedding = user_embed_model(input)  
        user_embedding = torch.squeeze(user_embedding)

        # POI to embedding and fuse embeddings
        input_seq_embed = []
        for idx in range(len(input_seq)):  
            poi_embedding = poi_embeddings[input_seq[idx]]
            poi_embedding = torch.squeeze(poi_embedding).to(device=args.device)

            geo_embedding = geo_embeddings[input_seq_geo[idx]]
            geo_embedding = torch.squeeze(geo_embedding).to(device=args.device)

            image_embedding = image_embeddings[input_seq[idx]]
            image_embedding = torch.squeeze(image_embedding).to(device=args.device)
            review_embedding = review_embeddings[input_seq[idx]]
            review_embedding = torch.squeeze(review_embedding).to(device=args.device)
            if dataset_name in ['Alaska', 'Hawaii']:
                meta_embedding = meta_embeddings[input_seq[idx]]
                meta_embedding = torch.squeeze(meta_embedding).to(device=args.device)
                review_summary_embedding = review_summary_embeddings[input_seq[idx]]
                review_summary_embedding = torch.squeeze(review_summary_embedding).to(device=args.device)
            elif dataset_name in ['NYC', 'TKY', 'GB']:
                meta_embedding = meta_embeddings[input_seq[idx]]
                meta_embedding = torch.squeeze(meta_embedding).to(device=args.device)

            # Time to vector
            if args.use_time:
                time_embedding = time_embed_model(
                    torch.tensor([input_seq_time[idx]], dtype=torch.float).to(device=args.device))
                time_embedding = torch.squeeze(time_embedding).to(device=args.device)

            # Fuse user+poi embeds
            if dataset_name in ['Alaska', 'Hawaii']:
                multimodal_fused_embedding = embed_fuse_multimodal_model(image_embedding, meta_embedding, review_embedding, review_summary_embedding)
            elif dataset_name in ['NYC', 'TKY', 'GB']:
                multimodal_fused_embedding = embed_fuse_multimodal_model(image_embedding, meta_embedding,
                                                                         review_embedding)
            else:
                multimodal_fused_embedding = embed_fuse_multimodal_model(image_embedding, review_embedding)
            fused_embedding1 = embed_fuse_model1(user_embedding, poi_embedding)

            if args.use_time:
                fused_embedding2 = embed_fuse_model2(time_embedding, geo_embedding)
            else:
                fused_embedding2 = embed_fuse_model2(geo_embedding)
                
            concat_embedding = torch.cat((fused_embedding1, fused_embedding2, multimodal_fused_embedding), dim=-1)
            # concat_embedding = torch.cat((fused_embedding1, fused_embedding2), dim=-1)

            # Save final embed
            input_seq_embed.append(concat_embedding)

        return input_seq_embed

    def adjust_pred_prob_by_graph(y_pred_poi):
        pass

     # %% ====================== Train ======================

    poi_embed_model = poi_embed_model.to(device=args.device)
    # node_attn_model = node_attn_model.to(device=args.device)
    user_embed_model = user_embed_model.to(device=args.device)
    image_embed_model = image_embed_model.to(device=args.device)
    review_embed_model = review_embed_model.to(device=args.device)
    if dataset_name in ['Alaska', 'Hawaii']:
        meta_embed_model = meta_embed_model.to(device=args.device)
        review_summary_embed_model = review_summary_embed_model.to(device=args.device)
    elif dataset_name in ['NYC', 'TKY', 'GB']:
        meta_embed_model = meta_embed_model.to(device=args.device)
    embed_fuse_multimodal_model = embed_fuse_multimodal_model.to(device=args.device)

    if args.use_time:
        time_embed_model = time_embed_model.to(device=args.device)
    if args.use_cat:
        cat_embed_model = cat_embed_model.to(device=args.device)
    if args.geo_graph_enabled:
        geo_embed_model = geo_embed_model.to(device=args.device)
    if args.fusion_method == 'fusion':
        embed_fuse_model1 = embed_fuse_model1.to(device=args.device)
        if args.use_time:
            embed_fuse_model2 = embed_fuse_model2.to(device=args.device)
    seq_model = seq_model.to(device=args.device)


    # %% Loop epoch
    # For plotting
    train_epochs_top1_acc_list = []
    train_epochs_top5_acc_list = []
    train_epochs_top10_acc_list = []
    train_epochs_top20_acc_list = []
    train_epochs_mAP20_list = []
    train_epochs_ndcg1_list = [] 
    train_epochs_ndcg5_list = [] 
    train_epochs_ndcg10_list = [] 
    train_epochs_ndcg20_list = [] 
    train_epochs_mrr_list = []
    train_epochs_loss_list = []
    train_epochs_poi_loss_list = []
    if args.use_time:
        train_epochs_time_loss_list = []
    if args.geo_graph_enabled:
        train_epochs_geo_loss_list = []
    train_epochs_cat_loss_list = []
    val_epochs_top1_acc_list = []
    val_epochs_top5_acc_list = []
    val_epochs_top10_acc_list = []
    val_epochs_top20_acc_list = []
    val_epochs_mAP20_list = []
    val_epochs_ndcg1_list = []
    val_epochs_ndcg5_list = []
    val_epochs_ndcg10_list = []
    val_epochs_ndcg20_list = []
    val_epochs_mrr_list = []
    val_epochs_loss_list = []
    val_epochs_poi_loss_list = []
    if args.use_time:
        val_epochs_time_loss_list = []
    if args.geo_graph_enabled:
        val_epochs_geo_loss_list = []
    val_epochs_cat_loss_list = []
    # For saving ckpt
    max_val_score = -np.inf

    # For early stopping
    best_val_score = 0
    best_val_epoch = 0
    best_val_top1_acc = 0
    best_val_top5_acc = 0
    best_val_top10_acc = 0
    best_val_top20_acc = 0
    best_val_mAP20 = 0
    best_val_ndcg1 = 0
    best_val_ndcg5 = 0
    best_val_ndcg10 = 0
    best_val_ndcg20 = 0
    best_val_mrr = 0
    previous_val_top20_acc = 0
    previous_val_ndcg20 = 0
    patience_times = 0

    last_val_score = 0
    last_val_epoch = 0
    last_val_top1_acc = 0
    last_val_top5_acc = 0
    last_val_top10_acc = 0
    last_val_top20_acc = 0
    last_val_mAP20 = 0
    last_val_ndcg1 = 0
    last_val_ndcg5 = 0
    last_val_ndcg10 = 0
    last_val_ndcg20 = 0
    last_val_mrr = 0


    def build_sim(context):
        context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        return sim

    def build_knn_neighbourhood(adj, topk):
        knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return weighted_adjacency_matrix

    def compute_normalized_laplacian(adj):
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return L_norm

    if args.use_A_plus:
        image_adj = build_sim(X_image)
        image_adj = build_knn_neighbourhood(image_adj, args.knn_k)
        review_adj = build_sim(X_review)
        review_adj = build_knn_neighbourhood(review_adj, args.knn_k)
        if dataset_name in ['Alaska', 'Hawaii']:
            meta_adj = build_sim(X_meta)
            meta_adj = build_knn_neighbourhood(meta_adj, args.knn_k)
            review_summary_adj = build_sim(X_review_summary)
            review_summary_adj = build_knn_neighbourhood(review_summary_adj, args.knn_k)
        elif dataset_name in ['NYC', 'TKY', 'GB']:
            meta_adj = build_sim(X_meta)
            meta_adj = build_knn_neighbourhood(meta_adj, args.knn_k)
        image_adj[image_adj != image_adj] = 0
        review_adj[review_adj != review_adj] = 0
        if dataset_name in ['Alaska', 'Hawaii']:
            meta_adj[meta_adj != meta_adj] = 0
            review_summary_adj[review_summary_adj != review_summary_adj] = 0
        elif dataset_name in ['NYC', 'TKY', 'GB']:
            meta_adj[meta_adj != meta_adj] = 0
        if dataset_name in ['Alaska', 'Hawaii']:
            multimodal_adj = args.multimodal_graph_weights[0] * image_adj \
                             + args.multimodal_graph_weights[1] * meta_adj \
                             + args.multimodal_graph_weights[2] * review_adj \
                             + args.multimodal_graph_weights[3] * review_summary_adj
        elif dataset_name in ['NYC', 'TKY', 'GB']:
            multimodal_adj = args.multimodal_graph_weights[0] * image_adj \
                             + args.multimodal_graph_weights[1] * meta_adj \
                             + args.multimodal_graph_weights[2] * review_adj
        else:
            multimodal_adj = args.multimodal_graph_weights[0] * image_adj \
                             + args.multimodal_graph_weights[2] * review_adj
        multimodal_adj = compute_normalized_laplacian(multimodal_adj)

        A_plus = args.A_plus_weights[0] * A + args.A_plus_weights[1] * multimodal_adj
        A_plus = args.A_plus_weights[0] * A + args.A_plus_weights[1] * A
        A = A_plus


    for epoch in range(args.epochs):
        time_epoch_start = time.time()
        logging.info(f"{'*' * 50}Epoch:{epoch:03d}{'*' * 50}\n")
        poi_embed_model.train()
        # node_attn_model.train()
        user_embed_model.train()

        image_embed_model.train()
        review_embed_model.train()
        if dataset_name in ['Alaska', 'Hawaii']:
            meta_embed_model.train()
            review_summary_embed_model.train()
        elif dataset_name in ['NYC', 'TKY', 'GB']:
            meta_embed_model.train()
        embed_fuse_multimodal_model.train()

        if args.use_time:
            time_embed_model.train()
        if args.use_cat:
            cat_embed_model.train()
        if args.geo_graph_enabled:
            geo_embed_model.train()
        if args.fusion_method == 'fusion':
            embed_fuse_model1.train()
            if args.use_time:
                embed_fuse_model2.train()
        seq_model.train()

        train_batches_top1_acc_list = []
        train_batches_top5_acc_list = []
        train_batches_top10_acc_list = []
        train_batches_top20_acc_list = []
        train_batches_mAP20_list = []
        train_batches_ndcg1_list = []
        train_batches_ndcg5_list = []
        train_batches_ndcg10_list = []
        train_batches_ndcg20_list = []
        train_batches_mrr_list = []
        train_batches_loss_list = []
        train_batches_poi_loss_list = []
        if args.use_time:
            train_batches_time_loss_list = []
        if args.geo_graph_enabled:
            train_batches_geo_loss_list = []
        train_batches_cat_loss_list = []
        src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device) 
        # Loop batch
        for b_idx, batch in enumerate(train_loader):
            if len(batch) != args.batch:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)

            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []
            if args.use_time:
                batch_seq_labels_time = []
            if args.geo_graph_enabled:
                batch_seq_labels_geo = []
            batch_seq_labels_cat = []


            if args.geo_graph_enabled:
                geo_embeddings = geo_embed_model(X_geo, A_geo)

            poi_embeddings = poi_embed_model(X, A)  
            image_embeddings = image_embed_model(X_image, A)
            review_embeddings = review_embed_model(X_review, A)
            if dataset_name in ['Alaska', 'Hawaii']:
                meta_embeddings = meta_embed_model(X_meta, A)
                review_summary_embeddings = review_summary_embed_model(X_review_summary, A)
            elif dataset_name in ['NYC', 'TKY', 'GB']:
                meta_embeddings = meta_embed_model(X_meta, A)
            # Convert input seq to embeddings
            for sample in batch: 
             
                traj_id = sample[0]
                input_seq = [each[0] for each in sample[1]] 
                label_seq = [each[0] for each in sample[2]] 
                if args.use_time:
                    input_seq_time = [each[1] for each in sample[1]]
                    label_seq_time = [each[1] for each in sample[2]]

                label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
                if args.geo_graph_enabled:
                    lable_seq_geos = [poi_idx2geo_idx_dict[each] for each in label_seq]
                    if dataset_name in ['Alaska', 'Hawaii']:
                        input_seq_embed = torch.stack(input_traj_to_embeddings_with_geo(sample, poi_embeddings,
                                                                                    geo_embeddings, 
                                                                                    image_embeddings,
                                                                                    meta_embeddings,
                                                                                    review_embeddings,
                                                                                    review_summary_embeddings
                                                                                    ))  
                    elif dataset_name in ['NYC', 'TKY', 'GB']:
                        input_seq_embed = torch.stack(input_traj_to_embeddings_with_geo(sample, poi_embeddings,
                                                                                        geo_embeddings,
                                                                                        image_embeddings,
                                                                                        meta_embeddings,
                                                                                        review_embeddings,
                                                                                        review_embeddings
                                                                                        ))  
                    else:
                        input_seq_embed = torch.stack(input_traj_to_embeddings_with_geo(sample, poi_embeddings,
                                                                                       geo_embeddings,
                                                                                       image_embeddings,
                                                                                        image_embeddings,
                                                                                       review_embeddings,
                                                                                        review_embeddings
                                                                                        ))  
                else:
                    input_seq_embed = torch.stack(input_traj_to_embeddings(sample, poi_embeddings))  
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))  
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                if args.use_time:
                    batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))

                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))
                if args.geo_graph_enabled:
                    batch_seq_labels_geo.append(torch.LongTensor(lable_seq_geos))

            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1) 
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            if args.use_time:
                label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)

            label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)
            if args.geo_graph_enabled:
                label_padded_geo = pad_sequence(batch_seq_labels_geo, batch_first=True, padding_value=-1)

            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)  
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            if args.use_time:
                y_time = label_padded_time.to(device=args.device, dtype=torch.float)
            if args.geo_graph_enabled:
                y_geo = label_padded_geo.to(device=args.device, dtype=torch.long)

            y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
            if args.use_time and args.geo_graph_enabled:
                y_pred_poi, y_pred_time, y_pred_cat, y_pred_geo = seq_model(x, src_mask)
            else:
                y_pred_poi, y_pred_time, y_pred_cat = seq_model(x, src_mask)

            # Graph Attention adjusted prob
            if args.adjust_enabled:
                y_pred_poi_adjusted = adjust_pred_prob_by_graph(y_pred_poi)
            else:
                y_pred_poi_adjusted = y_pred_poi

            loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)
            if args.use_time:
                loss_time = criterion_time(torch.squeeze(y_pred_time), y_time)
            if args.geo_graph_enabled:
                loss_geo = criterion_geo(y_pred_geo.transpose(1, 2), y_geo)

            if args.cat_loss_type == 'id_loss':
                loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat) 
            elif args.cat_loss_type == 'embedding_loss':

                mask = (y_cat != -1)

                y_cat_filtered = y_cat.masked_select(mask)

                y_pred_cat_filtered = y_pred_cat.masked_select(mask.unsqueeze(-1)).view(-1, y_pred_cat.size(-1))

                filtered_embeddings = torch.index_select(X, 0, y_cat_filtered.long())

                probabilities_pred = F.softmax(y_pred_cat_filtered, dim=1)

                probabilities_true = filtered_embeddings

                probabilities_true_flat = probabilities_true.view(-1, y_pred_cat.size(-1))
                probabilities_pred_flat = probabilities_pred.view(-1, y_pred_cat.size(-1))


                loss_cat = F.kl_div(probabilities_pred_flat.log(), probabilities_true_flat, reduction='batchmean')


 # Final loss
            if args.multi_loss_weight:
                if args.reg_enabled:
                    if args.loss_time_enabled and args.loss_cat_enabled and args.loss_geo_enabled:
                        loss = precision1 * loss_poi + task_weight1 \
                               + precision2 * loss_time + task_weight2 \
                               + precision3 * loss_cat + task_weight3 \
                               + precision4 * loss_geo + task_weight4 \
                               + args.reg_weight * l2_regularization
                    elif not args.loss_time_enabled and args.loss_cat_enabled and args.loss_geo_enabled:
                        loss = precision1 * loss_poi + task_weight1 \
                               + precision3 * loss_cat + task_weight3 \
                               + precision4 * loss_geo + task_weight4 \
                               + args.reg_weight * l2_regularization
                    elif args.loss_time_enabled and not args.loss_cat_enabled and args.loss_geo_enabled:
                        loss = precision1 * loss_poi + task_weight1 \
                               + precision2 * loss_time + task_weight2 \
                               + precision4 * loss_geo + task_weight4 \
                               + args.reg_weight * l2_regularization
                    elif args.loss_time_enabled and args.loss_cat_enabled and not args.loss_geo_enabled:
                        loss = precision1 * loss_poi + task_weight1 \
                               + precision2 * loss_time + task_weight2 \
                               + precision3 * loss_cat + task_weight3 \
                               + args.reg_weight * l2_regularization

                else:
                    if args.loss_time_enabled and args.loss_cat_enabled and args.loss_geo_enabled:
                        loss = precision1 * loss_poi + task_weight1 \
                               + precision2 * loss_time + task_weight2 \
                               + precision3 * loss_cat + task_weight3 \
                               + precision4 * loss_geo + task_weight4
                    elif not args.loss_time_enabled and args.loss_cat_enabled and args.loss_geo_enabled:
                        loss = precision1 * loss_poi + task_weight1 \
                               + precision3 * loss_cat + task_weight3 \
                               + precision4 * loss_geo + task_weight4
                    elif not args.loss_time_enabled and not args.loss_cat_enabled and args.loss_geo_enabled:
                        loss = precision1 * loss_poi + task_weight1 \
                               + precision4 * loss_geo + task_weight4
                    elif not args.loss_time_enabled and args.loss_cat_enabled and not args.loss_geo_enabled:
                        loss = precision1 * loss_poi + task_weight1 \
                               + precision3 * loss_cat + task_weight3
                    elif args.loss_time_enabled and not args.loss_cat_enabled and args.loss_geo_enabled:
                        loss = precision1 * loss_poi + task_weight1 \
                               + precision2 * loss_time + task_weight2 \
                               + precision4 * loss_geo + task_weight4
                    elif args.loss_time_enabled and args.loss_cat_enabled and not args.loss_geo_enabled:
                        loss = precision1 * loss_poi + task_weight1 \
                               + precision2 * loss_time + task_weight2 \
                               + precision3 * loss_cat + task_weight3
            else:
                if args.use_time and args.geo_graph_enabled:
                    loss = loss_poi + loss_time * args.time_loss_weight + loss_cat * args.cat_loss_weight + loss_geo * args.geo_loss_weight
                else:
                    loss = loss_poi + loss_time * args.time_loss_weight + loss_cat * args.cat_loss_weight
            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            max_norm = 1.0 
            # torch.nn.utils.clip_grad_norm_([task_weight3], max_norm)

            optimizer.step()

            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            ndcg1 = 0
            ndcg5 = 0
            ndcg10 = 0
            ndcg20 = 0
            mrr = 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi_adjusted.detach().cpu().numpy()
            if args.use_time:
                batch_pred_times = y_pred_time.detach().cpu().numpy()
            if args.geo_graph_enabled:
                batch_pred_geos = y_pred_geo.detach().cpu().numpy()
            batch_pred_cats = y_pred_cat.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len] 
                pred_pois = pred_pois[:seq_len, :]  
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)   
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                ndcg1 += ndcg_last_timestep(label_pois, pred_pois, k=1)
                ndcg5 += ndcg_last_timestep(label_pois, pred_pois, k=5)
                ndcg10 += ndcg_last_timestep(label_pois, pred_pois, k=10)
                ndcg20 += ndcg_last_timestep(label_pois, pred_pois, k=20)
                
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
            train_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            train_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            train_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            train_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            train_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            train_batches_ndcg1_list.append(ndcg1 / len(batch_label_pois))
            train_batches_ndcg5_list.append(ndcg5 / len(batch_label_pois))
            train_batches_ndcg10_list.append(ndcg10 / len(batch_label_pois))
            train_batches_ndcg20_list.append(ndcg20 / len(batch_label_pois))
            
            train_batches_mrr_list.append(mrr / len(batch_label_pois))
            train_batches_loss_list.append(loss.detach().cpu().numpy())
            train_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            if args.use_time:
                train_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
            train_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())
            if args.geo_graph_enabled:
                train_batches_geo_loss_list.append(loss_geo.detach().cpu().numpy())
            # Report training progress
            if (b_idx % (args.batch * 5)) == 0:
                sample_idx = 0
                batch_pred_pois_wo_attn = y_pred_poi.detach().cpu().numpy()
                if args.use_time and args.geo_graph_enabled:
                    logging.info(f'Epoch:{epoch}, batch:{b_idx}, '
                             f'train_batch_loss:{loss.item():.2f}, '
                             f'train_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                             f'train_move_loss:{np.mean(train_batches_loss_list):.2f}\n'
                             f'train_move_poi_loss:{np.mean(train_batches_poi_loss_list):.2f}\n'
                             f'train_move_time_loss:{np.mean(train_batches_time_loss_list):.2f}\n'
                             f'train_move_geo_loss:{np.mean(train_batches_geo_loss_list):.2f}\n'
                             f'train_move_top1_acc:{np.mean(train_batches_top1_acc_list):.4f}\n'
                             f'train_move_top5_acc:{np.mean(train_batches_top5_acc_list):.4f}\n'
                             f'train_move_top10_acc:{np.mean(train_batches_top10_acc_list):.4f}\n'
                             f'train_move_top20_acc:{np.mean(train_batches_top20_acc_list):.4f}\n'
                             f'train_move_mAP20:{np.mean(train_batches_mAP20_list):.4f}\n'
                             f'train_move_ndcg1:{np.mean(train_batches_ndcg1_list):.4f}\n'
                             f'train_move_ndcg5:{np.mean(train_batches_ndcg5_list):.4f}\n'
                             f'train_move_ndcg10:{np.mean(train_batches_ndcg10_list):.4f}\n'
                             f'train_move_ndcg20:{np.mean(train_batches_ndcg20_list):.4f}\n'
                             
                             f'train_move_MRR:{np.mean(train_batches_mrr_list):.4f}\n'
                             f'traj_id:{batch[sample_idx][0]}\n'
                             f'input_seq: {batch[sample_idx][1]}\n'
                             f'label_seq:{batch[sample_idx][2]}\n'
                             f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_cat:{[poi_idx2cat_idx_dict[each[0]] for each in batch[sample_idx][2]]}\n'
                             f'pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_time:{list(batch_seq_labels_time[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n' 
                             f'pred_seq_time:{list(np.squeeze(batch_pred_times)[sample_idx][:batch_seq_lens[sample_idx]])} \n' +
                             '=' * 100)
                else:
                    logging.info(f'Epoch:{epoch}, batch:{b_idx}, '
                                 f'train_batch_loss:{loss.item():.2f}, '
                                 f'train_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                                 f'train_move_loss:{np.mean(train_batches_loss_list):.2f}\n'
                                 f'train_move_poi_loss:{np.mean(train_batches_poi_loss_list):.2f}\n'
                                 f'train_move_top1_acc:{np.mean(train_batches_top1_acc_list):.4f}\n'
                                 f'train_move_top5_acc:{np.mean(train_batches_top5_acc_list):.4f}\n'
                                 f'train_move_top10_acc:{np.mean(train_batches_top10_acc_list):.4f}\n'
                                 f'train_move_top20_acc:{np.mean(train_batches_top20_acc_list):.4f}\n'
                                 f'train_move_mAP20:{np.mean(train_batches_mAP20_list):.4f}\n'
                                 f'train_move_ndcg1:{np.mean(train_batches_ndcg1_list):.4f}\n'
                                 f'train_move_ndcg5:{np.mean(train_batches_ndcg5_list):.4f}\n'
                                 f'train_move_ndcg10:{np.mean(train_batches_ndcg10_list):.4f}\n'
                                 f'train_move_ndcg20:{np.mean(train_batches_ndcg20_list):.4f}\n'
                                 
                                 f'train_move_MRR:{np.mean(train_batches_mrr_list):.4f}\n'
                                 f'traj_id:{batch[sample_idx][0]}\n'
                                 f'input_seq: {batch[sample_idx][1]}\n'
                                 f'label_seq:{batch[sample_idx][2]}\n'
                                 f'label_seq_cat:{[poi_idx2cat_idx_dict[each[0]] for each in batch[sample_idx][2]]}\n'
                                 f'pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                                 f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                                 f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n' +
                                 '=' * 100)
                time_batch_end = time.time()
                print("batch:", b_idx, " cost time:", time_epoch_start - time_batch_end)

        # train end --------------------------------------------------------------------------------------------------------
        poi_embed_model.eval()
        # node_attn_model.eval()
        user_embed_model.eval()
        image_embed_model.eval()
        review_embed_model.eval()
        if dataset_name in ['Alaska', 'Hawaii']:
            meta_embed_model.eval()
            review_summary_embed_model.eval()
        elif dataset_name in ['NYC', 'TKY', 'GB']:
            meta_embed_model.eval()
        embed_fuse_multimodal_model.eval()
        if args.use_time:
            time_embed_model.eval()
        if args.use_cat:
            cat_embed_model.eval()
        if args.geo_graph_enabled:
            geo_embed_model.eval()
        if args.fusion_method == "fusion":
            embed_fuse_model1.eval()
            if args.use_time:
                embed_fuse_model2.eval()
        seq_model.eval()
        val_batches_top1_acc_list = []
        val_batches_top5_acc_list = []
        val_batches_top10_acc_list = []
        val_batches_top20_acc_list = []
        val_batches_mAP20_list = []
        val_batches_ndcg1_list = []
        val_batches_ndcg5_list = []
        val_batches_ndcg10_list = []
        val_batches_ndcg20_list = []
        
        val_batches_mrr_list = []
        val_batches_loss_list = []
        val_batches_poi_loss_list = []
        val_batches_cat_loss_list = []
        val_batches_geo_loss_list = []
        if args.use_time:
            val_batches_time_loss_list = []
        val_batches_cat_loss_list = []
        if args.geo_graph_enabled:
            val_batches_geo_loss_list = []
        src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)
        for vb_idx, batch in enumerate(val_loader):
            if len(batch) != args.batch:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)

            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []
            if args.use_time:
                batch_seq_labels_time = []
            batch_seq_labels_cat = []
            if args.geo_graph_enabled:
                batch_seq_labels_geo = []

            if args.geo_graph_enabled:
                geo_embeddings = geo_embed_model(X_geo, A_geo)

            poi_embeddings = poi_embed_model(X, A)
            image_embeddings = image_embed_model(X_image, A)
            review_embeddings = review_embed_model(X_review, A)
            if dataset_name in ['Alaska', 'Hawaii']:
                meta_embeddings = meta_embed_model(X_meta, A)
                review_summary_embeddings = review_summary_embed_model(X_review_summary, A)
            elif dataset_name in ['NYC', 'TKY', 'GB']:
                meta_embeddings = meta_embed_model(X_meta, A)
            # Convert input seq to embeddings
            for sample in batch:
                traj_id = sample[0]
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                if args.use_time:
                    input_seq_time = [each[1] for each in sample[1]]
                    label_seq_time = [each[1] for each in sample[2]]
                label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
                if args.geo_graph_enabled:
                    label_seq_geos = [poi_idx2geo_idx_dict[each] for each in label_seq]
                    if dataset_name in ['Alaska', 'Hawaii']:
                        input_seq_embed = torch.stack(input_traj_to_embeddings_with_geo(sample, poi_embeddings, geo_embeddings,
                                                                                    image_embeddings, meta_embeddings,
                                                                                    review_embeddings, review_summary_embeddings
                                                                                    ))
                    elif dataset_name in ['NYC', 'TKY', 'GB']:
                        input_seq_embed = torch.stack(
                            input_traj_to_embeddings_with_geo(sample, poi_embeddings, geo_embeddings,
                                                              image_embeddings, meta_embeddings,
                                                              review_embeddings, review_embeddings
                                                              ))
                    else:
                        input_seq_embed = torch.stack(input_traj_to_embeddings_with_geo(sample, poi_embeddings, geo_embeddings,
                                                                                    image_embeddings, image_embeddings,
                                                                                        review_embeddings, review_embeddings
                                                                                        ))
                else:
                    input_seq_embed = torch.stack(input_traj_to_embeddings(sample, poi_embeddings))
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                if args.geo_graph_enabled:
                    batch_seq_labels_geo.append(torch.LongTensor(label_seq_geos))
                if args.use_time:
                    batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))

            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            if args.use_time:
                label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)
            label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)
            if args.geo_graph_enabled:
                label_padded_geo = pad_sequence(batch_seq_labels_geo, batch_first=True, padding_value=-1)
            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)  # label
            if args.use_time:
                y_time = label_padded_time.to(device=args.device, dtype=torch.float)
            y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
            if args.geo_graph_enabled:
                y_geo = label_padded_geo.to(device=args.device, dtype=torch.long)
            if args.use_time and args.geo_graph_enabled:
                y_pred_poi, y_pred_time, y_pred_cat, y_pred_geo = seq_model(x, src_mask)
            else:
                y_pred_poi, y_pred_time, y_pred_cat = seq_model(x, src_mask)

            y_pred_poi_adjusted = y_pred_poi

            # Calculate loss
            loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)

            if args.use_time:

                if y_pred_time.shape[0] == 1:
                    break
                loss_time = criterion_time(torch.squeeze(y_pred_time), y_time)
            if args.geo_graph_enabled:
                loss_geo = criterion_geo(y_pred_geo.transpose(1, 2), y_geo)
            if args.cat_loss_type == 'id_loss':
                loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)
            elif args.cat_loss_type == 'embedding_loss':
                mask = (y_cat != -1)

                y_cat_filtered = y_cat.masked_select(mask)

                y_pred_cat_filtered = y_pred_cat.masked_select(mask.unsqueeze(-1)).view(-1, y_pred_cat.size(-1))

                filtered_embeddings = torch.index_select(X, 0, y_cat_filtered.long())

                probabilities_pred = F.softmax(y_pred_cat_filtered, dim=1)

                probabilities_true = filtered_embeddings

                probabilities_true_flat = probabilities_true.view(-1, y_pred_cat.size(-1))
                probabilities_pred_flat = probabilities_pred.view(-1, y_pred_cat.size(-1))

                loss_cat = F.kl_div(probabilities_pred_flat.log(), probabilities_true_flat, reduction='batchmean')


            if args.multi_loss_weight:
                if args.reg_enabled:
                    if args.loss_time_enabled and args.loss_cat_enabled and args.loss_geo_enabled:
                        loss = precision1 * loss_poi + task_weight1 \
                               + precision2 * loss_time + task_weight2 \
                               + precision3 * loss_cat + task_weight3 \
                               + precision4 * loss_geo + task_weight4 \
                           + args.reg_weight * l2_regularization.item()
                    elif not args.loss_time_enabled and args.loss_cat_enabled and args.loss_geo_enabled:
                        loss = precision1 * loss_poi + task_weight1 \
                               + precision3 * loss_cat + task_weight3 \
                               + precision4 * loss_geo + task_weight4 \
                           + args.reg_weight * l2_regularization.item()
                    elif args.loss_time_enabled and not args.loss_cat_enabled and args.loss_geo_enabled:
                        loss = precision1 * loss_poi + task_weight1 \
                               + precision2 * loss_time + task_weight2 \
                               + precision4 * loss_geo + task_weight4 \
                           + args.reg_weight * l2_regularization.item()
                    elif args.loss_time_enabled and args.loss_cat_enabled and not args.loss_geo_enabled:
                        loss = precision1 * loss_poi + task_weight1 \
                               + precision2 * loss_time + task_weight2 \
                               + precision3 * loss_cat + task_weight3 \
                           + args.reg_weight * l2_regularization.item()

                else:
                    if args.loss_time_enabled and args.loss_cat_enabled and args.loss_geo_enabled:
                        loss = precision1 * loss_poi + task_weight1 \
                               + precision2 * loss_time + task_weight2 \
                               + precision3 * loss_cat + task_weight3 \
                               + precision4 * loss_geo + task_weight4
                    elif not args.loss_time_enabled and args.loss_cat_enabled and args.loss_geo_enabled:
                        loss = precision1 * loss_poi + task_weight1 \
                               + precision3 * loss_cat + task_weight3 \
                               + precision4 * loss_geo + task_weight4
                    elif not args.loss_time_enabled and not args.loss_cat_enabled and args.loss_geo_enabled:
                        loss = precision1 * loss_poi + task_weight1 \
                               + precision4 * loss_geo + task_weight4
                    elif not args.loss_time_enabled and args.loss_cat_enabled and not args.loss_geo_enabled:
                        loss = precision1 * loss_poi + task_weight1 \
                               + precision3 * loss_cat + task_weight3
                    elif args.loss_time_enabled and not args.loss_cat_enabled and args.loss_geo_enabled:
                        loss = precision1 * loss_poi + task_weight1 \
                               + precision2 * loss_time + task_weight2 \
                               + precision4 * loss_geo + task_weight4
                    elif args.loss_time_enabled and args.loss_cat_enabled and not args.loss_geo_enabled:
                        loss = precision1 * loss_poi + task_weight1 \
                               + precision2 * loss_time + task_weight2 \
                               + precision3 * loss_cat + task_weight3
            else:
                if args.use_time and args.geo_graph_enabled:
                    loss = loss_poi + loss_time * args.time_loss_weight + loss_cat * args.cat_loss_weight + loss_geo * args.geo_loss_weight
                else:
                    loss = loss_poi + loss_time * args.time_loss_weight + loss_cat * args.cat_loss_weight

            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0 
            mAP20 = 0
            ndcg1 = 0
            ndcg5 = 0
            ndcg10 = 0
            ndcg20 = 0

            mrr = 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi_adjusted.detach().cpu().numpy()
            if args.use_time:
                batch_pred_times = y_pred_time.detach().cpu().numpy()
            batch_pred_cats = y_pred_cat.detach().cpu().numpy()
            if args.geo_graph_enabled:
                batch_pred_geos = y_pred_geo.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len] 
                pred_pois = pred_pois[:seq_len, :]  
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                ndcg1 += ndcg_last_timestep(label_pois, pred_pois, k=1)
                ndcg5 += ndcg_last_timestep(label_pois, pred_pois, k=5)
                ndcg10 += ndcg_last_timestep(label_pois, pred_pois, k=10)
                ndcg20 += ndcg_last_timestep(label_pois, pred_pois, k=20)
                
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
            val_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            val_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            val_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            val_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            val_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            val_batches_ndcg1_list.append(ndcg1 / len(batch_label_pois))
            val_batches_ndcg5_list.append(ndcg5 / len(batch_label_pois))
            val_batches_ndcg10_list.append(ndcg10 / len(batch_label_pois))
            val_batches_ndcg20_list.append(ndcg20 / len(batch_label_pois))
            
            val_batches_mrr_list.append(mrr / len(batch_label_pois))
            val_batches_loss_list.append(loss.detach().cpu().numpy())
            val_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            if args.use_time:
                val_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
            val_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())
            if args.geo_graph_enabled:
                val_batches_geo_loss_list.append(loss_geo.detach().cpu().numpy())



            # Report validation progress
            if (vb_idx % (args.batch * 2)) == 0:
                sample_idx = 0
                batch_pred_pois_wo_attn = y_pred_poi.detach().cpu().numpy()
                if args.use_time and args.geo_graph_enabled:
                    logging.info(f'Epoch:{epoch}, batch:{vb_idx}, '
                             f'val_batch_loss:{loss.item():.2f}, '
                             f'val_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                             f'val_move_loss:{np.mean(val_batches_loss_list):.2f} \n'
                             f'val_move_poi_loss:{np.mean(val_batches_poi_loss_list):.2f} \n'
                             f'val_move_time_loss:{np.mean(val_batches_time_loss_list):.2f} \n'
                             f'val_move_geo_loss:{np.mean(val_batches_geo_loss_list):.2f} \n'
                             f'val_move_top1_acc:{np.mean(val_batches_top1_acc_list):.4f} \n'
                             f'val_move_top5_acc:{np.mean(val_batches_top5_acc_list):.4f} \n'
                             f'val_move_top10_acc:{np.mean(val_batches_top10_acc_list):.4f} \n'
                             f'val_move_top20_acc:{np.mean(val_batches_top20_acc_list):.4f} \n'
                             f'val_move_mAP20:{np.mean(val_batches_mAP20_list):.4f} \n'
                             f'val_move_ndcg1:{np.mean(val_batches_ndcg1_list):.4f} \n'
                             f'val_move_ndcg5:{np.mean(val_batches_ndcg5_list):.4f} \n'
                             f'val_move_ndcg10:{np.mean(val_batches_ndcg10_list):.4f} \n'
                             f'val_move_ndcg20:{np.mean(val_batches_ndcg20_list):.4f} \n'
                             
                             f'val_move_MRR:{np.mean(val_batches_mrr_list):.4f} \n'
                             f'traj_id:{batch[sample_idx][0]}\n'
                             f'input_seq:{batch[sample_idx][1]}\n'
                             f'label_seq:{batch[sample_idx][2]}\n'
                             f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_cat:{[poi_idx2cat_idx_dict[each[0]] for each in batch[sample_idx][2]]}\n'
                             f'pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_time:{list(batch_seq_labels_time[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n' 
                             f'pred_seq_time:{list(np.squeeze(batch_pred_times)[sample_idx][:batch_seq_lens[sample_idx]])} \n' +
                             '=' * 100)
                else:
                    logging.info(f'Epoch:{epoch}, batch:{vb_idx}, '
                                 f'val_batch_loss:{loss.item():.2f}, '
                                 f'val_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                                 f'val_move_loss:{np.mean(val_batches_loss_list):.2f} \n'
                                 f'val_move_poi_loss:{np.mean(val_batches_poi_loss_list):.2f} \n'
                                 f'val_move_top1_acc:{np.mean(val_batches_top1_acc_list):.4f} \n'
                                 f'val_move_top5_acc:{np.mean(val_batches_top5_acc_list):.4f} \n'
                                 f'val_move_top10_acc:{np.mean(val_batches_top10_acc_list):.4f} \n'
                                 f'val_move_top20_acc:{np.mean(val_batches_top20_acc_list):.4f} \n'
                                 f'val_move_mAP20:{np.mean(val_batches_mAP20_list):.4f} \n'
                                 f'val_move_ndcg1:{np.mean(val_batches_ndcg1_list):.4f} \n'
                                 f'val_move_ndcg5:{np.mean(val_batches_ndcg5_list):.4f} \n'
                                 f'val_move_ndcg10:{np.mean(val_batches_ndcg10_list):.4f} \n'
                                 f'val_move_ndcg20:{np.mean(val_batches_ndcg20_list):.4f} \n'
                                 
                                 f'val_move_MRR:{np.mean(val_batches_mrr_list):.4f} \n'
                                 f'traj_id:{batch[sample_idx][0]}\n'
                                 f'input_seq:{batch[sample_idx][1]}\n'
                                 f'label_seq:{batch[sample_idx][2]}\n'
                                 f'label_seq_cat:{[poi_idx2cat_idx_dict[each[0]] for each in batch[sample_idx][2]]}\n'
                                 f'pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                                 f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                                 f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n' +
                                 '=' * 100)
        # valid end --------------------------------------------------------------------------------------------------------

        # Calculate epoch metrics
        epoch_train_top1_acc = np.mean(train_batches_top1_acc_list)
        epoch_train_top5_acc = np.mean(train_batches_top5_acc_list)
        epoch_train_top10_acc = np.mean(train_batches_top10_acc_list)
        epoch_train_top20_acc = np.mean(train_batches_top20_acc_list)
        epoch_train_mAP20 = np.mean(train_batches_mAP20_list)
        epoch_train_ndcg1 = np.mean(train_batches_ndcg1_list)
        epoch_train_ndcg5 = np.mean(train_batches_ndcg5_list)
        epoch_train_ndcg10 = np.mean(train_batches_ndcg10_list)
        epoch_train_ndcg20 = np.mean(train_batches_ndcg20_list)
        
        epoch_train_mrr = np.mean(train_batches_mrr_list)
        epoch_train_loss = np.mean(train_batches_loss_list)
        epoch_train_poi_loss = np.mean(train_batches_poi_loss_list)
        if args.use_time:
            epoch_train_time_loss = np.mean(train_batches_time_loss_list)
        if args.geo_graph_enabled:
            epoch_train_geo_loss = np.mean(train_batches_geo_loss_list)
        epoch_train_cat_loss = np.mean(train_batches_cat_loss_list)
        epoch_val_top1_acc = np.mean(val_batches_top1_acc_list)
        epoch_val_top5_acc = np.mean(val_batches_top5_acc_list)
        epoch_val_top10_acc = np.mean(val_batches_top10_acc_list)
        epoch_val_top20_acc = np.mean(val_batches_top20_acc_list)
        epoch_val_mAP20 = np.mean(val_batches_mAP20_list)
        epoch_val_ndcg1 = np.mean(val_batches_ndcg1_list)
        epoch_val_ndcg5 = np.mean(val_batches_ndcg5_list)
        epoch_val_ndcg10 = np.mean(val_batches_ndcg10_list)
        epoch_val_ndcg20 = np.mean(val_batches_ndcg20_list)
        epoch_val_mrr = np.mean(val_batches_mrr_list)
        epoch_val_loss = np.mean(val_batches_loss_list)
        epoch_val_poi_loss = np.mean(val_batches_poi_loss_list)
        if args.use_time:
            epoch_val_time_loss = np.mean(val_batches_time_loss_list)
        if args.geo_graph_enabled:
            epoch_val_geo_loss = np.mean(val_batches_geo_loss_list)
        epoch_val_cat_loss = np.mean(val_batches_cat_loss_list)

        # Save metrics to list
        train_epochs_loss_list.append(epoch_train_loss)
        train_epochs_poi_loss_list.append(epoch_train_poi_loss)
        if args.use_time:
            train_epochs_time_loss_list.append(epoch_train_time_loss)
        if args.geo_graph_enabled:
            train_epochs_geo_loss_list.append(epoch_train_geo_loss)
        train_epochs_cat_loss_list.append(epoch_train_cat_loss)
        train_epochs_top1_acc_list.append(epoch_train_top1_acc)
        train_epochs_top5_acc_list.append(epoch_train_top5_acc)
        train_epochs_top10_acc_list.append(epoch_train_top10_acc)
        train_epochs_top20_acc_list.append(epoch_train_top20_acc)
        train_epochs_mAP20_list.append(epoch_train_mAP20)
        train_epochs_ndcg1_list.append(epoch_train_ndcg1)
        train_epochs_ndcg5_list.append(epoch_train_ndcg5)
        train_epochs_ndcg10_list.append(epoch_train_ndcg10)
        train_epochs_ndcg20_list.append(epoch_train_ndcg20)
        
        train_epochs_mrr_list.append(epoch_train_mrr)
        val_epochs_loss_list.append(epoch_val_loss)
        val_epochs_poi_loss_list.append(epoch_val_poi_loss)
        if args.use_time:
            val_epochs_time_loss_list.append(epoch_val_time_loss)
        if args.geo_graph_enabled:
            val_epochs_geo_loss_list.append(epoch_val_geo_loss)
        val_epochs_cat_loss_list.append(epoch_val_cat_loss)
        val_epochs_top1_acc_list.append(epoch_val_top1_acc)
        val_epochs_top5_acc_list.append(epoch_val_top5_acc)
        val_epochs_top10_acc_list.append(epoch_val_top10_acc)
        val_epochs_top20_acc_list.append(epoch_val_top20_acc)
        val_epochs_mAP20_list.append(epoch_val_mAP20)
        val_epochs_ndcg1_list.append(epoch_val_ndcg1)
        val_epochs_ndcg5_list.append(epoch_val_ndcg5)
        val_epochs_ndcg10_list.append(epoch_val_ndcg10)
        val_epochs_ndcg20_list.append(epoch_val_ndcg20)
        
        val_epochs_mrr_list.append(epoch_val_mrr)
        if args.multi_loss_weight:
            val_epochs_task_weight1_list.append(task_weight1.item())
            val_epochs_task_weight2_list.append(task_weight2.item())
            val_epochs_task_weight3_list.append(task_weight3.item())
            val_epochs_task_weight4_list.append(task_weight4.item())

        # Monitor loss and score
        monitor_loss = epoch_val_loss
        monitor_score = np.mean(epoch_val_top1_acc * 4 + epoch_val_top20_acc)

        # early stopping
        if epoch_val_top20_acc > best_val_top20_acc:
            best_val_loss = epoch_val_loss
            best_val_top1_acc = epoch_val_top1_acc
            best_val_top5_acc = epoch_val_top5_acc
            best_val_top10_acc = epoch_val_top10_acc
            best_val_top20_acc = epoch_val_top20_acc
            best_val_mAP20 = epoch_val_mAP20
            best_val_ndcg1 = epoch_val_ndcg1
            best_val_ndcg5 = epoch_val_ndcg5
            best_val_ndcg10 = epoch_val_ndcg10
            best_val_ndcg20 = epoch_val_ndcg20
            
            best_val_mrr = epoch_val_mrr
            best_val_epoch = epoch

        now_score = epoch_val_top20_acc + epoch_val_ndcg20
        previous_score = previous_val_top20_acc + previous_val_ndcg20
        if now_score == previous_score:
            patience_times += 1
        else:
            patience_times = 0
        if patience_times >= 5:
            logging.info(f"Early stopping at epoch {epoch}")
            break

        previous_val_top20_acc = epoch_val_top20_acc
        previous_val_ndcg20 = epoch_val_ndcg20

        # Learning rate schuduler
        lr_scheduler.step(monitor_loss)

        # Print epoch results
        if args.use_time and args.geo_graph_enabled:
            logging.info(f"Epoch {epoch}/{args.epochs}\n"
                     f"train_loss:{epoch_train_loss:.4f}, "
                     f"train_poi_loss:{epoch_train_poi_loss:.4f}, "
                     f"train_time_loss:{epoch_train_time_loss:.4f}, "
                     f"train_geo_loss:{epoch_train_geo_loss:.4f}, "
                     f"train_cat_loss:{epoch_train_cat_loss:.4f}, "
                     f"train_top1_acc:{epoch_train_top1_acc:.4f}, "
                     f"train_top5_acc:{epoch_train_top5_acc:.4f}, "
                     f"train_top10_acc:{epoch_train_top10_acc:.4f}, "
                     f"train_top20_acc:{epoch_train_top20_acc:.4f}, "
                     f"train_mAP20:{epoch_train_mAP20:.4f}, "
                     f"train_ndcg1:{epoch_train_ndcg1:.4f}, "
                     f"train_ndcg5:{epoch_train_ndcg5:.4f}, "
                     f"train_ndcg10:{epoch_train_ndcg10:.4f}, "
                     f"train_ndcg20:{epoch_train_ndcg20:.4f}, "
                     
                     f"train_mrr:{epoch_train_mrr:.4f}\n"
                     f"val_loss: {epoch_val_loss:.4f}, "
                     f"val_poi_loss: {epoch_val_poi_loss:.4f}, "
                     f"val_time_loss: {epoch_val_time_loss:.4f}, "
                     f"val_geo_loss: {epoch_val_geo_loss:.4f}, "
                     f"val_cat_loss: {epoch_val_cat_loss:.4f}, "
                     f"val_top1_acc:{epoch_val_top1_acc:.4f}, "
                     f"val_top5_acc:{epoch_val_top5_acc:.4f}, "
                     f"val_top10_acc:{epoch_val_top10_acc:.4f}, "
                     f"val_top20_acc:{epoch_val_top20_acc:.4f}, "
                     f"val_mAP20:{epoch_val_mAP20:.4f}, "
                     f"val_ndcg1:{epoch_val_ndcg1:.4f}, "
                     f"val_ndcg5:{epoch_val_ndcg5:.4f}, "
                     f"val_ndcg10:{epoch_val_ndcg10:.4f}, "
                     f"val_ndcg20:{epoch_val_ndcg20:.4f}, "
                     
                     f"val_mrr:{epoch_val_mrr:.4f}")
        else:
            logging.info(f"Epoch {epoch}/{args.epochs}\n"
                         f"train_loss:{epoch_train_loss:.4f}, "
                         f"train_poi_loss:{epoch_train_poi_loss:.4f}, "
                         f"train_cat_loss:{epoch_train_cat_loss:.4f}, "
                         f"train_top1_acc:{epoch_train_top1_acc:.4f}, "
                         f"train_top5_acc:{epoch_train_top5_acc:.4f}, "
                         f"train_top10_acc:{epoch_train_top10_acc:.4f}, "
                         f"train_top20_acc:{epoch_train_top20_acc:.4f}, "
                         f"train_mAP20:{epoch_train_mAP20:.4f}, "
                         f"train_ndcg1:{epoch_train_ndcg1:.4f}, "
                         f"train_ndcg5:{epoch_train_ndcg5:.4f}, "
                         f"train_ndcg10:{epoch_train_ndcg10:.4f}, "
                         f"train_ndcg20:{epoch_train_ndcg20:.4f}, "
                         f"train_mrr:{epoch_train_mrr:.4f}\n"
                         f"val_loss: {epoch_val_loss:.4f}, "
                         f"val_poi_loss: {epoch_val_poi_loss:.4f}, "
                         f"val_cat_loss: {epoch_val_cat_loss:.4f}, "
                         f"val_top1_acc:{epoch_val_top1_acc:.4f}, "
                         f"val_top5_acc:{epoch_val_top5_acc:.4f}, "
                         f"val_top10_acc:{epoch_val_top10_acc:.4f}, "
                         f"val_top20_acc:{epoch_val_top20_acc:.4f}, "
                         f"val_mAP20:{epoch_val_mAP20:.4f}, "
                         f"val_ndcg1:{epoch_val_ndcg1:.4f}, "
                         f"val_ndcg5:{epoch_val_ndcg5:.4f}, "
                         f"val_ndcg10:{epoch_val_ndcg10:.4f}, "
                         f"val_ndcg20:{epoch_val_ndcg20:.4f}, "
                         f"val_mrr:{epoch_val_mrr:.4f}")

        # Save poi and user embeddings
        if args.save_embeds:
            embeddings_save_dir = os.path.join(args.save_dir, 'embeddings')
            if not os.path.exists(embeddings_save_dir): os.makedirs(embeddings_save_dir)
            # Save best epoch embeddings
            if monitor_score >= max_val_score:
                # Save poi embeddings
                poi_embeddings = poi_embed_model(X, A).detach().cpu().numpy()
                poi_embedding_list = []
                for poi_idx in range(len(poi_id2idx_dict)):
                    poi_embedding = poi_embeddings[poi_idx]
                    poi_embedding_list.append(poi_embedding)
                save_poi_embeddings = np.array(poi_embedding_list)
                np.save(os.path.join(embeddings_save_dir, 'saved_poi_embeddings'), save_poi_embeddings)
                # Save user embeddings
                user_embedding_list = []
                for user_idx in range(len(user_id2idx_dict)):
                    input = torch.LongTensor([user_idx]).to(device=args.device)
                    user_embedding = user_embed_model(input).detach().cpu().numpy().flatten()
                    user_embedding_list.append(user_embedding)
                user_embeddings = np.array(user_embedding_list)
                np.save(os.path.join(embeddings_save_dir, 'saved_user_embeddings'), user_embeddings)
                # Save cat embeddings
                cat_embedding_list = []
                for cat_idx in range(len(cat_id2idx_dict)):
                    input = torch.LongTensor([cat_idx]).to(device=args.device)
                    cat_embedding = cat_embed_model(input).detach().cpu().numpy().flatten()
                    cat_embedding_list.append(cat_embedding)
                cat_embeddings = np.array(cat_embedding_list)
                np.save(os.path.join(embeddings_save_dir, 'saved_cat_embeddings'), cat_embeddings)
                # Save time embeddings
                if args.use_time:
                    time_embedding_list = []
                    for time_idx in range(args.time_units):
                        input = torch.FloatTensor([time_idx]).to(device=args.device)
                        time_embedding = time_embed_model(input).detach().cpu().numpy().flatten()
                        time_embedding_list.append(time_embedding)
                    time_embeddings = np.array(time_embedding_list)
                    np.save(os.path.join(embeddings_save_dir, 'saved_time_embeddings'), time_embeddings)

        # Save model state dict
        if args.save_weights:
            state_dict = {
                'epoch': epoch,
                'poi_embed_state_dict': poi_embed_model.state_dict(),
                # 'node_attn_state_dict': node_attn_model.state_dict(),
                'user_embed_state_dict': user_embed_model.state_dict(),
                'time_embed_state_dict': time_embed_model.state_dict(),
                'cat_embed_state_dict': cat_embed_model.state_dict(),
                'seq_model_state_dict': seq_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'user_id2idx_dict': user_id2idx_dict,
                'poi_id2idx_dict': poi_id2idx_dict,
                'cat_id2idx_dict': cat_id2idx_dict,
                'poi_idx2cat_idx_dict': poi_idx2cat_idx_dict,
                # 'node_attn_map': node_attn_model(X, A),
                'args': args,
                'epoch_train_metrics': {
                    'epoch_train_loss': epoch_train_loss,
                    'epoch_train_poi_loss': epoch_train_poi_loss,
                    'epoch_train_time_loss': epoch_train_time_loss,
                    'epoch_train_cat_loss': epoch_train_cat_loss,
                    'epoch_train_top1_acc': epoch_train_top1_acc,
                    'epoch_train_top5_acc': epoch_train_top5_acc,
                    'epoch_train_top10_acc': epoch_train_top10_acc,
                    'epoch_train_top20_acc': epoch_train_top20_acc,
                    'epoch_train_mAP20': epoch_train_mAP20,
                    'epoch_train_ndcg1': epoch_train_ndcg1,
                    'epoch_train_ndcg5': epoch_train_ndcg5,
                    'epoch_train_ndcg10': epoch_train_ndcg10,
                    'epoch_train_ndcg20': epoch_train_ndcg20,
                    
                    'epoch_train_mrr': epoch_train_mrr
                },
                'epoch_val_metrics': {
                    'epoch_val_loss': epoch_val_loss,
                    'epoch_val_poi_loss': epoch_val_poi_loss,
                    'epoch_val_time_loss': epoch_val_time_loss,
                    'epoch_val_cat_loss': epoch_val_cat_loss,
                    'epoch_val_top1_acc': epoch_val_top1_acc,
                    'epoch_val_top5_acc': epoch_val_top5_acc,
                    'epoch_val_top10_acc': epoch_val_top10_acc,
                    'epoch_val_top20_acc': epoch_val_top20_acc,
                    'epoch_val_mAP20': epoch_val_mAP20,
                    'epoch_val_ndcg1': epoch_val_ndcg1,
                    'epoch_val_ndcg5': epoch_val_ndcg5,
                    'epoch_val_ndcg10': epoch_val_ndcg10,
                    'epoch_val_ndcg20': epoch_val_ndcg20,
                    
                    'epoch_val_mrr': epoch_val_mrr
                }
            }
            model_save_dir = os.path.join(args.save_dir, 'checkpoints')
            # Save best val score epoch
            if monitor_score >= max_val_score:
                if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
                torch.save(state_dict, rf"{model_save_dir}/best_epoch.state.pt")
                with open(rf"{model_save_dir}/best_epoch.txt", 'w') as f:
                    print(state_dict['epoch_val_metrics'], file=f)
                max_val_score = monitor_score

        # Save train/val metrics for plotting purpose
        with open(os.path.join(args.save_dir, 'metrics-train.txt'), "w") as f:
            print(f'train_epochs_loss_list={[float(f"{each:.4f}") for each in train_epochs_loss_list]}', file=f)
            print(f'train_epochs_poi_loss_list={[float(f"{each:.4f}") for each in train_epochs_poi_loss_list]}', file=f)
            if args.use_time:
                print(f'train_epochs_time_loss_list={[float(f"{each:.4f}") for each in train_epochs_time_loss_list]}',
                  file=f)
            print(f'train_epochs_cat_loss_list={[float(f"{each:.4f}") for each in train_epochs_cat_loss_list]}', file=f)
            print(f'train_epochs_top1_acc_list={[float(f"{each:.4f}") for each in train_epochs_top1_acc_list]}', file=f)
            print(f'train_epochs_top5_acc_list={[float(f"{each:.4f}") for each in train_epochs_top5_acc_list]}', file=f)
            print(f'train_epochs_top10_acc_list={[float(f"{each:.4f}") for each in train_epochs_top10_acc_list]}',
                  file=f)
            print(f'train_epochs_top20_acc_list={[float(f"{each:.4f}") for each in train_epochs_top20_acc_list]}',
                  file=f)
            print(f'train_epochs_mAP20_list={[float(f"{each:.4f}") for each in train_epochs_mAP20_list]}', file=f)
            print(f'train_epochs_ndcg1_list={[float(f"{each:.4f}") for each in train_epochs_ndcg1_list]}', file=f)
            print(f'train_epochs_ndcg5_list={[float(f"{each:.4f}") for each in train_epochs_ndcg5_list]}', file=f)
            print(f'train_epochs_ndcg10_list={[float(f"{each:.4f}") for each in train_epochs_ndcg10_list]}', file=f)
            print(f'train_epochs_ndcg20_list={[float(f"{each:.4f}") for each in train_epochs_ndcg20_list]}', file=f)
            
            print(f'train_epochs_mrr_list={[float(f"{each:.4f}") for each in train_epochs_mrr_list]}', file=f)
        with open(os.path.join(args.save_dir, 'metrics-val.txt'), "w") as f:
            print(f'val_epochs_loss_list={[float(f"{each:.4f}") for each in val_epochs_loss_list]}', file=f)
            print(f'val_epochs_poi_loss_list={[float(f"{each:.4f}") for each in val_epochs_poi_loss_list]}', file=f)
            if args.use_time:
                print(f'val_epochs_time_loss_list={[float(f"{each:.4f}") for each in val_epochs_time_loss_list]}', file=f)
            print(f'val_epochs_cat_loss_list={[float(f"{each:.4f}") for each in val_epochs_cat_loss_list]}', file=f)
            print(f'val_epochs_geo_loss_list={[float(f"{each:.4f}") for each in val_epochs_geo_loss_list]}', file=f)
            print(f'val_epochs_top1_acc_list={[float(f"{each:.4f}") for each in val_epochs_top1_acc_list]}', file=f)
            print(f'val_epochs_top5_acc_list={[float(f"{each:.4f}") for each in val_epochs_top5_acc_list]}', file=f)
            print(f'val_epochs_top10_acc_list={[float(f"{each:.4f}") for each in val_epochs_top10_acc_list]}', file=f)
            print(f'val_epochs_top20_acc_list={[float(f"{each:.4f}") for each in val_epochs_top20_acc_list]}', file=f)
            print(f'val_epochs_mAP20_list={[float(f"{each:.4f}") for each in val_epochs_mAP20_list]}', file=f)
            print(f'val_epochs_ndcg1_list={[float(f"{each:.4f}") for each in val_epochs_ndcg1_list]}', file=f)
            print(f'val_epochs_ndcg5_list={[float(f"{each:.4f}") for each in val_epochs_ndcg5_list]}', file=f)
            print(f'val_epochs_ndcg10_list={[float(f"{each:.4f}") for each in val_epochs_ndcg10_list]}', file=f)
            print(f'val_epochs_ndcg20_list={[float(f"{each:.4f}") for each in val_epochs_ndcg20_list]}', file=f)
            
            print(f'val_epochs_mrr_list={[float(f"{each:.4f}") for each in val_epochs_mrr_list]}', file=f)
            if args.multi_loss_weight:
                print(f'val_epochs_task_weight1_list={[float(f"{each:.4f}") for each in val_epochs_task_weight1_list]}', file=f)
                print(f'val_epochs_task_weight2_list={[float(f"{each:.4f}") for each in val_epochs_task_weight2_list]}', file=f)
                print(f'val_epochs_task_weight3_list={[float(f"{each:.4f}") for each in val_epochs_task_weight3_list]}', file=f)
                print(f'val_epochs_task_weight4_list={[float(f"{each:.4f}") for each in val_epochs_task_weight4_list]}', file=f)

            print(f'last_top1_acc={epoch_val_top1_acc}', file=f)
            print(f'last_top5_acc={epoch_val_top5_acc}', file=f)
            print(f'last_top10_acc={epoch_val_top10_acc}', file=f)
            print(f'last_top20_acc={epoch_val_top20_acc}', file=f)
            print(f'last_mAP20={epoch_val_mAP20}', file=f)
            print(f'last_ndcg1={epoch_val_ndcg1}', file=f)
            print(f'last_ndcg5={epoch_val_ndcg5}', file=f)
            print(f'last_ndcg10={epoch_val_ndcg10}', file=f)
            print(f'last_ndcg20={epoch_val_ndcg20}', file=f)
            
            print(f'last_mrr={epoch_val_mrr}', file=f)

            print(f'best_top1_acc={best_val_top1_acc}', file=f)
            print(f'best_top5_acc={best_val_top5_acc}', file=f)
            print(f'best_top10_acc={best_val_top10_acc}', file=f)
            print(f'best_top20_acc={best_val_top20_acc}', file=f)
            print(f'best_mAP20={best_val_mAP20}', file=f)
            print(f'best_ndcg1={best_val_ndcg1}', file=f)
            print(f'best_ndcg5={best_val_ndcg5}', file=f)
            print(f'best_ndcg10={best_val_ndcg10}', file=f)
            print(f'best_ndcg20={best_val_ndcg20}', file=f)
            
            print(f'best_mrr={best_val_mrr}', file=f)
            print(f'best_epoch={best_val_epoch}', file=f)
            print(f'sum_of_train_epochs={epoch}', file=f)


if __name__ == '__main__':

    args = parameter_parser()
    args.feature1 = 'checkin_cnt'
    if args.cat_encoder == "OneHot":
        args.feature2 = 'poi_catid'
    elif args.cat_encoder == "SentenceTransformer":
        args.feature2 = 'poi_catname'
    args.feature3 = 'latitude'
    args.feature4 = 'longitude'
    train(args)

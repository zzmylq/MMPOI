"""Parsing the parameters."""
import argparse
import os

import torch

if torch.cuda.is_available():
    device = torch.device('cuda:5')
else:
    device = torch.device('cpu')

def parameter_parser():
    parser = argparse.ArgumentParser(description="Run GETNext.")

    '''————————————————major parameters————————————————'''
    parser.add_argument('--seed',
                        type=int,
                        default=618,
                        help='Random seed')
    parser.add_argument('--knn-k',
                        type=int,
                        default=5,
                        help='knn k')
    parser.add_argument('--A-plus-weights',
                        type=list,
                        default=[0.4, 0.6],
                        help='A, multimodal')
    parser.add_argument('--gcn-dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate for gcn')
    parser.add_argument('--transformer-dropout',
                        type=float,
                        default=0.1,
                        help='Dropout rate for transformer')
    parser.add_argument('--batch',
                        type=int,
                        default=64,
                        help='Batch size.')

    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='Initial learning rate.')

    '''————————————————dataset————————————————'''

    parser.add_argument('--data-adj-mtx',
                        type=str,
                        default='dataset/NYC/graph_A.csv',
                        help='Graph adjacent path')
    parser.add_argument('--data-node-feats',
                        type=str,
                        default='dataset/NYC/graph_X.csv',
                        help='Graph node features path')
    parser.add_argument('--data-train',
                        type=str,
                        default='dataset/NYC/NYC_train.csv',
                        help='Training data path')
    parser.add_argument('--data-val',
                        type=str,
                        default='dataset/NYC/NYC_test.csv',
                        help='Validation data path')
    parser.add_argument('--geo-adj-mtx',
                        type=str,
                        default='dataset/NYC/geo_graph_A.csv',
                        help='Geo graph adjacent path')
    parser.add_argument('--geo-node-feats',
                        type=str,
                        default='dataset/NYC/geo_graph_X.csv',
                        help='Geo graph node features path')

    '''————————————————ablation study————————————————'''

    parser.add_argument('--use-A-plus',
                        type=bool,
                        default=True,
                        help='whether to use A+')

    parser.add_argument('--loss-geo-enabled',
                        type=bool,
                        default=True,
                        help='add geo loss to the model')

    parser.add_argument('--loss-cat-enabled',
                        type=bool,
                        default=False,
                        help='add cat loss to the model')


    parser.add_argument('--cat-encoder',
                        type=str,
                        default='SentenceTransformer',
                        help='Categorical encoder')


    parser.add_argument('--loss-time-enabled',
                        type=bool,
                        default=False,
                        help='add time loss to the model')


    '''——————————————————————————————————————————————'''

    parser.add_argument('--reg-enabled',
                        type=bool,
                        default=False,
                        help='whether to use l2 regularization')

    parser.add_argument('--reg-weight',
                        type=float,
                        default=0.00001,
                        help='Scale factor for the reg loss term')

    parser.add_argument('--device',
                        type=str,
                        default=device,
                        help='')
    # Data

    parser.add_argument('--multimodal-enabled',
                        type=bool,
                        default=True,
                        help='wheather to use multimodal data')


    parser.add_argument('--cat-loss-type',
                        type=str,
                        default='embedding_loss',
                        help='whether to use multi loss weight')

    parser.add_argument('--fusion-method',
                        type=str,
                        default="fusion",
                        help='which fusion method to use')

    parser.add_argument('--use-time',
                        type=bool,
                        default=True,
                        help='Whether to use time feature')

    parser.add_argument('--use-cat',
                        type=bool,
                        default=False,
                        help='Whether to use cat embedding to fusion')

    parser.add_argument('--cat-type',
                        type=str,
                        default='SentenceTransformer',
                        help='Concat-aware Categorical encoder, only used by use_cat=True')

    parser.add_argument('--NLP-embedding-dim',
                        type=int,
                        default=768,
                        help='The embedding dim of NLP encoder, only used by cat_type=SentenceTransformer')

    parser.add_argument('--cat-freeze',
                        type=bool,
                        default=True,
                        help='Concat-aware Categorical freeze, only used by cat_type=SentenceTransformer')

    parser.add_argument('--adjust-enabled',
                        type=bool,
                        default=False,
                        help='adjust_pred_prob_by_graph enabled')

    parser.add_argument('--geo-graph-enabled',
                        type=bool,
                        default=True,
                        help='add geo graph to the model')
    parser.add_argument('--geo-loss-weight',
                        type=float,
                        default=1,
                        help='Scale factor for the geo loss term')

    parser.add_argument('--cat-loss-weight',
                        type=float,
                        default=1,
                        help='Scale factor for the cat loss term')

    parser.add_argument('--multi-loss-weight',
                        type=bool,
                        default=True,
                        help='whether to use multi loss weight')

    parser.add_argument('--task-weight-negative',
                        type=bool,
                        default=True,
                        help='whether to use E^(-x) to weight the task loss')




    parser.add_argument('--multimodal-graph-weights',
                        type=list,
                        default=[1, 1, 1, 1],
                        help='image, meta, review, review_summary')



    parser.add_argument('--short-traj-thres',
                        type=int,
                        default=2,
                        help='Remove over-short trajectory')
    parser.add_argument('--time-units',
                        type=int,
                        default=48,
                        help='Time unit is 0.5 hour, 24/0.5=48')
    parser.add_argument('--time-feature',
                        type=str,
                        default='norm_in_day_time',
                        help='The name of time feature in the data')

    # Model hyper-parameters
    parser.add_argument('--poi-embed-dim',
                        type=int,
                        default=128,
                        help='POI embedding dimensions')
    parser.add_argument('--geo-embed-dim',
                        type=int,
                        default=128,
                        help='Geo embedding dimensions')

    parser.add_argument('--multimodal-embed-dim',
                        type=int,
                        default=128,
                        help='Multimodal embedding dimensions')

    parser.add_argument('--user-embed-dim',
                        type=int,
                        default=128,
                        help='User embedding dimensions')

    parser.add_argument('--gcn-nhid',
                        type=list,
                        default=[32, 64],
                        help='List of hidden dims for gcn layers')
    parser.add_argument('--transformer-nhid',
                        type=int,
                        default=1024,
                        help='Hid dim in TransformerEncoder')
    parser.add_argument('--transformer-nlayers',
                        type=int,
                        default=2, # 2
                        help='Num of TransformerEncoderLayer')
    parser.add_argument('--transformer-nhead',
                        type=int,
                        default=2, # 2
                        help='Num of heads in multiheadattention')

    parser.add_argument('--time-embed-dim',
                        type=int,
                        default=128,
                        help='Time embedding dimensions')
    parser.add_argument('--cat-embed-dim',
                        type=int,
                        default=32,
                        help='Category embedding dimensions')
    parser.add_argument('--time-loss-weight',
                        type=int,
                        default=10,
                        help='Scale factor for the time loss term')
    parser.add_argument('--node-attn-nhid',
                        type=int,
                        default=128,
                        help='Node attn map hidden dimensions')



    parser.add_argument('--epochs',
                        type=int,
                        default=200,
                        help='Number of epochs to train.')

    parser.add_argument('--lr-scheduler-factor',
                        type=float,
                        default=0.1,
                        help='Learning rate scheduler factor')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=5e-4,
                        help='Weight decay (L2 loss on parameters).')

    # Experiment config
    parser.add_argument('--save-weights',
                        action='store_true',
                        default=False,
                        help='whether save the model')
    parser.add_argument('--save-embeds',
                        action='store_true',
                        default=False,
                        help='whether save the embeddings')
    parser.add_argument('--workers',
                        type=int,
                        default=0,
                        help='Num of workers for dataloader.')
    parser.add_argument('--project',
                        default='runs/train',
                        help='save to project/name')
    parser.add_argument('--name',
                        default='exp',
                        help='save to project/name')
    parser.add_argument('--exist-ok',
                        action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False, help='Disables CUDA training.')
    parser.add_argument('--mode',
                        type=str,
                        default='client',
                        help='python console use only')
    parser.add_argument('--port',
                        type=int,
                        default=64973,
                        help='python console use only')

    return parser.parse_args()

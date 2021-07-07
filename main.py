import math
import os
from copy import deepcopy
from scipy.linalg import svd
import numpy as np

from tqdm import tqdm
import scipy.io as scio
import scipy.sparse
from config import config

from models.HyperG import HyperG

import sklearn
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import CategoriesSampler, DataSet
from utils import get_embedding, mean_confidence_interval, setup_seed


def initial_embed(reduce, d):
    reduce = reduce.lower()
    assert reduce in ['isomap', 'itsa', 'mds', 'lle', 'se', 'pca', 'none']
    if reduce == 'isomap':
        from sklearn.manifold import Isomap
        embed = Isomap(n_components=d)
    elif reduce == 'itsa':
        from sklearn.manifold import LocallyLinearEmbedding
        embed = LocallyLinearEmbedding(n_components=d,
                                       n_neighbors=5, method='ltsa')
    elif reduce == 'mds':
        from sklearn.manifold import MDS
        embed = MDS(n_components=d, metric=False)
    elif reduce == 'lle':
        from sklearn.manifold import LocallyLinearEmbedding
        embed = LocallyLinearEmbedding(n_components=d, n_neighbors=5,eigen_solver='dense')
    elif reduce == 'se':
        from sklearn.manifold import SpectralEmbedding
        embed = SpectralEmbedding(n_components=d)
    elif reduce == 'pca':
        from sklearn.decomposition import PCA
        embed = PCA(n_components=d,random_state=0)
 
    return embed

def test(args):

    setup_seed(23)
    import warnings
    warnings.filterwarnings('ignore')
    if args.dataset == 'miniimagenet':
        num_classes = 64
    elif args.dataset == 'tieredimagenet':
        num_classes = 351
    elif args.dataset == 'cifar':
        num_classes = 64
    elif args.dataset == 'fc100':
        num_classes = 60
		
    if args.resume is not None:
        from models.resnet12 import resnet12
        model = resnet12(num_classes).to(args.device)
        state_dict = torch.load(args.resume)
        model.load_state_dict(state_dict)

        from models.r_resnet12 import r_resnet12
        r_model = r_resnet12(num_classes).to(args.device)
        r_state_dict = torch.load(args.r_resume)
        r_model.load_state_dict(r_state_dict)

    model.to(args.device)
    model.eval()
    r_model.to(args.device)
    r_model.eval()

    if args.dataset == 'miniimagenet':
        data_root = os.path.join(args.folder, '/home/wfliu/xdd_xr/LaplacianShot-master-org/LaplacianShot-master/data/')
    elif args.dataset == 'tieredimagenet':
        data_root = '/home/tieredimagenet'
    elif args.dataset == 'cifar':
        data_root = '/home/cifar'
    elif args.dataset == 'fc100':
        data_root = '/home/fc100'

    else:
        print("error!!!!!!!!!!")

    hyperG = HyperG(num_class=args.num_test_ways,step=args.step, reduce=args.embed, d=args.dim)

    dataset = DataSet(data_root, 'test', args.img_size)
    sampler = CategoriesSampler(dataset.label, args.num_batches,
                                args.num_test_ways, (args.num_shots, 15, args.unlabel))
    testloader = DataLoader(dataset, batch_sampler=sampler,
                            shuffle=False, num_workers=0, pin_memory=True)

    k = args.num_shots * args.num_test_ways
    loader = tqdm(testloader, ncols=0)
    if(args.unlabel==0):
        iterations = 22
    else:
        iterations = args.unlabel+2+5

    acc_list = [[] for _ in range(iterations)]
    acc_list_task = [[] for _ in range(iterations)]
    acc_list_softmax = [[] for _ in range(iterations)]

 
    for data, indicator in loader:
        targets = torch.arange(args.num_test_ways).repeat(args.num_shots+15+args.unlabel).long()[
            indicator[:args.num_test_ways*(args.num_shots+15+args.unlabel)] != 0]
        data = data[indicator != 0].to(args.device)

        data_r = get_embedding(r_model, data, args.device)
        data_x = get_embedding(model, data, args.device)

        if args.dim != 512:
            if args.unlabel != 0:
                data_train1 = np.concatenate((data_r[:k], data_r[k+15*args.num_test_ways:k+15*args.num_test_ways+args.unlabel*args.num_test_ways]), axis=0)
                data_train2 = np.concatenate((data_x[:k], data_x[k+15*args.num_test_ways:k+15*args.num_test_ways+args.unlabel*args.num_test_ways]), axis=0)
                data_train = np.concatenate((data_train1, data_train2), axis=0)
                embed_data = initial_embed(args.embed, args.dim)
                embed_fit = embed_data.fit(data_train)
                data_r = embed_data.transform(data_r[:k+15*args.num_test_ways+args.unlabel*args.num_test_ways])
                data_x = embed_data.transform(data_x[:k+15*args.num_test_ways+args.unlabel*args.num_test_ways])
            else:
                data_train1 = np.concatenate((data_r[:k], data_r[k:k+15*args.num_test_ways]), axis=0)
                data_train2 = np.concatenate((data_x[:k], data_x[k:k+15*args.num_test_ways]), axis=0)

                data_train = np.concatenate((data_train1, data_train2), axis=0)
                embed_data = initial_embed(args.embed, args.dim)
                embed_fit = embed_data.fit(data_train)
                data_r = embed_data.transform(data_train1)
                data_x = embed_data.transform(data_train2)
        data_r_concat = np.concatenate((data_r, data_x), axis=1)

        train_targets = targets[:k]
        test_targets = targets[k:k+15*args.num_test_ways]

        train_embeddings_task = data_r_concat[:k]
        test_embeddings_task = data_r_concat[k:k+15*args.num_test_ways]

        if args.unlabel != 0:
            unlabel_embeddings_task = data_r_concat[k+15*args.num_test_ways:k+15*args.num_test_ways+args.unlabel*args.num_test_ways]
        else:
            unlabel_embeddings_task = None

        hyperG.fit(train_embeddings_task, train_targets)
        acc = hyperG.predict(test_embeddings_task,unlabel_embeddings_task, True, test_targets,args.eta)
 
        for i in range(len(acc)):
            acc_list[i].append(acc[i])
 

    cal_accuracy(acc_list)

def cal_accuracy(acc_list_task):

    mean_list_task = []
    ci_list_task = []

    for item in acc_list_task:
        mean, ci = mean_confidence_interval(item)
        mean_list_task.append(mean)
        ci_list_task.append(ci)

    print("Test Acc Mean_task{}".format(
        ' '.join([str(i*100)[:6] for i in mean_list_task])))
    print("Test Acc ci_task{}".format(' '.join([str(i*100)[:6] for i in ci_list_task])))
 

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(args)
    if args.mode == 'test':
        test(args)
    else:
        raise NameError


if __name__ == '__main__':
    args = config()
    main(args)

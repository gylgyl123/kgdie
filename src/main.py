import argparse
import pickle
import random

import numpy as np
import torch

from data_loader import load_data
from train import train

if __name__ == '__main__':
    seed = 555
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    import os

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # # 参数设置
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
    # parser.add_argument('--dim', type=int, default=16, help='dimension of entity and relation embeddings')
    # parser.add_argument('--n_hop', type=int, default=1, help='maximum hops')
    # parser.add_argument('--n_hist', type=int, default=20, help='number of hist')
    # parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
    # parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
    # parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    # parser.add_argument('--kge_type', type=str, default='STransD', help='KGE方案')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--n_epoch', type=int, default=40, help='the number of epochs')
    # parser.add_argument('--n_memory', type=int, default=8, help='size of ripple set for each hop')
    # parser.add_argument('--item_update_mode', type=str, default='plus_transform',
    #                     help='how to update item at the end of each hop')
    # parser.add_argument('--using_all_hops', type=bool, default=True,
    #                     help='whether using outputs of all hops or just the last hop when making prediction')
    # parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use gpu')
    # args = parser.parse_args()
    #
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
    parser.add_argument('--model_name', type=str, default='kg_die', help='which model to use')
    parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
    parser.add_argument('--n_hop', type=int, default=1, help='maximum hops')
    parser.add_argument('--n_hist', type=int, default=20 , help='number of hist')
    parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
    parser.add_argument('--kge_weight', type=float, default=0.001, help='weight of the KGE term')
    parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
    parser.add_argument('--kge_type', type=str, default='STransD', help='KGE方案')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--n_memory', type=int, default=8, help='size of ripple set for each hop')
    parser.add_argument('--item_update_mode', type=str, default='concat_transform',
                        help='how to update item at the end of each hop')
    parser.add_argument('--using_all_hops', type=bool, default=True,
                        help='whether using outputs of all hops or just the last hop when making prediction')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use gpu')
    args = parser.parse_args()

    show_loss = True
    '''
        获取数据：
            train_data, eval_data, test_data：训练集，验证集，测试集，数据格式为 [uid, iid, label]
            n_entity, n_relation: 实体和关系的数量
            ripple_set：波纹集
    '''

    if os.path.exists('./data/' + args.dataset + '/data_info' + str(args.n_memory) + '.pkl'):
        with open('./data/' + args.dataset + '/data_info' + str(args.n_memory) + '.pkl', 'rb') as f:
            data_info = pickle.load(f)
    else:
        data_info = load_data(args)
        with open('./data/' + args.dataset + '/data_info' + str(args.n_memory) + '.pkl', 'wb') as f:
            pickle.dump(data_info, f)


    # print(data_info)
    # 训练
    metric_re = train(args, data_info, show_loss)

    import pandas as pd

    metric_re = pd.DataFrame(metric_re)
    # metric_re.to_csv('D:\\project\\python\\RippleTrm\\result\\amazon\\KG-DSI_with_trm_plus.csv')
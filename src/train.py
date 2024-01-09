import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import RippleNetTrm



def train(args, data_info, show_loss):
    # 数据准备
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_user = data_info[3]
    n_entity = data_info[4]
    n_relation = data_info[5]
    ripple_set = data_info[6]
    # n_entity = data_info[3]
    # n_relation = data_info[4]
    # ripple_set = data_info[5]


    # 创建模型并初始化

    if args.model_name == 'kg_die':
        model = RippleNetTrm(args, n_user, n_entity, n_relation)

    # model = RippleNetTrm(args, n_user, n_entity, n_relation)
    # model = RippleNetTrm(args, n_entity, n_relation)
    if args.use_cuda:
        model.cuda()

    # 优化器
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr,
    )

    # train_data_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
    # eval_data_loader = DataLoader(dataset=eval_data, batch_size=args.batch_size, shuffle=False)
    # test_data_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    metrics = {
        'train_auc': [],
        'train_acc': [],
        'val_auc': [],
        'val_acc': [],
        'test_auc': [],
        'test_acc': [],
    }

    metrics_name = ['train_auc', 'train_acc', 'val_auc', 'val_acc', 'test_auc', 'test_acc']

    for step in range(args.n_epoch):
        start_time = time.time()
        # 打乱数据
        idx = list(range(len(train_data)))
        random.shuffle(idx)
        # train_data = train_data[idx]
        # train_data = list(train_data.data.cpu())
        # torch.cuda.empty_cache()
        # np.random.shuffle(train_data)
        # train_data = torch.stack(train_data).to('cuda')
        # for i, data in enumerate(train_data_loader, 0):
        #     return_dict = model(
        #         data[:, 0],
        #         data[:, 1],
        #         data[:, 2: 2 + args.n_hist],
        #         data[:, -1],
        #         data[:, 2 + args.n_hist: 2 + args.n_hist + args.n_hist * args.n_memory],
        #         data[:,
        #         2 + args.n_hist + args.n_hist * args.n_memory: 2 + args.n_hist + args.n_hist * args.n_memory * 2],
        #         data[:,
        #         2 + args.n_hist + args.n_hist * args.n_memory * 2: 2 + args.n_hist + args.n_hist * args.n_memory * 3],
        #         data[:, -2]
        #     )
        #     loss = return_dict["loss"]
        #
        #     # 梯度置为0
        #     optimizer.zero_grad()
        #     # 反向传播
        #     loss.backward()
        #     optimizer.step()

        start = 0
        while start < len(train_data):
            # 先生成一个batch的训练数据，然后调用模型的forward方法，进行前向运算
            # 前向计算会输出预测结果（概率值）和损失（包括总损失，BCE损失，KGE损失，正则损失）
            data = train_data[idx[start:start + args.batch_size]][:]
            return_dict = model(
                data[:, 0],
                data[:, 1],
                data[:, 2: 2 + args.n_hist],
                data[:, -1],
                data[:, 2 + args.n_hist: 2 + args.n_hist + (args.n_hist + 1) * args.n_memory],
                data[:,
                2 + args.n_hist + (args.n_hist + 1) * args.n_memory: 2 + args.n_hist + (
                        args.n_hist + 1) * args.n_memory * 2],
                data[:,
                2 + args.n_hist + (args.n_hist + 1) * args.n_memory * 2: 2 + args.n_hist + (
                        args.n_hist + 1) * args.n_memory * 3],
                data[:, -2]
                #     )
                # *get_feed_dict(
                #     args, model, train_data, ripple_set, start, start + args.batch_size
                # )
            )
            loss = return_dict["loss"]

            # 梯度置为0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()

            # 标记自增
            start += args.batch_size

        # evaluation
        train_auc, train_acc = evaluation(args, model, train_data, ripple_set, args.batch_size)
        eval_auc, eval_acc = evaluation(args, model, eval_data, ripple_set, args.batch_size)
        test_auc, test_acc = evaluation(args, model, test_data, ripple_set, args.batch_size)

        metrics_re = [train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc]
        i = 0
        for me in metrics_name:
            metrics[me].append(metrics_re[i])
            i += 1
        print(
            'epoch %d:  %d s:  loss %.4f train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
            % (
                step, time.time() - start_time, loss.item(), train_auc, train_acc, eval_auc, eval_acc, test_auc,
                test_acc))
    return metrics


def evaluation(args, model, dataset, ripple_set, batch_size):
    auc_list = []
    acc_list = []
    model.eval()

    start = 0
    with torch.no_grad():
        while start < dataset.shape[0]:
            data = dataset[start:start + args.batch_size, :]
            auc, acc = model.evaluate(
                data[:, 0],
                data[:, 1],
                data[:, 2: 2 + args.n_hist],
                data[:, -1],
                data[:, 2 + args.n_hist: 2 + args.n_hist + (args.n_hist + 1) * args.n_memory],
                data[:,
                2 + args.n_hist + (args.n_hist + 1) * args.n_memory: 2 + args.n_hist + (
                        args.n_hist + 1) * args.n_memory * 2],
                data[:,
                2 + args.n_hist + (args.n_hist + 1) * args.n_memory * 2: 2 + args.n_hist + (
                        args.n_hist + 1) * args.n_memory * 3],
                data[:, -2]
            )
            auc_list.append(auc)
            acc_list.append(acc)
            start += args.batch_size
    model.train()
    return float(np.mean(auc_list)), float(np.mean(acc_list))

    # while start < data.shape[0]:
    #     # auc, acc = model.evaluate(*get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
    #     auc, acc = model.evaluate()
    #     auc_list.append(auc)
    #     acc_list.append(acc)
    #     start += batch_size
    # model.train()
    # return float(np.mean(auc_list)), float(np.mean(acc_list))

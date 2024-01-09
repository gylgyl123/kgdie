import collections
import os
import pickle

import numpy as np

# 加载数据
import pandas as pd
import torch


def load_data(args):
    # 加载并分割数据集，注意 `验证集` 和 `测试集` 的记录中uid不在 `train集` 中的全部被去除了
    data, n_user = load_rating(args)
    train_data, eval_data, test_data = data

    print('train samples: %d' % len(train_data))
    # train_data, eval_data, test_data = load_rating(args)
    # 构建知识图谱
    if os.path.exists('./data/' + args.dataset + '/kg' + str(args.n_memory) + '.pkl'):
        with open('./data/' + args.dataset + '/kg' + str(args.n_memory) + '.pkl', 'rb') as f:
            n_entity = pickle.load(f)
            n_relation = pickle.load(f)
            kg = pickle.load(f)
    else:
        n_entity, n_relation, kg = load_kg(args)
        with open('./data/' + args.dataset + '/kg' + str(args.n_memory) + '.pkl', 'wb') as f:
            pickle.dump(n_entity, f)
            pickle.dump(n_relation, f)
            pickle.dump(kg, f)

    if os.path.exists('./data/' + args.dataset + '/ripple_set' + str(args.n_memory) + '.pkl'):
        with open('./data/' + args.dataset + '/ripple_set' + str(args.n_memory) + '.pkl', 'rb') as f:
            ripple_set = pickle.load(f)
    else:
        # 获取波纹集
        ripple_set = get_ripple_set(args, kg)
        with open('./data/' + args.dataset + '/ripple_set' + str(args.n_memory) + '.pkl', 'wb') as f:
            pickle.dump(ripple_set, f)

    if os.path.exists('./data/' + args.dataset + '/train_data_final' + str(args.n_memory) + '.pkl'):
        with open('./data/' + args.dataset + '/train_data_final' + str(args.n_memory) + '.pkl', 'rb') as f:
            train_data_final = pickle.load(f)
    else:
        train_data_final = construct_neighbor(args, train_data, ripple_set)
        with open('./data/' + args.dataset + '/train_data_final' + str(args.n_memory) + '.pkl', 'wb') as f:
            pickle.dump(train_data_final, f)

    if os.path.exists('./data/' + args.dataset + '/eval_data_final' + str(args.n_memory) + '.pkl'):
        with open('./data/' + args.dataset + '/eval_data_final' + str(args.n_memory) + '.pkl', 'rb') as f:
            eval_data_final = pickle.load(f)
    else:
        eval_data_final = construct_neighbor(args, eval_data, ripple_set)
        with open('./data/' + args.dataset + '/eval_data_final' + str(args.n_memory) + '.pkl', 'wb') as f:
            pickle.dump(eval_data_final, f)

    if os.path.exists('./data/' + args.dataset + '/test_data_final' + str(args.n_memory) + '.pkl'):
        with open('./data/' + args.dataset + '/test_data_final' + str(args.n_memory) + '.pkl', 'rb') as f:
            test_data_final = pickle.load(f)
    else:
        test_data_final = construct_neighbor(args, test_data, ripple_set)
        with open('./data/' + args.dataset + '/test_data_final' + str(args.n_memory) + '.pkl', 'wb') as f:
            pickle.dump(test_data_final, f)

    # eval_data_final = construct_neighbor(args, eval_data, ripple_set)
    # test_data_final = construct_neighbor(args, test_data, ripple_set)
    # 返回数据
    return train_data_final, eval_data_final, test_data_final, n_user, n_entity, n_relation, ripple_set
    # return train_data, eval_data, test_data, n_entity, n_relation, ripple_set


# 读取评分数据
def load_rating(args):
    print('reading rating file ...')

    # 读取 rating 文件
    rating_file = './data/' + args.dataset + '/rating_seq.pkl'  # path

    with open(rating_file, 'rb') as f:
        uids = pickle.load(f)
        iids = pickle.load(f)
        hists = pickle.load(f)
        target = pickle.load(f)
        seq_length = pickle.load(f)

    # temp = np.concatenate([uids, hists])
    uids = np.reshape(uids, [-1, 1])
    iids = np.reshape(iids, [-1, 1])
    hists = np.reshape(hists, [-1, args.n_hist])
    target = np.reshape(target, [-1, 1])
    seq_length = np.reshape(seq_length, [-1, 1])
    rating_data = np.concatenate([uids, iids, hists, seq_length, target], axis=1)
    n_user = max(uids.reshape((-1)).tolist())
    # n_item = len(set(rating_np[:, 1]))
    # 分割数据集并返回结果
    return dataset_split(rating_data, args.n_hist), n_user
    # return dataset_split(rating_data)


def dataset_split(rating_np, n_hist):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.1
    test_ratio = 0.1
    n_ratings = rating_np.shape[0]  # 统计评分记录数量

    # 采样验证集切片（对应评分文件中的行索引，采样10%）
    eval_indices = np.random.choice(
        n_ratings,
        size=int(n_ratings * eval_ratio),
        replace=False
    )

    # 在剩下的数据中，采样10%的测试集
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(
        list(left),
        size=int(n_ratings * test_ratio),
        replace=False
    )
    # 剩余的作为训练集
    train_indices = list(left - set(test_indices))
    # print(len(train_indices), len(eval_indices), len(test_indices))

    # traverse training data, only keeping the users with positive ratings
    # 遍历训练数据，获取训练集中 user id -> [pos_item1, pos_item2, ...] 的字典
    user_history_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][3 + n_hist]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)

    # 这里的操作相当于把验证集和测试集中 user id没有训练集中出现过的给去掉了
    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_history_dict]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]

    # print(len(train_indices), len(eval_indices), len(test_indices))

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = './data/' + args.dataset + '/kg_final'
    kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)

    # 统计实体数
    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    # 统计关系数
    n_relation = len(set(kg_np[:, 1]))

    # 构建知识图谱
    '''
        知识图谱的形式为:
        {
            h1: [(t1,r1), (t2,r2), ...],
            h2: ...
        }
    '''
    kg = construct_kg(kg_np)

    return n_entity, n_relation, kg


# 构建知识图
def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
        kg[tail].append((head, relation))
    avg_tail = 0
    num_nei = {}
    for key in kg:
        nei_n = len(kg[key])
        avg_tail += nei_n
        if nei_n in num_nei:
            num_nei[nei_n] += 1
        else:
            num_nei[nei_n] = 1
    keys = list(num_nei.keys())
    values = list(num_nei.values())
    nei_df = pd.DataFrame()
    nei_df['key'] = keys
    nei_df['value'] = values
    nei_df.to_csv('./data/' + 'amazon_book' + '/nei.csv')

    print("average number of neighbors: %d" % int(avg_tail // len(kg)))
    return kg


def construct_neighbor(args, data, ripple_set):
    users = torch.LongTensor(data[:, 0]).view(-1, 1)
    items = torch.LongTensor(data[:, 1]).view(-1, 1)
    labels = torch.LongTensor(data[:, 3 + args.n_hist]).view(-1, 1)  # label标签列表
    hists = torch.LongTensor(data[:, 2: 2 + args.n_hist])
    seq_length = torch.LongTensor(data[:, 2 + args.n_hist]).view(-1, 1)
    memories_h, memories_r, memories_t = [], [], []
    # 生成所有时序上的用户数据，放入列表 n_hist * [batch_size, n_memory]

    for idx in range(args.n_hist + 1):
        # 根据user id在波纹集中查找相应的 n-hop 数据，处理结果为三个形状为 n_hist * tensor(batch_size, n_memory) ，对应n-hop头/尾/关系的列表
        # head = hists[:, idx]
        head = data[:, 1 + idx]
        temp_h = torch.LongTensor([head.tolist() for i in range(args.n_memory)])
        memories_h.extend([temp_h.transpose(1, 0)])

        memories_r.extend(
            [torch.LongTensor(
                [
                    [
                        ripple_set[head][i][1] if head != 0 else 0
                        for i in range(args.n_memory)]
                    for head in data[:, 1 + idx]
                ]
            )]
        )
        memories_t.extend(
            [torch.LongTensor(
                [
                    [
                        ripple_set[head][i][0] if head != 0 else 0
                        for i in range(args.n_memory)]
                    for head in data[:, 1 + idx]
                ]
            )]
        )

    # h = torch.cat(memories_h, dim=1)
    # r = torch.cat(memories_r, dim=1)
    # t = torch.cat(memories_t, dim=1)
    concat_data = torch.cat(
        [users, items, hists, torch.cat(memories_h, dim=1), torch.cat(memories_r, dim=1), torch.cat(memories_t, dim=1),
         seq_length, labels],
        dim=1
    )

    if args.use_cuda:
        return concat_data.to('cuda')

    return concat_data


# 构建波纹集
def get_ripple_set(args, kg):
    print('constructing ripple set start...')

    # 构建一个iid到波纹集的字典，后续根据iid直接获取该波纹集
    '''
        {
            uid : [
                ([h1, h2], [r1, r2], [t1, t2]), ()
            ] 
    '''
    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = collections.defaultdict(list)

    for head in kg:
        replace = len(kg[head]) < args.n_memory
        indices = np.random.choice(
            len(kg[head]),
            size=args.n_memory,
            replace=replace
        )
        ripple_set[head] = np.array(kg[head])[indices]
    ripple_set = dict(sorted(ripple_set.items(), key=lambda x: x[0]))
    print('constructing ripple set end...')
    return ripple_set

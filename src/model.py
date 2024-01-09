import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.layers import MLPLayers

from sklearn.metrics import roc_auc_score
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.aggregator import Aggregator
from src.layer import DNN, AttentionSequencePoolingLayer, DynamicGRU
from src.transformer import TransformerEncoder
from src.utils import get_attention_mask


class RippleNetTrm(nn.Module):
    # 初始化
    def __init__(self, args, n_user, n_entity, n_relation):
        # def __init__(self, args, n_entity, n_relation):
        super(RippleNetTrm, self).__init__()
        # 设置超参数
        self._parse_args(args, n_user, n_entity, n_relation)
        self.tanh = nn.Tanh()  # 激活层
        if args.kge_type == 'MLP':
            # 实体嵌入层 （实体嵌入的维度是 dim）
            self.entity_emb = nn.Embedding(self.n_entity + 1, self.dim)
            # 关系嵌入层 （关系嵌入的维度是 dim*dim）
            self.relation_emb = nn.Embedding(self.n_relation + 1, self.dim * self.dim)
        elif args.kge_type == 'TransR':
            # 实体嵌入层 （实体嵌入的维度是 dim）
            self.entity_emb = nn.Embedding(self.n_entity + 1, self.dim)
            # # 2022年5月3日 16:25:13 修改关系嵌入和转换矩阵
            self.relation_emb = nn.Embedding(self.n_relation + 1, self.dim)
            # # 2022年5月3日 16:26:15 添加投影矩阵
            self.trans_w = nn.Embedding(self.n_relation + 1, self.dim * self.dim)  # 投影矩阵嵌入
        elif self.kge_type == 'STransD':
            # 实体嵌入层 （实体嵌入的维度是 dim）
            self.entity_emb = nn.Embedding(self.n_entity + 1, self.dim)
            self.trans_h = nn.Embedding(self.n_entity + 1, self.dim)
            self.trans_t = nn.Embedding(self.n_entity + 1, self.dim)
            self.relation_emb = nn.Embedding(self.n_relation + 1, self.dim)
            self.trans_r = nn.Embedding(self.n_relation + 1, self.dim)

        # self._parse_args(args, n_entity, n_relation)
        self.user_emb = nn.Embedding(self.n_user + 1, self.dim)

        # 关系嵌入层 （关系嵌入的维度是 dim*dim）
        # self.relation_emb = nn.Embedding(self.n_relation + 1, self.dim * self.dim)
        # 转换矩阵
        if self.item_update_mode in ['transform', 'plus_transform']:
            self.transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        elif self.item_update_mode == 'concat_transform':
            self.transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
            # self.transform_mlp = nn.ModuleList(
            #     [
            #         nn.Linear(self.dim * 3, self.dim * 2, bias=False),
            #         nn.Linear(self.dim * 2, self.dim, bias=False),
            #         # nn.Linear(self.dim, self.dim, bias=True),
            #         # nn.Linear(32, self.dim, bias=False)
            #     ]
            # )
            # self.relu = nn.LeakyReLU()

        # 位置编码
        self.position_embedding = nn.Embedding(self.n_hist, self.dim)
        # transformer层
        self.transformEncoder = TransformerEncoder(
            inner_size=256,
            hidden_dropout_prob=0.2,
            attn_dropout_prob=0.2,
            n_layers=1,
            n_heads=8,
            hidden_size=self.dim,
            overlay=True,
        )

        # self.gru = nn.LSTM(
        #     input_size=self.dim,
        #     hidden_size=self.dim,
        #     batch_first=True,
        #     num_layers=1
        # )

        # 注意力计算操作
        self.attention = AttentionSequencePoolingLayer(
            att_hidden_units=(200, 80),
            att_activation='dice',
            weight_normalization=True,
            embedding_dim=args.dim,
            return_score=False,
            supports_masking=False,
        )
        #
        # self.interest_evolution = DynamicGRU(
        #     input_size=args.dim,
        #     hidden_size=args.dim,
        #     gru_type='AUGRU'
        # )

        # DNN网络
        # self.dnn = DNN(
        #     inputs_dim=self.dim * 2,
        #     hidden_units=[256, 128, 64]
        # )
        #
        # # 输出一个单元的线性层
        # self.dnn_linear = nn.Linear(
        #     64, 1, bias=False
        # ).to('cuda')

        # DNN层
        self.dnn_mlp_layers = MLPLayers(
            [self.dim * 3, self.dim],
            activation='Dice',
            dropout=0.,
            bn=True
        )
        # 预测层
        self.dnn_predict_layers = nn.Linear(
            self.dim, 1
        )

        # 聚合层（？暂时使用一个聚合层把所有时序特征聚合）
        # self.aggregator = Aggregator(       # 聚合器
        #     self.batch_size,
        #     self.dim,
        #     'sum'
        # )
        # 二分类交叉熵损失
        self.criterion = nn.BCELoss()
        self.apply(self._init_weights)
        if args.use_cuda:
            self.to('cuda')

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _parse_args(self, args, n_user, n_entity, n_relation):
        # def _parse_args(self, args,  n_entity, n_relation):
        self.kge_type = args.kge_type
        self.n_user = n_user
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.n_memory = args.n_memory
        self.n_hist = args.n_hist
        self.item_update_mode = args.item_update_mode
        self.using_all_hops = args.using_all_hops

    def forward(
            self,
            users: torch.LongTensor,
            items: torch.LongTensor,
            hists: torch.LongTensor,
            labels: torch.LongTensor,
            memories_h: torch.LongTensor,
            memories_r: torch.LongTensor,
            memories_t: torch.LongTensor,
            seq_length
    ):
        """
        """
        batch_size = len(memories_h)
        memories_h = memories_h.view(batch_size, self.n_hist + 1, self.n_memory)
        memories_r = memories_r.view(batch_size, self.n_hist + 1, self.n_memory)
        memories_t = memories_t.view(batch_size, self.n_hist + 1, self.n_memory)

        '''先获取所有需要用到的特征的嵌入，包括user, item，波纹集的head/relation/tail'''
        user_embeddings = self.user_emb(users)
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)  # 获取候选项嵌入（entity）

        h_emb_list = []  # 头实体嵌入列表
        r_emb_list = []  # 关系嵌入列表
        t_emb_list = []  # 尾实体嵌入列表
        if self.kge_type == 'TransR':
            r_trans_w_list = []
        if self.kge_type == 'STransD':
            h_trans_w_list = []
            # t_trans_w_list = []
            r_trans_h_list = []
            r_trans_t_list = []

        # 获取所有的头尾关系嵌入
        for i in range(self.n_hist + 1):
            if self.kge_type == 'MLP':
                h_emb_list.append(self.entity_emb(memories_h[:, i, :]))
                r_emb_list.append(self.relation_emb(memories_r[:, i, :]).view(-1, self.n_memory, self.dim, self.dim))
                t_emb_list.append(self.entity_emb(memories_t[:, i, :]))
            elif self.kge_type == 'TransR':
                # n_hists * [batch size, n_memory, dim]
                h_emb_list.append(self.entity_emb(memories_h[:, i, :]))
                # n_hists * [batch size, n_memory, dim]
                r_emb_list.append(self.relation_emb(memories_r[:, i, :]))
                # n_hists * [batch size, n_memory, dim]
                t_emb_list.append(self.entity_emb(memories_t[:, i, :]))
                # n_hists * [batch_size, n_memory, dim, dim]
                r_trans_w_list.append(self.trans_w(memories_r[:, i, :]).view(-1, self.n_memory, self.dim, self.dim))
            elif self.kge_type == 'STransD':
                # n_hists * [batch size, n_memory, dim]
                h_emb_list.append(self.entity_emb(memories_h[:, i, :]))
                h_trans_w_list.append(self.entity_emb(memories_h[:, i, :]))
                # n_hists * [batch size, n_memory, dim]
                r_emb_list.append(self.relation_emb(memories_r[:, i, :]))
                r_trans_h_list.append(self.relation_emb(memories_r[:, i, :]))
                r_trans_t_list.append(self.relation_emb(memories_r[:, i, :]))
                # n_hists * [batch size, n_memory, dim]
                t_emb_list.append(self.entity_emb(memories_t[:, i, :]))
                # t_trans_w_list.append(self.entity_emb(memories_t[:, i, :]))
                # n_hists * [batch_size, n_memory, dim, dim]

        hist_list = []

        '''进行 n_hists 次 rippleNet 的关键计算，得到多个 item 表征'''
        for i in range(self.n_hist + 1):
            if self.kge_type == 'MLP':
                o = self._key_addressing(
                    h_emb_list[i], r_emb_list[i], t_emb_list[i], item_embeddings
                )
            elif self.kge_type == 'TransR':
                o = self._key_addressing(
                    h_emb_list[i], r_emb_list[i], t_emb_list[i],
                    item_embeddings,
                    r_trans_w=r_trans_w_list[i]
                )
            elif self.kge_type == 'STransD':
                o = self._key_addressing(
                    h_emb_list[i], r_emb_list[i], t_emb_list[i],
                    item_embeddings,
                    h_trans_w=h_trans_w_list[i],
                    r_trans_h=r_trans_h_list[i],
                    r_trans_t=r_trans_t_list[i],
                    # t_trans_w=t_trans_w_list[i]
                )
            if i == 0:
                item_embeddings = o
            else:
                hist_list.append(o)

        hist = torch.stack(hist_list, dim=1)  # [batch_size, n_hist, dim]

        # # GRU或LSTM部分
        # packed_keys = pack_padded_sequence(
        #     hist,
        #     lengths=seq_length.to('cpu'),
        #     batch_first=True,
        #     enforce_sorted=False
        # )
        # packed_interests, _ = self.gru(packed_keys)
        # hist, _ = pad_packed_sequence(
        #     packed_interests,
        #     batch_first=True,
        #     padding_value=0.0,
        #     total_length=self.n_hist
        # )

        # AUGRU部分
        # att_scores = self.attention(item_embeddings.unsqueeze(dim=1), hist, seq_length).squeeze(1)
        # packed_hist = pack_padded_sequence(hist, seq_length.to('cpu'), batch_first=True, enforce_sorted=False)
        # packed_scores = pack_padded_sequence(att_scores, lengths=seq_length.to('cpu'), batch_first=True,
        #                                      enforce_sorted=False)
        # outputs = self.interest_evolution(packed_hist, packed_scores)
        # outputs, _ = pad_packed_sequence(outputs, batch_first=True, padding_value=0.0, total_length=self.n_hist)
        # # pick last state
        # outputs = InterestEvolving._get_last_state(outputs, seq_length.to('cpu'))  # [b, H]
        # dnn_in = torch.cat([outputs, item_embeddings], dim=-1)
        '''进行Trm计算'''
        attention_mask = get_attention_mask(hists, bidirectional=False).to('cuda')
        # 位置编码
        position_ids = torch.arange(self.n_hist, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(len(items), self.n_hist).cuda()
        position_embedding = self.position_embedding(position_ids)
        hist += position_embedding

        # 进行trm计算
        trm_output = self.transformEncoder(
            hist,
            attention_mask,
            output_all_encoded_layers=False
        )

        # 将历史项与候选项做注意力计算
        hist = self.attention(item_embeddings.unsqueeze(dim=1), trm_output[-1], seq_length).squeeze()

        # hist = self.attention(item_embeddings.unsqueeze(dim=1), hist, seq_length).squeeze()
        # trm_output = torch.mean(trm_output[-1], dim=1)

        # hist = user_embeddings + hist

        dnn_in = torch.cat([user_embeddings, hist, item_embeddings], dim=-1)
        # dnn_out = self.dnn(dnn_in)
        dnn_out = self.dnn_mlp_layers(dnn_in)
        scores = self.dnn_predict_layers(dnn_out)
        # scores = self.dnn_linear(dnn_out)
        scores = torch.sigmoid(scores)
        scores = torch.squeeze(scores)
        # dnn_in = torch.cat([output, next_item], dim=-1)


        '''将用户的 N-hop 表征列表和 item表征 送入预测层'''
        # scores = self.predict(item_embeddings, hist)
        # scores = self.predict(user_embeddings, item_embeddings)

        '''计算损失'''
        if self.kge_type == 'MLP':
            return_dict = self._compute_loss(
                scores, labels, h_emb_list, t_emb_list, r_emb_list
            )
        elif self.kge_type == 'TransR':
            return_dict = self._compute_loss(
                scores, labels, h_emb_list, t_emb_list, r_emb_list, r_trans_w_list=r_trans_w_list
            )
        elif self.kge_type == 'STransD':
            return_dict = self._compute_loss(
                scores, labels, h_emb_list, t_emb_list, r_emb_list,
                h_trans_w_list=h_trans_w_list,
                r_trans_h_list=r_trans_h_list,
                r_trans_t_list=r_trans_t_list,
                # t_trans_w_list=t_trans_w_list,
            )
        return_dict["scores"] = scores

        return return_dict

    def _compute_loss(self, scores, labels, h_emb_list, t_emb_list, r_emb_list,
                      r_trans_w_list=None,
                      h_trans_w_list=None, r_trans_h_list=None, r_trans_t_list=None, t_trans_w_list=None):
        # 基础损失（二分类交叉熵损失计算CTR预测的损失）
        base_loss = self.criterion(scores, labels.float())
        # 计算KGE损失
        kge_loss = 0

        if self.kge_type == 'MLP':
            for i in range(self.n_hist):
                # [batch size, n_memory, 1, dim]
                h_expanded = torch.unsqueeze(h_emb_list[i], dim=2)
                # [batch size, n_memory, dim, 1]
                t_expanded = torch.unsqueeze(t_emb_list[i], dim=3)
                # [batch size, n_memory, dim, dim]
                hRt = torch.squeeze(
                    torch.matmul(torch.matmul(h_expanded, r_emb_list[i]), t_expanded)
                )
                kge_loss += torch.sigmoid(hRt).mean()
        elif self.kge_type == 'TransR':
            for i in range(self.n_hist):
                Rh = torch.matmul(h_emb_list[i].unsqueeze(dim=2), r_trans_w_list[i]).squeeze()
                Rt = torch.matmul(t_emb_list[i].unsqueeze(dim=2), r_trans_w_list[i]).squeeze()
                # [batch size, n_memory, dim, dim]
                hRt = torch.squeeze(Rh + r_emb_list[i] - Rt).sum(dim=2)
                hRt = torch.sigmoid(hRt).mean()
                kge_loss += torch.sigmoid(hRt).mean()
        elif self.kge_type == 'STransD':
            for i in range(self.n_hist):
                Rh = h_emb_list[i] + torch.sum(h_emb_list[i] * h_trans_w_list[i], axis=-1, keepdim=True) * \
                     r_trans_h_list[i]
                Rt = t_emb_list[i] + torch.sum(t_emb_list[i] * h_trans_w_list[i], axis=-1, keepdim=True) * \
                     r_trans_t_list[i]
                # [batch size, n_memory, dim]
                # hRt = torch.squeeze(Rh + r_emb_list[i] - Rt).sum(dim=2)
                hRt = torch.sum(Rh * r_emb_list[i] * Rt, dim=-1)
                hRt = torch.sigmoid(hRt).mean()
                kge_loss += torch.sigmoid(hRt).mean()

        # kge_loss /= self.n_hist
        kge_loss = -self.kge_weight * kge_loss
        # 正则损失
        l2_loss = 0
        for i in range(self.n_hist):
            l2_loss += (h_emb_list[i] * h_emb_list[i]).sum()
            l2_loss += (t_emb_list[i] * t_emb_list[i]).sum()
            l2_loss += (r_emb_list[i] * r_emb_list[i]).sum()
        l2_loss = self.l2_weight * l2_loss

        loss = base_loss + kge_loss + l2_loss
        return dict(base_loss=base_loss, kge_loss=kge_loss, l2_loss=l2_loss, loss=loss)

    def _key_addressing(self, h_emb, r_emb, t_emb, item_embeddings, r_trans_w=None,
                        h_trans_w=None, r_trans_h=None, r_trans_t=None, t_trans_w=None):
        """
        Trm版只用一跳邻居，所以直接去掉循环多个跳的设置
        """
        # o_list = []
        # for hop in range(self.n_hop):
        if self.kge_type == 'MLP':
            '''对应公式中的 $R_i * h_i$ 部分，类似于将 头实体投影到关系空间中的操作'''
            # [batch_size, n_memory, dim, 1]
            h_expanded = torch.unsqueeze(h_emb, dim=3)
            # [batch_size, n_memory, dim]
            Rh = torch.squeeze(
                torch.matmul(r_emb, h_expanded)
            )
            '''对应公式中的 $v^T * R_i * h_i 部分，可以视为是 item 嵌入 和 头实体 的相似度（注意力得分）$'''
            # [batch_size, dim, 1]
            v = torch.unsqueeze(item_embeddings, dim=2)
            # [batch_size, n_memory]
            probs = torch.squeeze(torch.matmul(Rh, v))
            '''对相似度得分进行归一化'''
            # [batch_size, n_memory]
            probs_normalized = F.softmax(probs, dim=1)
            probs_expanded = torch.unsqueeze(probs_normalized, dim=2)

        elif self.kge_type == 'TransR':
            Rh = torch.matmul(h_emb.unsqueeze(dim=2), r_trans_w).squeeze()
            Rt = torch.matmul(t_emb.unsqueeze(dim=2), r_trans_w).squeeze()
            # 计算kg得分
            kg_score = torch.mul(Rt, self.tanh(Rh + r_emb))
            kg_score = kg_score.sum(dim=2)  # [batch_size,1, n_dim]

            '''对相似度得分进行归一化'''
            # [batch_size, n_memory]
            probs_normalized = F.softmax(kg_score, dim=1)
            probs_expanded = torch.unsqueeze(probs_normalized, dim=2)
        elif self.kge_type == 'STransD':
            Rh = h_emb + torch.sum(h_emb * h_trans_w, axis=-1, keepdim=True) * r_trans_h
            Rt = t_emb + torch.sum(t_emb * h_trans_w, axis=-1, keepdim=True) * r_trans_t
            # 计算kg得分
            kg_score = torch.mul(Rt, self.tanh(Rh + r_emb))
            kg_score = kg_score.sum(dim=2)  # [batch_size,1, n_dim]
            '''对相似度得分进行归一化'''
            # [batch_size, n_memory]
            probs_normalized = F.softmax(kg_score, dim=1)
            probs_expanded = torch.unsqueeze(probs_normalized, dim=2)

        '''使用注意力得分对尾实体进行加权（尾实体就是用户行为历史的 n-hop 邻居），然后对所有加权后的尾实体进行求和'''
        # [batch_size, dim]
        o = (t_emb * probs_expanded).sum(dim=1)

        # 更新候选项的表征
        o = self._update_item_embedding(h_emb[:, -1, :], o)
        return o

    # def _key_addressing(self, h_emb, r_emb, t_emb, r_trans_w, item_embeddings):
    #     """
    #     :param h_emb: 头实体嵌入列表  [batch size, n_memory, dim]
    #     :param r_emb: 关系嵌入列表 [batch size, n_memory, dim, dim]
    #     :param t_emb: 尾实体嵌入向量  [batch size, n_memory, dim]
    #     :param item_embeddings: item 嵌入向量 [batch_size, dim]
    #     :return:
    #     """
    #     """Trm版只用一跳邻居，所以直接去掉循环多个跳的设置"""
    #     # o_list = []
    #     # for hop in range(self.n_hop):
    #     '''对应公式中的 $R_i * h_i$ 部分，类似于将 头实体投影到关系空间中的操作'''
    #     # # [batch_size, n_memory, dim, 1]
    #     # h_expanded = torch.unsqueeze(h_emb, dim=3)
    #     # # [batch_size, n_memory, dim]
    #     # Rh = torch.squeeze(
    #     #     torch.matmul(r_emb, h_expanded)
    #     # )
    #     #
    #     # # 2022年5月3日 16:17:28 将尾实体也投影到关系空间
    #     # t_expanded = torch.unsqueeze(t_emb, dim=3)
    #     # # [batch_size, n_memory, dim]
    #     # Rt = torch.squeeze(
    #     #     torch.matmul(r_emb, t_expanded)
    #     # )
    #     # 2022年5月3日 16:28:36 添加计算KGE得分
    #     Rh = torch.matmul(h_emb.unsqueeze(dim=2), r_trans_w).squeeze()
    #     Rt = torch.matmul(t_emb.unsqueeze(dim=2), r_trans_w).squeeze()
    #
    #     kg_score = torch.mul(Rt, self.tanh(Rh + r_emb))
    #
    #     kg_score = kg_score.sum(dim=2)  # [batch_size,1, n_dim]
    #
    #     '''对应公式中的 $v^T * R_i * h_i 部分，可以视为是 item嵌入 和 头实体 的相似度（注意力得分）$'''
    #     # # [batch_size, dim, 1]
    #     # v = torch.unsqueeze(item_embeddings, dim=2)
    #     # # [batch_size, n_memory]
    #     # probs = torch.squeeze(torch.matmul(Rh, v))
    #
    #     '''对相似度得分进行归一化'''
    #     # [batch_size, n_memory]
    #     probs_normalized = F.softmax(kg_score, dim=1)
    #
    #     # probs_normalized = probs_normalized.sum(dim=1)
    #
    #     probs_expanded = torch.unsqueeze(probs_normalized, dim=2)
    #
    #     '''使用注意力得分对尾实体进行加权（尾实体就是用户行为历史的 n-hop 邻居），然后对所有加权后的尾实体进行求和'''
    #     # [batch_size, dim]
    #     o = (t_emb * probs_expanded).sum(dim=1)
    #
    #     """！ 修改RippleNet，不更新候选项的表征"""
    #     o = self._update_item_embedding(h_emb[:, -1, :], o)
    #     return o

    # 更新物品表征（包含多种更新策略）
    def _update_item_embedding(self, item_embeddings, o):
        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = self.transform_matrix(o)
        elif self.item_update_mode == "plus_transform":
            item_embeddings = self.transform_matrix(item_embeddings + o)
        elif self.item_update_mode == 'concat_transform':
            item_embeddings = self.transform_matrix(
                torch.mean(torch.cat([torch.unsqueeze(item_embeddings, 1), torch.unsqueeze(o, 1)], dim=1), dim=1)
            )
            # trans_in = torch.cat([item_embeddings, item_embeddings + o, item_embeddings * o], dim=-1)
            # for layer in self.transform_mlp:
            #     trans_in = layer(trans_in)
            #     # trans_in = self.relu(trans_in)
            # trans_in = self.relu(trans_in)
            # item_embeddings = trans_in
            # item_embeddings = self.transform_matrix(torch.cat([item_embeddings, o], dim=-1))
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings

    # 预测层操作
    def predict(self, item_embeddings, o_list):
        y = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]

        # 用户表示 * 物品表示 并求和
        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)
        # 归一化后输出，作为预测概率
        return torch.sigmoid(scores)

    def evaluate(self, users, items, hists, labels, memories_h, memories_r, memories_t, seq_len):
        return_dict = self.forward(users, items, hists, labels, memories_h, memories_r, memories_t, seq_len)
        scores = return_dict["scores"].detach().cpu().numpy()
        labels = labels.cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        return auc, acc


class InterestEvolving(nn.Module):
    __SUPPORTED_GRU_TYPE__ = ['GRU', 'AIGRU', 'AGRU', 'AUGRU']

    def __init__(self,
                 input_size,
                 gru_type='GRU',
                 use_neg=False,
                 init_std=0.001,
                 att_hidden_size=(64, 16),
                 att_activation='sigmoid',
                 att_weight_normalization=False):
        super(InterestEvolving, self).__init__()
        if gru_type not in InterestEvolving.__SUPPORTED_GRU_TYPE__:
            raise NotImplementedError("gru_type: {gru_type} is not supported")
        self.gru_type = gru_type
        self.use_neg = use_neg

        if gru_type == 'GRU':
            self.attention = AttentionSequencePoolingLayer(
                embedding_dim=input_size,
                att_hidden_units=att_hidden_size,
                att_activation=att_activation,
                weight_normalization=att_weight_normalization,
                return_score=False
            )
            self.interest_evolution = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        elif gru_type == 'AIGRU':
            self.attention = AttentionSequencePoolingLayer(embedding_dim=input_size,
                                                           att_hidden_units=att_hidden_size,
                                                           att_activation=att_activation,
                                                           weight_normalization=att_weight_normalization,
                                                           return_score=True)
            self.interest_evolution = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        elif gru_type == 'AGRU' or gru_type == 'AUGRU':
            self.attention = AttentionSequencePoolingLayer(
                embedding_dim=input_size,
                att_hidden_units=att_hidden_size,
                att_activation=att_activation,
                weight_normalization=att_weight_normalization,
                return_score=True
            )
            self.interest_evolution = DynamicGRU(
                input_size=input_size,
                hidden_size=input_size,
                gru_type=gru_type
            )
        for name, tensor in self.interest_evolution.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    @staticmethod
    def _get_last_state(states, keys_length):
        # states [B, T, H]
        batch_size, max_seq_length, _ = states.size()

        mask = (torch.arange(max_seq_length, device=keys_length.device).repeat(
            batch_size, 1) == (keys_length.view(-1, 1) - 1))

        return states[mask]

    def forward(self, query, keys, keys_length, mask=None):
        """
        Parameters
        ----------
        query: 2D tensor, [B, H]
        keys: (masked_interests), 3D tensor, [b, T, H]
        keys_length: 1D tensor, [B]

        Returns
        -------
        outputs: 2D tensor, [B, H]
        """
        batch_size, dim = query.size()
        max_length = keys.size()[1]

        # check batch validation
        zero_outputs = torch.zeros(batch_size, dim, device=query.device)
        mask = keys_length > 0
        # [B] -> [b]
        keys_length = keys_length[mask]
        if keys_length.shape[0] == 0:
            return zero_outputs

        # [B, H] -> [b, 1, H]
        query = torch.masked_select(query, mask.view(-1, 1)).view(-1, dim).unsqueeze(1)

        if self.gru_type == 'GRU':
            packed_keys = pack_padded_sequence(keys, lengths=keys_length, batch_first=True, enforce_sorted=False)
            packed_interests, _ = self.interest_evolution(packed_keys)
            interests, _ = pad_packed_sequence(packed_interests, batch_first=True, padding_value=0.0,
                                               total_length=max_length)
            outputs = self.attention(query, interests, keys_length.unsqueeze(1))  # [b, 1, H]
            outputs = outputs.squeeze(1)  # [b, H]
        elif self.gru_type == 'AIGRU':
            att_scores = self.attention(query, keys, keys_length.unsqueeze(1))  # [b, 1, T]
            interests = keys * att_scores.transpose(1, 2)  # [b, T, H]
            packed_interests = pack_padded_sequence(interests, lengths=keys_length, batch_first=True,
                                                    enforce_sorted=False)
            _, outputs = self.interest_evolution(packed_interests)
            outputs = outputs.squeeze(0)  # [b, H]
        elif self.gru_type == 'AGRU' or self.gru_type == 'AUGRU':
            att_scores = self.attention(query, keys, keys_length.unsqueeze(1)).squeeze(1)  # [b, T]
            packed_interests = pack_padded_sequence(keys, lengths=keys_length, batch_first=True,
                                                    enforce_sorted=False)
            packed_scores = pack_padded_sequence(att_scores, lengths=keys_length, batch_first=True,
                                                 enforce_sorted=False)
            outputs = self.interest_evolution(packed_interests, packed_scores)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True, padding_value=0.0, total_length=max_length)
            # pick last state
            outputs = InterestEvolving._get_last_state(outputs, keys_length)  # [b, H]
        # [b, H] -> [B, H]
        zero_outputs[mask] = outputs
        return zero_outputs

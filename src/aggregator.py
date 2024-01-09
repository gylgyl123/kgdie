import torch
import torch.nn.functional as F


class Aggregator(torch.nn.Module):
    '''
    Aggregator class   聚合器类
    Mode in ['sum', 'concat', 'neighbor']
    '''
    
    def __init__(self, batch_size, dim, aggregator):
        super(Aggregator, self).__init__()
        self.batch_size = batch_size
        self.dim = dim
        if aggregator == 'concat':
            self.weights = torch.nn.Linear(2 * dim, dim, bias=True)
        else:
            self.weights = torch.nn.Linear(dim, dim, bias=True)
        self.aggregator = aggregator
        
    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, act):
        batch_size = user_embeddings.size(0)  # 获取batch size
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        '''获得邻域的线性组合 [batch_size, 1, dim]'''
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        # 根据参数选择不同的聚合方式
        if self.aggregator == 'sum':
            output = (self_vectors + neighbors_agg).view((-1, self.dim))
            
        elif self.aggregator == 'concat':
            output = torch.cat((self_vectors, neighbors_agg), dim=-1)
            output = output.view((-1, 2 * self.dim))
            
        else:
            output = neighbors_agg.view((-1, self.dim))

        # 聚合之后与权重相乘
        output = self.weights(output)
        # 最后激活并输出
        return act(output.view((self.batch_size, -1, self.dim)))
        
    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        '''
        This aims to aggregate neighbor vectors  它的目的是聚集相邻向量
        '''
        # reshape
        # [batch_size, 1, dim] -> [batch_size, 1, 1, dim]
        user_embeddings = user_embeddings.view((self.batch_size, 1, 1, self.dim))

        # 这里就是用 user embed与所有的关系进行了内积计算，作为相似度
        # [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, n_neighbor]
        user_relation_scores = (user_embeddings * neighbor_relations).sum(dim = -1)
        # 对分数进行softmax
        user_relation_scores_normalized = F.softmax(user_relation_scores, dim = -1)

        # reshape
        # [batch_size, -1, n_neighbor] -> [batch_size, -1, n_neighbor, 1]
        user_relation_scores_normalized = user_relation_scores_normalized.unsqueeze(dim = -1)

        # 将权重值与实体表示相乘，然后对所有neighbors嵌入求和，得到邻居表示（即论文中的领域的线性组合）
        # [batch_size, -1, n_neighbor, 1] * [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, dim]
        neighbors_aggregated = (user_relation_scores_normalized * neighbor_vectors).sum(dim = 2)
        
        return neighbors_aggregated
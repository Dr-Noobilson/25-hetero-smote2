import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

# Content Aggregation Layer
class MAG_Content_Agg(nn.Module):
    def __init__(self, embed_dim, attr_size, dropout):
        super(MAG_Content_Agg, self).__init__()
        self.fc = nn.Linear(attr_size, embed_dim)
        self.dropout = dropout

    def forward(self, embed_list):
        # Concatenate the embeddings
        x = torch.cat(embed_list, dim=-1)
        x = F.normalize(F.relu(self.fc(x)), p=2, dim = -1)
        # x = F.dropout(x, self.dropout, training=self.training) 
        return x


class MAGNN_Agg(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(MAGNN_Agg, self).__init__()
        self.fc_s1s = nn.Linear(embed_dim, embed_dim)
        self.fc_s2s = nn.Linear(embed_dim, embed_dim)
        self.fc_s3s = nn.Linear(embed_dim, embed_dim)
        self.fc_s121s = nn.Linear(embed_dim, embed_dim)
        self.fc_s131s = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

        self.att_vec = nn.Parameter(torch.Tensor(5, embed_dim))
        nn.init.xavier_uniform_(self.att_vec)

    def forward(self, x_list, edge_index_list, x_node, edge_weight_list, edge_index_12, edge_index_13):

        edge_index_s1 = edge_index_list[1]
        edge_index_s2 = edge_index_list[2]
        edge_index_s3 = edge_index_list[3]

        if edge_weight_list[0] is None: 
            edge_weight_list = [torch.ones(edge_index.size(1), device=x_node.device) for edge_index in edge_index_list]

        # s-1-s aggregation
        msg_1 = scatter_mean(x_node[edge_index_s1[0]] * edge_weight_list[1].view(-1, 1), edge_index_s1[1], dim=0, dim_size=x_list[1].size(0))
        net_msg1 = (msg_1 + x_list[1]) / 2
        s1s_agg = scatter_mean(net_msg1[edge_index_s1[1]], edge_index_s1[0], dim=0, dim_size=x_node.size(0))
        s1s_agg = F.relu(self.fc_s1s(s1s_agg))
        s1s_agg = F.dropout(s1s_agg, self.dropout, training=self.training)

        # s-2-s aggregation
        msg_2 = scatter_mean(x_node[edge_index_s2[0]] * edge_weight_list[2].view(-1, 1), edge_index_s2[1], dim=0, dim_size=x_list[2].size(0))
        net_msg2 = (msg_2 + x_list[2]) / 2
        s2s_agg = scatter_mean(net_msg2[edge_index_s2[1]], edge_index_s2[0], dim=0, dim_size=x_node.size(0))
        s2s_agg = F.relu(self.fc_s2s(s2s_agg))
        s2s_agg = F.dropout(s2s_agg, self.dropout, training=self.training)

        # s-3-s aggregation
        msg_3 = scatter_mean(x_node[edge_index_s3[0]] * edge_weight_list[3].view(-1, 1), edge_index_s3[1], dim=0, dim_size=x_list[3].size(0))
        net_msg3 = (msg_3 + x_list[3]) / 2
        s3s_agg = scatter_mean(net_msg3[edge_index_s3[1]], edge_index_s3[0], dim=0, dim_size=x_node.size(0))
        s3s_agg = F.relu(self.fc_s3s(s3s_agg))
        s3s_agg = F.dropout(s3s_agg, self.dropout, training=self.training)

        # s-1-2-1-s aggregation
        # s -> 1
        msg_1_s121s = scatter_mean(x_node[edge_index_s1[0]] * edge_weight_list[1].view(-1, 1), edge_index_s1[1], dim=0, dim_size=x_list[1].size(0))
        net_msg1_s121s = (msg_1_s121s+ x_list[1]) / 2
        # 1 -> 2
        msg_2_s121s = scatter_mean(net_msg1_s121s[edge_index_12[0]], edge_index_12[1], dim=0, dim_size=x_list[2].size(0))
        net_msg2_s121s = (msg_2_s121s+ x_list[2]) / 2
        # 2 -> 1
        msg_3_s121s = scatter_mean(net_msg2_s121s[edge_index_12[1]], edge_index_12[0], dim=0, dim_size=x_list[1].size(0))
        net_msg3_s121s = (msg_3_s121s+ x_list[1]) / 2
        # 1 -> s
        s121s_agg = scatter_mean(net_msg3_s121s[edge_index_s1[1]] * edge_weight_list[1].view(-1, 1), edge_index_s1[0], dim=0, dim_size=x_node.size(0))
        s121s_agg = F.relu(self.fc_s121s(s121s_agg))
        s121s_agg = F.dropout(s121s_agg, self.dropout, training=self.training)

        # s-1-3-1-s aggregation
        # s -> 1
        msg_1_s131s = scatter_mean(x_node[edge_index_s1[0]] * edge_weight_list[1].view(-1, 1), edge_index_s1[1], dim=0, dim_size=x_list[1].size(0))
        net_msg1_s131s = (msg_1_s131s+ x_list[1]) / 2
        # 1 -> 3
        msg_2_s131s = scatter_mean(net_msg1_s131s[edge_index_13[0]], edge_index_13[1], dim=0, dim_size=x_list[3].size(0))
        net_msg2_s131s = (msg_2_s131s+ x_list[3]) / 2
        # 3 -> 1
        msg_3_s131s = scatter_mean(net_msg2_s131s[edge_index_13[1]], edge_index_13[0], dim=0, dim_size=x_list[1].size(0))
        net_msg3_s131s = (msg_3_s131s+ x_list[1]) / 2
        # 1 -> s
        s131s_agg = scatter_mean(net_msg3_s131s[edge_index_s1[1]] * edge_weight_list[1].view(-1, 1), edge_index_s1[0], dim=0, dim_size=x_node.size(0))
        s131s_agg = F.relu(self.fc_s131s(s131s_agg))
        s131s_agg = F.dropout(s131s_agg, self.dropout, training=self.training)

        # Stack all meta-path embeddings for inter-metapath attention
        all_metapath = torch.stack([s1s_agg, s2s_agg, s3s_agg, s121s_agg, s131s_agg], dim=1)  # [num_nodes, 5, embed_dim]

        # Attention score computation
        att_scores = (all_metapath * self.att_vec.unsqueeze(0)).sum(dim=2)  # [num_nodes, 5]
        att_weights = F.softmax(att_scores, dim=1).unsqueeze(-1)            # [num_nodes, 5, 1]

        h_final = torch.sum(all_metapath * att_weights, dim=1)              # [num_nodes, embed_dim]

        return h_final
    

class MAG_ConEn(nn.Module):
    def __init__(self, embed_dim, args, dropout):
        super(MAG_ConEn, self).__init__()

        self.a_con = MAG_Content_Agg(embed_dim, args.A_emsize, dropout)
        self.p_con = MAG_Content_Agg(embed_dim, args.P_emsize, dropout)
        self.t_con = MAG_Content_Agg(embed_dim, args.T_emsize, dropout)
        self.c_con = MAG_Content_Agg(embed_dim, args.C_emsize, dropout)
        
        self.a_cont = torch.empty(0)
        self.p_cont = torch.empty(0)
        self.t_cont = torch.empty(0)
        self.c_cont = torch.empty(0)
        
    def forward(self, data):
        
        a_embed_list = [data['a'].x]
        p_embed_list = [data['p'].x]
        t_embed_list = [data['t'].x]
        c_embed_list = [data['c_embed'].x]
        
        self.a_cont = self.a_con(a_embed_list)
        self.p_cont = self.p_con(p_embed_list)
        self.t_cont = self.t_con(t_embed_list)
        self.c_cont = self.c_con(c_embed_list)
        
        return [self.a_cont, self.p_cont, self.t_cont, self.c_cont]
            


class MAG_NetEn(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(MAG_NetEn, self).__init__()
        
        self.a_het = MAGNN_Agg(embed_dim, dropout)
        # self.a_het = MAGNN_Agg(embed_dim, dropout)
        # self.d_het = MAGNN_Agg(embed_dim, dropout)
        
    def forward(self, x_list, data):
    
        a_edges_list = [data['a', 'walk', 'a'].edge_index, data['a', 'walk', 'p'].edge_index, data['a', 'walk', 't'].edge_index, data['a', 'walk', 'c'].edge_index]
        p_edges_list = [data['p', 'walk', 'p'].edge_index, data['p', 'walk', 'a'].edge_index, data['p', 'walk', 't'].edge_index, data['p', 'walk', 'c'].edge_index]
        # t_edges_list = [data['t', 'walk', 't'].edge_index, data['t', 'walk', 'a'].edge_index, data['t', 'walk', 'p'].edge_index, data['t', 'walk', 'c'].edge_index]
        # c_edges_list = [data['c', 'walk', 'c'].edge_index, data['c', 'walk', 'a'].edge_index, data['c', 'walk', 'p'].edge_index, data['c', 'walk', 't'].edge_index]
        edge_weight_list = [None, None, None, None]
        
        x_list[0] = self.a_het(x_list, a_edges_list, x_list[0], edge_weight_list, p_edges_list[2], p_edges_list[3]) 
        
        return x_list


class MAG_classify(nn.Module):
    def __init__(self, embed_dim, nclass, dropout):
        super(MAG_classify, self).__init__()
        self.a_het = MAGNN_Agg(embed_dim, dropout)
        self.mlp = nn.Linear(embed_dim, nclass)    
        self.dropout = dropout
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x_list, edge_list, edge_weight_list, edge_index_mid, edge_index_mid2):
        x = self.a_het(x_list, edge_list, x_list[0], edge_weight_list, edge_index_mid, edge_index_mid2)
        x = torch.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x


class EdgePredictor(nn.Module):
    def __init__(self, nembed, dropout=0.1):
        super(EdgePredictor, self).__init__()
        self.dropout = dropout
        self.lin1 = nn.Linear(nembed, nembed)
        self.lin2 = nn.Linear(nembed, nembed)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.lin1.weight,std=0.05)
        nn.init.normal_(self.lin2.weight,std=0.05)

    def forward(self, node_embed1, node_embed2):
        
        combine1 = self.lin1(node_embed1)
        combine2 = self.lin2(node_embed2)
        result = torch.mm(combine1, combine2.transpose(-1, -2))

        adj_out = torch.sigmoid(result)         # Apply sigmoid along dim=1 (rows)
        return adj_out
    
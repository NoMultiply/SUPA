from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class DecayLayer(nn.Module):
    def __init__(self, device, w=1, decay_method='log'):
        super(DecayLayer, self).__init__()
        self.device = device
        self.decay_method = decay_method
        self.w = w

    def exp_decay(self, delta_t):
        return torch.exp(-self.w * delta_t)

    def log_decay(self, delta_t):
        return 1 / torch.log(2.7183 + self.w * delta_t)

    def rev_decay(self, delta_t):
        return 1 / (1 + self.w * delta_t)

    def forward(self, delta_t):
        if self.decay_method == 'exp':
            return self.exp_decay(delta_t)
        elif self.decay_method == 'log':
            return self.log_decay(delta_t)
        elif self.decay_method == 'rev':
            return self.rev_decay(delta_t)
        else:
            return self.exponetial_decay(delta_t)


class Updater(nn.Module):
    def __init__(self, n_node_types, embedding_size, device):
        super(Updater, self).__init__()
        self.n_node_types = n_node_types
        self.embedding_size = embedding_size
        self.device = device

        self.decay = DecayLayer(device)
        self.alpha = Parameter(torch.FloatTensor(n_node_types))
        self.alpha.data.uniform_(0.0, 1.0)

    def forward(self, embeddings, delta, node_types):
        return embeddings * self.decay(torch.sigmoid(self.alpha[node_types]) * delta).view(-1, 1)


class SUPAModel(nn.Module):
    def __init__(self, device, n_nodes, embedding_size, n_node_types, n_edge_types, node_types, edge_types,
                 n_negative=5, time_agg=86400):
        super(SUPAModel, self).__init__()
        self.device = device
        self.n_nodes = n_nodes
        self.embedding_size = embedding_size
        self.n_negative = n_negative
        self.time_agg = time_agg

        self.nodes = set()
        self.node_emb = nn.Embedding(n_nodes, embedding_size)
        self.node_emb.weight.data.uniform_(-1, 1)
        self.short_emb = nn.Embedding(n_nodes, embedding_size)
        self.short_emb.weight.data.uniform_(-1, 1)

        self.edge_types = edge_types
        self.n_edge_types = n_edge_types
        self.edge_embedding = Parameter(torch.FloatTensor(n_nodes, n_edge_types, embedding_size))
        self.edge_embedding.data.uniform_(-1, 1)

        self.decay_layer = DecayLayer(device)
        self.recent_timestamp = torch.zeros(n_nodes, dtype=torch.float, requires_grad=False).to(device)
        list_node_types = [0 for _ in range(n_nodes)]
        for k, v in node_types.items():
            list_node_types[k] = v
        self.node_types = torch.tensor(list_node_types, dtype=torch.long, requires_grad=False)
        self.updater = Updater(n_node_types, embedding_size, device)

        self.to(device)

    def reset_time(self):
        print('[SUPAModel] Reset time')
        self.nodes = set()

    def node_embeddings(self, nodes, edge_types):
        return self.target_embeddings(nodes, edge_types) + self.context_embeddings(nodes, edge_types)

    def target_embeddings(self, nodes, _):
        return self.node_emb(nodes) + self.short_emb(nodes)

    def context_embeddings(self, nodes, edge_types):
        return self.edge_embedding[nodes, edge_types]

    def update_nodes(self, edges):
        self.nodes.update(edges[:, [0, 1]].cpu().unique().numpy())
        nodes = torch.LongTensor(list(self.nodes)).to(self.device)
        return nodes

    def update_times(self, edges):
        for edge in edges:
            self.recent_timestamp[edge[0]] = edge[3].float()
            self.recent_timestamp[edge[1]] = edge[3].float()

    def forward(self, edges, walks, walks_edge_types, nodes, batch_size, n_positive,
                repeat_edge_types, u_time_delta, v_time_delta, u_pos_reps_mask, v_pos_reps_mask,
                u_pos_reps_loss_mask, v_pos_reps_loss_mask):

        u_reps = self.node_emb(edges[:, 0]) + self.updater.forward(
            self.short_emb(edges[:, 0]), u_time_delta, self.node_types[edges[:, 0]])
        v_reps = self.node_emb(edges[:, 1]) + self.updater.forward(
            self.short_emb(edges[:, 1]), v_time_delta, self.node_types[edges[:, 1]])

        u_reps_edge = (u_reps + self.context_embeddings(edges[:, 0], edges[:, 2])) / 2
        v_reps_edge = (v_reps + self.context_embeddings(edges[:, 1], edges[:, 2])) / 2

        u_interact_info = u_reps
        v_interact_info = v_reps

        u_pos_reps = self.context_embeddings(walks[:, 0, :, :].view(batch_size, -1),
                                             walks_edge_types[:, 0, :, :].view(batch_size, -1))
        v_pos_reps = self.context_embeddings(walks[:, 1, :, :].view(batch_size, -1),
                                             walks_edge_types[:, 1, :, :].view(batch_size, -1))

        u_neg_reps = self.context_embeddings(
            nodes[
                torch.randint(0, nodes.size(0), (batch_size * self.n_negative * n_positive,)).to(self.device)
            ].view(batch_size, -1),
            repeat_edge_types
        )

        v_neg_reps = self.context_embeddings(
            nodes[
                torch.randint(0, nodes.size(0), (batch_size * self.n_negative * n_positive,)).to(self.device)
            ].view(batch_size, -1),
            repeat_edge_types
        )

        return (
            u_reps, v_reps, u_pos_reps, v_pos_reps, u_neg_reps, v_neg_reps, n_positive,
            u_pos_reps_mask, v_pos_reps_mask,
            u_interact_info, v_interact_info,
            u_pos_reps_loss_mask, v_pos_reps_loss_mask,
            u_reps_edge, v_reps_edge
        )

    @staticmethod
    def loss(u_reps, v_reps, u_pos_reps, v_pos_reps, u_neg_reps,
             v_neg_reps, n_positive, u_pos_reps_mask, v_pos_reps_mask,
             u_interact_info, v_interact_info,
             u_pos_reps_loss_mask, v_pos_reps_loss_mask,
             u_reps_edge, v_reps_edge):
        batch_size, embedding_size = u_reps.size()

        edge_loss = torch.bmm(u_reps_edge.view(batch_size, 1, embedding_size),
                              v_reps_edge.view(batch_size, embedding_size, 1)).sigmoid().log().squeeze()

        u_reps = u_reps.view(batch_size, embedding_size, 1)
        v_reps = v_reps.view(batch_size, embedding_size, 1)

        u_interact_info = u_interact_info.view(batch_size, embedding_size, 1)
        v_interact_info = v_interact_info.view(batch_size, embedding_size, 1)
        u_pos_loss = (torch.bmm(
            (u_pos_reps.view(batch_size, n_positive, -1) * u_pos_reps_mask.view(batch_size, n_positive, -1)),
            u_interact_info
        ).sigmoid().log().view(batch_size, -1) * u_pos_reps_loss_mask).sum(1)

        v_pos_loss = (torch.bmm(
            (v_pos_reps.view(batch_size, n_positive, -1) * v_pos_reps_mask.view(batch_size, n_positive, -1)),
            v_interact_info
        ).sigmoid().log().view(batch_size, -1) * v_pos_reps_loss_mask).sum(1)

        u_noise_loss = torch.bmm(u_neg_reps.view(batch_size, -1, embedding_size).neg(),
                                 u_reps).sigmoid().log().view(batch_size, -1).sum(1)

        v_noise_loss = torch.bmm(v_neg_reps.view(batch_size, -1, embedding_size).neg(),
                                 v_reps).sigmoid().log().view(batch_size, -1).sum(1)

        return -(edge_loss + u_pos_loss + v_pos_loss + u_noise_loss + v_noise_loss).mean()

    def train_batch(self, batch_data, device, max_patient, max_iter, optimizer, valid_interval,
                    valid_model_func, neighbors, data_iter, batch_i, valid_edges, h_delta):
        edges, walks, interaction_times, edge_types = batch_data
        edges = torch.LongTensor(edges)
        nodes_tensor = edges[:, [0, 1]].unique().to(device)
        edges = edges.to(device)
        walks = torch.LongTensor(walks).to(device)
        interaction_times = torch.LongTensor(interaction_times).to(device)
        edge_types = torch.LongTensor(edge_types).to(device)
        best_valid_mrr, patient = 0, 0
        epoch = 0
        best_epoch, best_mode_state_dict = 0, 0
        valid_mrr = None
        batch_nodes = self.update_nodes(edges)

        batch_size = edges.size(0)
        n_positive = walks[:, 0, :, :].view(batch_size, -1).size(1)
        repeat_edge_types = edges[:, 2].view(batch_size, 1).repeat(1, self.n_negative * n_positive).view(batch_size, -1)
        float_current_t = edges[:, 3].float()
        u_time_delta = (float_current_t - self.recent_timestamp[edges[:, 0]]) / self.time_agg
        v_time_delta = (float_current_t - self.recent_timestamp[edges[:, 1]]) / self.time_agg
        current_t = edges[:, 3].view(batch_size, -1)
        _, _, nw, wl = interaction_times.size()
        u_pos_time_delta = (current_t - interaction_times[:, 0, :, :].view(batch_size, -1)).float() / self.time_agg
        v_pos_time_delta = (current_t - interaction_times[:, 1, :, :].view(batch_size, -1)).float() / self.time_agg
        u_pos_reps_mask = self.decay_layer(u_pos_time_delta).view(batch_size, nw, wl)
        v_pos_reps_mask = self.decay_layer(v_pos_time_delta).view(batch_size, nw, wl)
        u_pos_reps_loss_mask = (u_pos_reps_mask > h_delta).float()
        v_pos_reps_loss_mask = (v_pos_reps_mask > h_delta).float()
        u_pos_reps_mask = torch.cumprod(u_pos_reps_mask, dim=2).view(batch_size, -1)
        v_pos_reps_mask = torch.cumprod(v_pos_reps_mask, dim=2).view(batch_size, -1)
        u_pos_reps_loss_mask = torch.cumprod(u_pos_reps_loss_mask, dim=2).view(batch_size, -1)
        v_pos_reps_loss_mask = torch.cumprod(v_pos_reps_loss_mask, dim=2).view(batch_size, -1)

        mask_max = max(u_pos_reps_mask.max(), v_pos_reps_mask.max())
        mask_min = min(u_pos_reps_mask.min(), v_pos_reps_mask.min())
        mask_avg = (u_pos_reps_mask.mean() + v_pos_reps_mask.mean()) / 2

        if valid_interval != 0:
            init_mrr = valid_model_func(valid_edges, self, neighbors, nodes_tensor, device)
            best_valid_mrr = init_mrr
            best_mode_state_dict = deepcopy(self.state_dict())
        else:
            init_mrr = 0.0
            best_epoch = max_iter

        while True:
            loss = self.loss(*self.forward(
                edges, walks, edge_types, batch_nodes, batch_size, n_positive, repeat_edge_types,
                u_time_delta, v_time_delta, u_pos_reps_mask, v_pos_reps_mask
                , u_pos_reps_loss_mask, v_pos_reps_loss_mask
            ))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if valid_interval != 0 and (valid_interval is None or (epoch + 1) % valid_interval == 0):
                valid_mrr = valid_model_func(valid_edges, self, neighbors, nodes_tensor, device)
                if best_valid_mrr > valid_mrr:
                    patient += 1
                    if patient > max_patient:
                        break
                else:
                    best_valid_mrr = valid_mrr
                    best_epoch = epoch + 1
                    best_mode_state_dict = deepcopy(self.state_dict())
                    patient = 0
            epoch += 1
            if max_iter is not None and epoch >= max_iter:
                break
        loss = loss.item()

        if valid_interval != 0:
            data_iter.write(
                f'[Batch %4d] [Epoch: %4d/%4d] [Loss: %.3f] [Valid mrr: %.3f -> %.3f] '
                f'[mask: %.3f, %.3f, %.3f] [ua: %s]' %
                (batch_i + 1, best_epoch, epoch, loss, init_mrr, valid_mrr or 0.0,
                 mask_min, mask_avg, mask_max, ', '.join(map(lambda x: '%.3f' % x.item(), self.updater.alpha))))
            self.load_state_dict(best_mode_state_dict)
        else:
            data_iter.write(
                f'[Batch %4d] [Epoch: %4d/%4d] [Loss: %.3f] [mask: %.3f, %.3f, %.3f] [ua: %s]' %
                (batch_i + 1, best_epoch, epoch, loss, mask_min, mask_avg, mask_max,
                 ', '.join(map(lambda x: '%.3f' % x.item(), self.updater.alpha))))
        self.update_times(edges)

# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_dense_batch
from ..neural_network import GATConvNet, GCNConvNet, ResNetBlock, MLPNet


class ActorCritic(nn.Module):
    
    def __init__(self, p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim=64, allow_revocable=False, allow_rejection=False):
        super(ActorCritic, self).__init__()
        self.encoder = Encoder(v_net_feature_dim, embedding_dim=embedding_dim)
        self.actor = Actor(p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim,allow_revocable, allow_rejection)
        self.critic = Critic(p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim)
        self._last_hidden_state = None

    def encode(self, obs):
        x = obs['v_net_x']
        outputs, hidden_state = self.encoder(x)
        self._last_hidden_state = hidden_state
        return outputs

    def act(self, obs):
        logits, outputs, hidden_state = self.actor(obs)
        self._last_hidden_state = hidden_state
        return logits

    def evaluate(self, obs):
        value = self.critic(obs)
        return value

    def get_last_rnn_state(self):
        return self._last_hidden_state

    def set_last_rnn_hidden(self, hidden_state):
        self._last_hidden_state = hidden_state


class Actor(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim=64,allow_revocable=False, allow_rejection=False):
        super(Actor, self).__init__()
        self.decoder = Decoder(p_net_num_nodes, p_net_feature_dim, embedding_dim=embedding_dim,allow_revocable=allow_revocable, allow_rejection=allow_rejection)

    def forward(self, obs):
        """Return logits of actions"""
        logits, outputs, hidden_state = self.decoder(obs)
        return logits, outputs, hidden_state


class Critic(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim=64):
        super(Critic, self).__init__()
        self.decoder = Decoder(p_net_num_nodes, p_net_feature_dim, embedding_dim=embedding_dim)

    def forward(self, obs):
        """Return logits of actions"""
        logits, outputs, hidden_state = self.decoder(obs)
        value = torch.mean(logits, dim=-1, keepdim=True)
        return value


class Encoder(nn.Module):

    def __init__(self, v_net_feature_dim, embedding_dim=64):
        super(Encoder, self).__init__()
        self.emb = nn.Linear(v_net_feature_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, embedding_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        embeddings = F.relu(self.emb(x))
        outputs, hidden_state = self.gru(embeddings)
        return outputs, hidden_state
    

class Decoder(nn.Module):
    
    def __init__(self, p_net_num_nodes, feature_dim, embedding_dim=64,allow_revocable=False, allow_rejection=False):
        super(Decoder, self).__init__()
        self.allow_revocable = allow_revocable
        self.num_actions = p_net_num_nodes + int(allow_revocable) + int(allow_rejection)
        self.emb = nn.Embedding(self.num_actions + 1, embedding_dim)
        self.att = Attention(embedding_dim)
        self.gcn = GCNConvNet(feature_dim, embedding_dim, embedding_dim=embedding_dim, dropout_prob=0., return_batch=True)
        #self.mlp = nn.Sequential(
        #    nn.Linear(embedding_dim, 1),
        #    nn.Flatten()
        #)
        # self.out = nn.Sequential(
        #     GCNConvNet(embedding_dim, 1, embedding_dim=embedding_dim, dropout_prob=0., return_batch=True),
        #     nn.Flatten(),
        # )
        self.output_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, self.num_actions)  # not per-node anymore
        )
        self.gru = nn.GRU(embedding_dim, embedding_dim)
        self._last_hidden_state = None

    def forward(self, obs):
        batch_p_net = obs['p_net']
        hidden_state = obs['hidden_state']
        p_node_embeddings = self.gcn(batch_p_net)
        p_node_embeddings = p_node_embeddings.reshape(batch_p_net.num_graphs, -1, p_node_embeddings.shape[-1])
        p_node_embeddings = p_node_embeddings + hidden_state
        #logits = self.mlp(p_node_embeddings)
        logits = self.output_head(hidden_state.squeeze(0))  # shape: (batch_size, num_actions)

        p_node_id = obs['p_node_id']
        hidden_state = hidden_state.permute(1, 0, 2)
        encoder_outputs = obs['encoder_outputs']
        mask = obs['mask']
        p_node_emb = self.emb(p_node_id).unsqueeze(0)
        context, attention = self.att(hidden_state, encoder_outputs, mask)
        context = context.unsqueeze(0)
        outputs, hidden_state = self.gru(p_node_emb, hidden_state)
        return logits, outputs, hidden_state
    

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden shape: (num_layers * num_directions, batch_size, hidden_dim)
        # encoder_outputs shape: (batch_size, seq_len, hidden_dim * num_directions)
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        hidden = hidden.transpose(0, 1).repeat(1, seq_len, 1)  # shape: (batch_size, seq_len, hidden_dim)
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))  # shape: (batch_size, seq_len, hidden_dim)
        attn_weights = F.softmax(self.v(energy).squeeze(2), dim=1)  # shape: (batch_size, seq_len)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e10)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # shape: (batch_size, 1, hidden_dim * num_directions)
        return context, attn_weights
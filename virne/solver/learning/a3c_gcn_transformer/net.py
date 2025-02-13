import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from ..neural_network import GATConvNet, GCNConvNet, ResNetBlock, MLPNet

###############################################################################
# net.py (Transformer-based version)
#
# This file replaces the original GRU-based Encoder/Decoder with a Transformer-
# based Encoder/Decoder, while preserving the overall class structure and API.
# 
# Key points:
#  1) Encoder now uses nn.TransformerEncoder (instead of GRU).
#  2) Decoder uses nn.TransformerDecoder (instead of GRU + custom attention).
#  3) We keep a GCN for the physical network representation.
#  4) The method signatures remain the same: `Encoder` returns (outputs, hidden),
#     and `Decoder` returns (logits, outputs, hidden_state).
###############################################################################


class ActorCritic(nn.Module):
    """
    High-level actor-critic model that holds:
      - Transformer-based Encoder (for v_net / SFC sequence).
      - Actor (Transformer-based Decoder + GCN).
      - Critic (Transformer-based Decoder + GCN).
    """
    def __init__(self, p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim=64,
                 n_heads=8, n_layers=3, dropout=0.1):
        super(ActorCritic, self).__init__()
        self.encoder = Encoder(v_net_feature_dim, embedding_dim=embedding_dim,
                               n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        self.actor = Actor(p_net_num_nodes, p_net_feature_dim, v_net_feature_dim,
                           embedding_dim=embedding_dim, n_heads=n_heads,
                           n_layers=n_layers, dropout=dropout)
        self.critic = Critic(p_net_num_nodes, p_net_feature_dim, v_net_feature_dim,
                             embedding_dim=embedding_dim, n_heads=n_heads,
                             n_layers=n_layers, dropout=dropout)

    def encode(self, obs):
        """
        obs: dict => {'v_net_x': tensor of shape (batch_size, seq_len, v_net_feature_dim)}
        Returns the Transformer encoder outputs and final hidden_state.
        """
        x = obs['v_net_x']  # (batch_size, seq_len, v_net_feature_dim)
        outputs = self.encoder(x)  # (batch_size, seq_len, emb), (1, batch_size, emb)
        return outputs

    def act(self, obs):
        """
        obs is a dictionary containing:
          - 'p_net' (PyG Batch)
          - 'p_node_id' (LongTensor of shape [batch_size])
          - 'hidden_state' (FloatTensor of shape [batch_size, embedding_dim])
          - 'encoder_outputs' (FloatTensor of shape [batch_size, seq_len, embedding_dim])
          - 'action_mask' (FloatTensor of shape [batch_size, ..., ...]) (optional use)
          - 'mask' (FloatTensor for padded seq)
        Returns: logits, outputs, hidden_state
        """
        logits, outputs = self.actor(obs)
        return logits

    def evaluate(self, obs):
        """
        Returns a scalar value estimate for the given obs.
        """
        value = self.critic(obs)
        return value



class Actor(nn.Module):
    """
    Actor part: uses the Transformer-based Decoder to produce action logits.
    """
    def __init__(self, p_net_num_nodes, p_net_feature_dim, v_net_feature_dim,
                 embedding_dim=64, n_heads=8, n_layers=3, dropout=0.1):
        super(Actor, self).__init__()
        self.decoder = Decoder(
            p_net_num_nodes, p_net_feature_dim,
            embedding_dim=embedding_dim,
            n_heads=n_heads, n_layers=n_layers, dropout=dropout
        )

    def forward(self, obs):
        """
        Return logits of actions.
        """
        logits, outputs = self.decoder(obs)
        return logits, outputs


class Critic(nn.Module):
    """
    Critic part: reuses the same Decoder module to produce a scalar state-value.
    """
    def __init__(self, p_net_num_nodes, p_net_feature_dim, v_net_feature_dim,
                 embedding_dim=64, n_heads=8, n_layers=3, dropout=0.1):
        super(Critic, self).__init__()
        self.decoder = Decoder(
            p_net_num_nodes, p_net_feature_dim,
            embedding_dim=embedding_dim,
            n_heads=n_heads, n_layers=n_layers, dropout=dropout
        )

    def forward(self, obs):
        """
        Return a scalar value (mean of the logits or another aggregator).
        """
        logits, outputs = self.decoder(obs)
        value = torch.mean(logits, dim=-1, keepdim=True)  # shape: (batch_size, 1)
        return value


class Encoder(nn.Module):
    """
    Transformer-based Encoder that replaces the GRU from the original code.
    """
    def __init__(self, v_net_feature_dim, embedding_dim=64, n_heads=8, n_layers=3, dropout=0.1):
        super(Encoder, self).__init__()
        # Project the VNF features to the embedding dimension
        self.emb = nn.Linear(v_net_feature_dim, embedding_dim)

        # Standard Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
            batch_first=True   # (batch_size, seq_len, emb)  <---- changed
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        """
        Args:
          x: shape (batch_size, seq_len, v_net_feature_dim)
        Returns:
          encoder_outputs: (batch_size, seq_len, embedding_dim)
          hidden_state: (batch_size, 1, embedding_dim) -> last token as summary
        """
        # (1) Embed & permute => (batch_size, seq_len, embedding_dim)
        x = self.emb(x)                # (batch_size, seq_len, embedding_dim)

        # (2) Run through the Transformer encoder
        encoder_outputs = self.transformer_encoder(x)  # => (batch_size, seq_len, emb)

        # (4) Return  (batch_size, seq_len, emb)
        return encoder_outputs


class Decoder(nn.Module):
    """
    Transformer-based Decoder that replaces the GRU+Attention from the original code.
    It also uses GCN to get physical node embeddings. Then merges them to produce logits.
    """
    def __init__(self, p_net_num_nodes, feature_dim,
                 embedding_dim=64, n_heads=8, n_layers=3, dropout=0.1):
        super(Decoder, self).__init__()

        # Embedding for the "action" tokens or placeholders
        self.emb = nn.Embedding(p_net_num_nodes + 1, embedding_dim)

        # GCN for physical network embeddings 
        self.gcn = GCNConvNet(feature_dim, embedding_dim, embedding_dim=embedding_dim, dropout_prob=0., return_batch=True)

        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # MLP to produce final logits (per node)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Flatten()  # shape => (batch_size, num_nodes)
        )

    def forward(self, obs):
        """
        obs: dictionary with:
          - p_net (Batch from PyG)
          - p_node_id (LongTensor [batch_size])
          - hidden_state (FloatTensor [batch_size, 1, embedding_dim])
          - encoder_outputs (FloatTensor [batch_size, seq_len, embedding_dim])
          - mask (FloatTensor [batch_size, seq_len]) (optional)
          - action_mask (FloatTensor [batch_size, p_net_num_nodes]) (unused here or can be used)
        Returns:
          logits (batch_size, num_p_nodes)
          outputs (batch_size, 1, embedding_dim)   # last decoder output
          hidden_state (batch_size, 1, embedding_dim)
        """
        # (1) Obtain physical network node embeddings from GCN
        batch_p_net = obs['p_net']  # PyG batch
        p_node_embeddings = self.gcn(batch_p_net)
        # reshape => (batch_size, num_p_nodes, emb)
        p_node_embeddings = p_node_embeddings.reshape(batch_p_net.num_graphs,
                                                      -1,
                                                      p_node_embeddings.shape[-1])

        # (3) Prepare the "target" token for Transformer Decoder
        p_node_id = obs['p_node_id']        # shape (batch_size,)
        p_node_emb = self.emb(p_node_id)    # => (batch_size, embedding_dim)
        # we do single-step decoding => shape => (batch_size, 1, embedding_dim)
        tgt = p_node_emb.unsqueeze(1) # shape (batch_size, 1, embedding_dim)

        # (4) TransformerDecoder expects memory => (batch_size, seq_len, emb)
        memory = obs['encoder_outputs']  # => (batch_size, seq_len, emb)

        # We can optionally build masks if needed
        # e.g., if 'mask' in obs, we might create a src_key_padding_mask or similar.
        tgt_mask = None
        memory_mask = None

        # (5) Pass through the Transformer Decoder
        # => shape (batch_size, 1, embedding_dim)
        dec_out = self.transformer_decoder(tgt, memory)
 
        # (7) Merge the decoder output with p_node_embeddings to produce logits.
        # A simple approach: add them (broadcasting) or do something else
        # We'll add them for demonstration:
        # => shape (batch_size, num_p_nodes, embedding_dim)
        combined = p_node_embeddings + dec_out

        # (8) MLP => shape (batch_size, num_p_nodes)
        logits = self.mlp(combined)

        # We define 'outputs' just as dec_out, and 'hidden_state' likewise
        outputs = dec_out  # (batch_size, 1, emb)
        
        return logits, outputs

 
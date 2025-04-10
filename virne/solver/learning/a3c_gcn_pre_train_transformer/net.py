# ==============================================================================
# net.py  (Enhanced Transformer with Deeper GAT and Improved Aggregation)
# ==============================================================================  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GENConv
from torch_geometric.utils import scatter

class MultiHeadGENLayer(nn.Module):
    def __init__(self, in_dim, out_dim, aggr='softmax', edge_dim=2):
        super().__init__()
        self.conv = GENConv(
            in_channels=in_dim,
            out_channels=out_dim,
            aggr=aggr,
            edge_dim=edge_dim,
            t=1.0,
            learn_t=True,
            num_layers=2
        )
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.2)
        #NEW 
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # Update node features
        out_x = self.conv(x, edge_index, edge_attr)
        out_x = self.norm(out_x)
        out_x = F.elu(out_x)
        out_x = self.dropout(out_x)

        # Compute edge features: concat endpoint node features
        row, col = edge_index
        edge_feat = torch.cat([out_x[row], out_x[col]], dim=1)
        out_e = self.edge_mlp(edge_feat)

        return out_x, out_e


# --- ActorCritic Model ---
class ActorCritic(nn.Module):
    def __init__(self, p_net_num_nodes, p_net_feature_dim, v_net_feature_dim,
                 embedding_dim=128, n_heads=8, n_layers=6, dropout=0.2, **kwargs):
        super().__init__()
        
        common_kwargs = dict(
            p_net_num_nodes=p_net_num_nodes,
            p_net_feature_dim=p_net_feature_dim,
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            **kwargs  # include allow_rejection, allow_revocable, etc.
        )
        
        max_seq_len = kwargs.get("max_seq_len", 15) 
        self.encoder = Encoder(v_net_feature_dim, embedding_dim, n_heads, n_layers, dropout, max_seq_len=max_seq_len)
        self.actor = Actor(**common_kwargs)
        self.critic = Critic(**common_kwargs)

    def encode(self, obs):
        return self.encoder(obs['v_net_x'])

    def act(self, obs ):
        return self.actor(obs)

    def evaluate(self, obs):
        return self.critic(obs)

# --- Encoder Module ---
class Encoder(nn.Module):
    def __init__(self, v_net_feature_dim, embedding_dim=128, n_heads=8, n_layers=6, dropout=0.2, max_seq_len=15):
        super().__init__()
        
        self.token_embed = nn.Linear(v_net_feature_dim, embedding_dim)
        self.position_embed = nn.Embedding(max_seq_len, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # Switch to GELU for better gradient flow
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embedding_dim)
        # In Encoder
        nn.init.xavier_uniform_(self.token_embed.weight, gain=1.0)  # Linear layers often use gain=1
        for layer in self.transformer_encoder.layers:
            nn.init.xavier_uniform_(layer.linear1.weight, gain=1.43)  # Approximate gain for GELU
            nn.init.zeros_(layer.linear1.bias)
            nn.init.xavier_uniform_(layer.linear2.weight, gain=1.0)
            nn.init.zeros_(layer.linear2.bias)

    def forward(self, x):  # x: [B, T, F]
        B, T, _ = x.size()
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)  # [B, T]
        x = self.token_embed(x) + self.position_embed(positions)               # [B, T, E]
        x = self.transformer_encoder(x)
        return self.norm(x)  # Optional final norm for stabilization

# --- Actor Module ---
class Actor(nn.Module):
    def __init__(self, p_net_num_nodes, p_net_feature_dim, embedding_dim=128,
                 n_heads=8, n_layers=6, dropout=0.2, **kwargs):
        super().__init__()
        # Retrieve special action flags from kwargs
        allow_revocable = kwargs.get("allow_revocable", False)
        allow_rejection = kwargs.get("allow_rejection", False)
        self.decoder = AutoregressiveDecoder(
            p_net_num_nodes=p_net_num_nodes,
            p_net_feature_dim=p_net_feature_dim,
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            is_actor=True, 
            allow_revocable=allow_revocable,
            allow_rejection=allow_rejection
        )

    def forward(self, obs):
        return self.decoder(obs)

# --- Critic Module ---
class Critic(nn.Module):
    def __init__(self, p_net_num_nodes, p_net_feature_dim, embedding_dim=128,
                 n_heads=8, n_layers=6, dropout=0.2, **kwargs):
        super().__init__()
        # Retrieve special action flags from kwargs
        allow_revocable = kwargs.get("allow_revocable", False)
        allow_rejection = kwargs.get("allow_rejection", False)
        self.decoder = AutoregressiveDecoder(
            p_net_num_nodes=p_net_num_nodes,
            p_net_feature_dim=p_net_feature_dim,
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            is_actor=False,
            allow_revocable=allow_revocable,      # ← pass these through
            allow_rejection=allow_rejection
        )
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, 1)
        )

    def forward(self, obs):
        last_emb = self.decoder(obs, return_last_embed=True)
        return self.value_head(last_emb)

# --- Autoregressive Decoder ---
class AutoregressiveDecoder(nn.Module):
    """Autoregressive Transformer Decoder for generating actions in a reinforcement learning environment. 
    Processes physical network graph features and historical actions to produce action logits or embeddings.
    Supports both actor (for action selection) and critic (for value estimation) modes.
    """
    def __init__(self, p_net_num_nodes, p_net_feature_dim, embedding_dim=128,
                 n_heads=8, n_layers=6, dropout=0.2, is_actor=True,
                 allow_revocable=False, allow_rejection=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Define special tokens and vocabulary size
        self.num_actions = p_net_num_nodes + int(allow_rejection) + int(allow_revocable)
        self.start_token = self.num_actions
        self.pad_token = self.num_actions + 1
        self.vocab_size = self.num_actions + 2  # Includes start and pad tokens

        # Embeddings
        self.rtg_embed = nn.Linear(1, embedding_dim)  # Embeds return-to-go scalar
        self.token_embed = nn.Embedding(self.vocab_size, embedding_dim)  # Embeds action tokens

        # Graph Attention Network (GAT) for physical network feature extraction
        self.gat_layers = nn.ModuleList([
            MultiHeadGENLayer(
                in_dim=p_net_feature_dim if i == 0 else embedding_dim,
                out_dim=embedding_dim
            )
            for i in range(3)
        ])
        self.gat_projection = nn.Identity()  # Optional projection layer

        # Transformer decoder configuration
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embedding_dim)

        # Output head for actor or critic
        self.is_actor = is_actor
        if is_actor:
            # NEW: Cross-attention decoder → nodes
            self.node_cross_attention = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=False
            )

            # NEW: Output head for attention-based fusion
            self.output_head = nn.Sequential(
                nn.Linear(2 * embedding_dim, embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, self.num_actions)
            )
        else:
            self.output_head = nn.Identity()  # Critic uses external value head

    def forward(self, obs, return_last_embed=False ):
        """Forward pass for the autoregressive decoder. 
        Args:
            obs (dict): Observation containing physical network, history actions, encoder outputs, and RTG.
            return_last_embed (bool): If True, returns the last decoder embedding for critic.
            return_failure_pred (bool): If True, returns logits for failure prediction (unused). 
        Returns:
            torch.Tensor: Action logits (B, num_actions) or last embedding (B, E) based on flags.
        """
        # Extract physical network features
        batch_p_net = obs['p_net']
        node_features = batch_p_net.x
        edge_index = batch_p_net.edge_index
        edge_attr = batch_p_net.edge_attr

        # Process graph with GAT layers
        edge_embeddings = []
        for gat_layer in self.gat_layers:
            node_features, edge_features = gat_layer(node_features, edge_index, edge_attr)
            edge_embeddings.append(edge_features)

        # Project node features and reshape
        graph_embedding = self.gat_projection(node_features) 
 
        # Embed history actions
        history_actions = obs['history_actions']  # Shape: (B, T)
        assert history_actions.max().item() < self.token_embed.num_embeddings, (
            f"Invalid action index {history_actions.max().item()} for embedding size "
            f"{self.token_embed.num_embeddings}"
        )
        action_embeddings = self.token_embed(history_actions)  # Shape: (B, T, E)

        # Add return-to-go (RTG) conditioning
        if 'rtg' not in obs:
            raise ValueError("Observation missing 'rtg' key")
        rtg = obs['rtg']  # Shape: (B, T, 1) or (B, T)
        if rtg.dim() == 2:
            rtg = rtg.unsqueeze(-1)
        rtg_embedding = self.rtg_embed(rtg)  # Shape: (B, T, E)

        # Combine action and RTG embeddings
        combined_target = action_embeddings + rtg_embedding  # Shape: (B, T, E)

        # Create causal and padding masks
        batch_size, seq_len = history_actions.shape
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=history_actions.device), diagonal=1
        ).bool()
        padding_mask = (history_actions == self.pad_token)

        # Decode with transformer
        decoder_output = self.transformer_decoder(
            tgt=combined_target,
            memory=obs['encoder_outputs'],
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask
        )  # Shape: (B, T, E)

        # Extract last time step for output
        last_decoder_output = decoder_output[:, -1, :]  # Shape: (B, E)
        
        # === EARLY EXIT FOR CRITIC ===
        if return_last_embed or not self.is_actor:
            #print("early critic exit!!!!!!!!!!!!!")
            return self.norm(last_decoder_output)
        # =============================

        # node_embeddings = graph_embedding from GAT (TotalNodes, E)
        node_batch = batch_p_net.batch  # shape: (TotalNodes,)
        B = batch_p_net.num_graphs

        # Reshape to group node embeddings per graph
        max_nodes = scatter(torch.ones_like(node_batch), node_batch, dim=0, reduce='sum').max().item()
        padded_nodes = torch.zeros(B, max_nodes, self.embedding_dim, device=graph_embedding.device)
        padding_mask = torch.ones(B, max_nodes, dtype=torch.bool, device=graph_embedding.device)

        for i in range(B):
            idxs = (node_batch == i).nonzero(as_tuple=True)[0]
            padded_nodes[i, :len(idxs)] = graph_embedding[idxs]
            padding_mask[i, :len(idxs)] = False

        # Transpose to match attention input: (Seq, B, E)
        key = value = padded_nodes.transpose(0, 1)  # (N_max, B, E)
        query = last_decoder_output.unsqueeze(0)    # (1, B, E)

        attn_output, _ = self.node_cross_attention(
            query=query, key=key, value=value,
            key_padding_mask=padding_mask
        )  # Output: (1, B, E)

        attn_context = attn_output.squeeze(0)  # (B, E)

        # Broadcast attended context to each node
        attn_context_per_node = attn_context[node_batch]  # (TotalNodes, E)

        # Combine: Node state + decoder context (from attention)
        combined = torch.cat([
            F.normalize(graph_embedding, dim=-1),
            F.normalize(attn_context_per_node, dim=-1)
        ], dim=-1)  # (TotalNodes, 2E)

        logits_per_node = self.output_head(combined)  # (TotalNodes, num_actions)
        logits = scatter(logits_per_node, node_batch, dim=0, reduce='mean')  # (B, num_actions)
        logits = torch.clamp(logits, min=-1e9, max=1e9)
 
        return logits
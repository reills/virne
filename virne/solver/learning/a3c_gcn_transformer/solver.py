# ==============================================================================
# solver.py (Refactored for Autoregressive Transformer)
# ==============================================================================
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Data, Batch
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence

from virne.base import Solution, SolutionStepEnvironment
from virne.solver import registry
from .instance_env import InstanceEnv
from .net import ActorCritic
from virne.solver.learning.rl_base import RLSolver, PPOSolver, InstanceAgent, A2CSolver, RolloutBuffer
from ..utils import get_pyg_data


@registry.register(
    solver_name='a3c_gcn_transformer', 
    env_cls=SolutionStepEnvironment,
    solver_type='r_learning')
class A3CGcnTransformerSolver(InstanceAgent, A2CSolver):
    """
    A Reinforcement Learning-based solver that uses 
    Advantage Actor-Critic (A3C) as the training algorithm,
    with a Transformer-based architecture.
    """
    def __init__(self, controller, recorder, counter, **kwargs):
        A2CSolver.__init__(self, controller, recorder, counter, make_policy, obs_as_tensor, **kwargs)
        InstanceAgent.__init__(self, InstanceEnv)
        
        # new hyperparams
        self.entropy_coef = kwargs.get("entropy_coef", 0.01)
        self.normalize_advantage = kwargs.get("normalize_advantage", True)
 
        # Decide on a special start token (or padding token).
        # We'll treat p_net.num_nodes as <start_token>, and p_net.num_nodes+1 as <pad_token>, etc
        self.start_token_offset = 0  # We’ll just use p_net.num_nodes as start token.

        self.preprocess_encoder_obs = encoder_obs_to_tensor


    def update(self):
        """
        Overridden to add advantage normalization + entropy term.
        """
        # Collect batch data
        obs_tensors = self.preprocess_obs(self.buffer.observations, self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        returns = torch.FloatTensor(self.buffer.returns).to(self.device)

        # Evaluate
        values, action_logprobs, dist_entropy, _ = self.evaluate_actions(obs_tensors, actions, return_others=True)

        # advantage
        advantages = returns - values.detach()
        if self.normalize_advantage and advantages.numel() > 0:
            # add small epsilon for stability
            eps = 1e-8
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # standard actor-critic style
        actor_loss = - (action_logprobs * advantages).mean()
        critic_loss = F.mse_loss(returns, values)
        # add entropy better exploration
        loss = actor_loss + self.coef_critic_loss * critic_loss - self.entropy_coef * dist_entropy.mean()

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        info = {
            "loss_total": loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": dist_entropy.mean().item()
        }
        if self.verbose >= 1:
            print(f"[Update] {info}")

        self.buffer.clear()
        self.update_time += 1

    def evaluate_actions(self, obs, actions, return_others=False):
        """
        Evaluate the log-prob of the chosen actions + value + entropy
        to build the training losses.
        """
        logits = self.policy.act(obs)  # shape (B, p_net_num_nodes)
        values = self.policy.evaluate(obs).squeeze(-1)

        dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        if return_others:
            return values, action_logprobs, dist_entropy, {}
        else:
            return values, action_logprobs, dist_entropy


    def solve(self, instance):
        """Inference-only solve: uses a greedy or sample-based approach with the learned policy."""
        v_net, p_net = instance['v_net'], instance['p_net']
        sub_env = self.InstanceEnv(p_net, v_net, self.controller, self.recorder, self.counter, **self.basic_config)

        # 1) Encode the virtual network once. 
        encoder_obs = sub_env.get_observation()
        encoder_outputs = self.policy.encode(self.preprocess_encoder_obs(encoder_obs, device=self.device))

        # 2) Maintain a partial action sequence for the Transformer.
        #    We use [start_token] = p_net.num_nodes if you like. 
        #    Each step we append the new action.
        history_actions = [p_net.num_nodes + self.start_token_offset]  # <start_token>

        instance_done = False
        while not instance_done:
            # Create the observation dict for the next decision:
            instance_obs = {
                'p_net_x'           : encoder_obs['p_net_x'],
                'p_net_edge_index'  : encoder_obs['p_net_edge_index'],
                'history_actions'   : np.array(history_actions, dtype=np.int64),  # shape (t,)
                'encoder_outputs'   : encoder_outputs,
                'action_mask'       : np.expand_dims(sub_env.generate_action_mask(), axis=0)
            }
            tensor_instance_obs = self.preprocess_obs(instance_obs, device=self.device)

            # Select next action
            action, action_logprob = self.select_action(tensor_instance_obs, sample=True)
            next_obs, reward, instance_done, info = sub_env.step(action)

            # Append chosen action to the partial sequence
            history_actions.append(action)

            if instance_done:
                break

            encoder_obs = next_obs  # If your v_net_x changes each step, re-encode if needed.

        return sub_env.solution

    def learn_with_instance(self, instance):
        """Collect one trajectory (sub-episode) from this instance and store it in the buffer."""
        sub_buffer = RolloutBuffer()
        v_net, p_net = instance['v_net'], instance['p_net']
        sub_env = self.InstanceEnv(p_net, v_net, self.controller, self.recorder, self.counter, **self.basic_config)

        # Encode once
        encoder_obs = sub_env.get_observation()
        encoder_outputs = self.policy.encode(self.preprocess_encoder_obs(encoder_obs, device=self.device))

        # Start partial sequence with a <start_token>
        history_actions = [p_net.num_nodes + self.start_token_offset]

        instance_done = False
        while not instance_done:
            # Build observation
            instance_obs = {
                'p_net_x'           : encoder_obs['p_net_x'],
                'p_net_edge_index'  : encoder_obs['p_net_edge_index'],
                'history_actions'   : np.array(history_actions, dtype=np.int64),
                'encoder_outputs'   : encoder_outputs,
                'action_mask'       : np.expand_dims(sub_env.generate_action_mask(), axis=0)
            }
            tensor_instance_obs = self.preprocess_obs(instance_obs, device=self.device)

            # Pick action
            action, action_logprob = self.select_action(tensor_instance_obs, sample=True)
            value = self.estimate_value(tensor_instance_obs) if hasattr(self.policy, 'evaluate') else None

            # Step env
            next_obs, reward, instance_done, info = sub_env.step(action)

            # Store rollout info
            sub_buffer.add(instance_obs, action, reward, instance_done, action_logprob, value=value)

            # Update partial sequence
            history_actions.append(action)

            # Move on
            if not instance_done:
                encoder_obs = next_obs

        # Final value for GAE or advantage bootstrapping
        last_value = 0.
        if hasattr(self.policy, 'evaluate'):
            next_tensor_obs = self.preprocess_obs(
                {
                    'p_net_x'          : encoder_obs['p_net_x'],
                    'p_net_edge_index' : encoder_obs['p_net_edge_index'],
                    'history_actions'  : np.array(history_actions, dtype=np.int64),
                    'encoder_outputs'  : encoder_outputs,
                    'action_mask'      : np.expand_dims(sub_env.generate_action_mask(), axis=0)
                },
                device=self.device
            )
            last_value = self.estimate_value(next_tensor_obs)

        return sub_env.solution, sub_buffer, last_value


def make_policy(agent, **kwargs):
    # Same as before, but we will finalize the critic in net.py with a value head.
    action_dim = agent.p_net_setting_num_nodes
    feature_dim = agent.p_net_setting_num_node_resource_attrs + agent.p_net_setting_num_link_resource_attrs + 5

    policy = ActorCritic(
        p_net_num_nodes=action_dim,
        p_net_feature_dim=5,
        v_net_feature_dim=2,
        embedding_dim=agent.embedding_dim
    ).to(agent.device)

    optimizer = torch.optim.Adam([
        {'params': policy.encoder.parameters(), 'lr': agent.lr_actor},
        {'params': policy.actor.parameters(),   'lr': agent.lr_actor},
        {'params': policy.critic.parameters(),  'lr': agent.lr_critic}
    ], weight_decay=agent.weight_decay)
    return policy, optimizer


def encoder_obs_to_tensor(obs, device):
    """Process the SFC/v_net features for the Transformer Encoder."""
    if isinstance(obs, dict):
        v_net_x = obs['v_net_x']
        # shape => (seq_len, v_net_feature_dim)
        obs_v_net_x = torch.FloatTensor(v_net_x).unsqueeze(0).to(device) 
        return {'v_net_x': obs_v_net_x}
    elif isinstance(obs, list):
        # Handling a batch of obs
        obs_batch = []
        for o in obs:
            v_net_x = torch.FloatTensor(o['v_net_x']).to(device)
            obs_batch.append(v_net_x)

        # Pad them along seq dimension
        # [BatchSize, MaxSeqLen, FeatureDim]
        padded = pad_sequence(obs_batch, batch_first=True)
        return {'v_net_x': padded}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")

def obs_as_tensor(obs, device):
    """Preprocess the entire observation (including partial action sequence) into tensor form."""
    if isinstance(obs, dict):
        # Single sample
        data = get_pyg_data(obs['p_net_x'], obs['p_net_edge_index'])
        obs_p_net = Batch.from_data_list([data]).to(device)

        # The full partial action sequence (history)
        history_actions = obs['history_actions']  # shape (t,)
        history_actions = torch.LongTensor(history_actions).unsqueeze(0).to(device)  # => (1, t)

        obs_encoder_outputs = torch.as_tensor(obs['encoder_outputs'], dtype=torch.float32, device=device)
        obs_action_mask     = torch.as_tensor(obs['action_mask'],     dtype=torch.float32, device=device)

        return {
            'p_net'           : obs_p_net,
            'history_actions' : history_actions,  # shape (1, t)
            'encoder_outputs' : obs_encoder_outputs,  # shape (1, seq_len, emb)
            'action_mask'     : obs_action_mask,
            'mask'            : None
        }

    elif isinstance(obs, list):
        # Batch of samples
        p_net_data_list      = []
        history_actions_list = []
        encoder_outputs_list = []
        action_mask_list     = []

        for o in obs:
            data = get_pyg_data(o['p_net_x'], o['p_net_edge_index'])
            p_net_data_list.append(data)

            ha = torch.LongTensor(o['history_actions'])
            history_actions_list.append(ha)
            encoder_outputs_list.append(o['encoder_outputs'])
            action_mask_list.append(o['action_mask'])

        obs_p_net = Batch.from_data_list(p_net_data_list).to(device)

        # Pad history_actions
        hist_padded = pad_sequence(history_actions_list, batch_first=True, padding_value=0).to(device)

        # Convert encoder outputs to a single padded tensor
        # Suppose each is (1, seq_len, emb). We'll find max seq_len among them
        # for simplicity. We'll do a naive approach:
        batch_size  = len(encoder_outputs_list)
        max_seq_len = max(eo.shape[1] for eo in encoder_outputs_list)
        emb_dim     = encoder_outputs_list[0].shape[2]

        enc_padded = torch.zeros((batch_size, max_seq_len, emb_dim), dtype=torch.float32, device=device)
        for i, eo in enumerate(encoder_outputs_list):
            slen = eo.shape[1]
            enc_padded[i, :slen, :] = torch.as_tensor(eo, device=device)
 
        action_mask_np = np.array(action_mask_list)  # Convert list of arrays to a single NumPy array
        act_mask_t = torch.as_tensor(action_mask_np, dtype=torch.float32, device=device)  # Convert to tensor



        return {
            'p_net'           : obs_p_net,
            'history_actions' : hist_padded,       # (batch_size, max_hist_len)
            'encoder_outputs' : enc_padded,        # (batch_size, max_seq_len, emb_dim)
            'action_mask'     : act_mask_t,
            'mask'            : None
        }

    else:
        raise ValueError('obs type error: expected dict or list')

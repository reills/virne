# ==============================================================================
# instance_env.py
# ==============================================================================
import numpy as np
import networkx as nx
import os
import warnings
import torch
from gym import spaces
from virne.solver.learning.rl_base import JointPRStepInstanceRLEnv, PlaceStepInstanceRLEnv

_norm_vector_p = None
_norm_vector_v = None
class InstanceEnv(JointPRStepInstanceRLEnv):

    def __init__(self, p_net, v_net, controller, recorder, counter, **kwargs):
        super(InstanceEnv, self).__init__(p_net, v_net, controller, recorder, counter, **kwargs)
        num_p_net_node_attrs = len(self.p_net.get_node_attrs(['resource', 'extrema']))
        num_p_net_link_attrs = len(self.p_net.get_link_attrs(['resource', 'extrema']))
        num_p_net_features = num_p_net_node_attrs + 1
         
        # Use module-level cache
        global _norm_vector_p, _norm_vector_v
        # Set normalization paths
        #dataset_path = "/home/stephen-reilly/dev/virne/dataset/training-data/merged_training_data.pt"
        dataset_path = "/Users/stephenreilly/Desktop/github/virne/virne/solver/learning/a3c_gcn_pre_train_transformer/merged_training_data.pt"
        if _norm_vector_p is None or _norm_vector_v is None:
            print("Loading normalization vectors...")

            if os.path.exists(dataset_path):
                print(f"Computing normalization vectors from: {dataset_path}")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*") 
                    data = torch.load(dataset_path)

                # Expect each item in data to be a dict with 'p_net_x' and 'v_net_x'
                p_samples = torch.cat([s['p_net_x'].clone().detach() if isinstance(s['p_net_x'], torch.Tensor) else torch.tensor(s['p_net_x']) for s in data], dim=0)
                v_samples = torch.cat([s['v_net_x'].clone().detach() if isinstance(s['v_net_x'], torch.Tensor) else torch.tensor(s['v_net_x']) for s in data], dim=0)

                _norm_vector_p = torch.max(p_samples, dim=0)[0].float()
                _norm_vector_v = torch.max(v_samples, dim=0)[0].float()
                print("Normalization vectors computed.")
            else:
                raise FileNotFoundError(f"No pretrained dataset found.\n"
                                        f"Expected one of:\n  {dataset_path}")

        self.norm_vector_p = _norm_vector_p
        self.norm_vector_v = _norm_vector_v
        
        self.max_revokes=40 

    def compute_reward(self, solution, revoke=False):
        """Per-step reward to encourage full VNet acceptance and discourage poor routing or excessive revoke."""

        vnodes = self.v_net.num_nodes            # Typically 8
        placements = solution['num_placed_nodes']
        revokes = solution['revoke_times']
        revoke_pct = revokes / max(1, self.max_revokes)

        # Case 1: Early rejection — should almost never happen, very bad
        if solution['early_rejection']:
            reward = -vnodes * 1.5  # Scaled: -1.5   Penalty applied at the step where early reject happens
        # Case 2: Revoke was taken — penalize each time it happens
        elif revoke:
            reward = -(.5 + 0.1 * revokes)  # Increasing penalty per revoke step
        # Case 3: Final step of a full success — high reward, scaled with revoke penalty
        elif solution['result']:
            # This will be called once at the last vnode placement step
            penalty = revoke_pct * vnodes
            bonus = vnodes - penalty
            reward = vnodes + bonus  # e.g., 8 + bonus
        # Case 4: Routing partially succeeded — some virtual nodes got routed
        elif solution['route_result']:
            # Encourage partial progress in routing
            penalty = placements * revoke_pct
            bonus = placements - penalty
            reward = placements + bonus
        # Case 5: All nodes placed, but routing failed entirely
        elif solution['place_result']:
            reward = -vnodes * 0.75  # Scaled: -0.75 # Mild penalty; worse than partial route, better than total failure
        # Case 6: Total failure (e.g., placement failed)
        else:
            reward = -vnodes  # Large negative signal
        # Normalize by number of vnodes to make reward scale consistent across different topologies
        scaled_reward = reward / vnodes

        self.solution['v_net_reward'] += scaled_reward
        return scaled_reward

    
    def get_observation(self):
        """
        Returns the observation as a dictionary containing:
            - p_net_x and its edge index and edge attributes.
            - v_net_x and its edge index.
            - action_mask and history_actions. 
        """
        p_net_obs = self._get_p_net_obs()
        v_net_obs = self._get_v_net_obs()
        history_actions = self.get_history_actions()
        return {
            'p_net_x': p_net_obs['x'], 
            'p_net_edge_index': p_net_obs['edge_index'],
            'edge_attr': p_net_obs['edge_attr'],
            'v_net_x': v_net_obs['x'],
            'v_net_edge_index': v_net_obs['edge_index'],
            'action_mask': self.generate_action_mask(),
            'history_actions': history_actions,
        }

    def _get_v_net_obs(self):
        """
        Build v_net_x with four components:
        1. Resource features from the virtual network (using obs_handler)
        2. Virtual node degrees (computed using obs_handler)
        3. Virtual total capacity (sum of resource features)
        4. Total required bandwidth (sum of bandwidth demands from connected edges)
        5. Betweenness centrality
        6. Bandwidth variance
        """
        if self.curr_v_node_id >= self.v_net.num_nodes:
            return {'x': np.empty((0, 4), dtype=np.float32)}
        
        # Resource features
        resource_features = self.obs_handler.get_node_attrs_obs(
            self.v_net, 
            node_attr_types=['resource'], 
            node_attr_benchmarks=self.node_attr_benchmarks
        )
        
        # Virtual node degrees
        degrees = self.obs_handler.get_node_degree_obs(
            self.v_net, 
            self.obs_handler.get_degree_benchmark(self.v_net)
        )
        
        # Total capacity
        total_capacity = np.sum(resource_features, axis=1, keepdims=True)
        
        # Total required bandwidth (sum of 'bw' demands from connected edges)
        # Get edge demands similar to load_vnet_demands
        edge_demands = {
            tuple(sorted([str(u), str(v)])): float(data.get('bw', 0.0))
            for u, v, data in self.v_net.edges(data=True)
        }
        v_nodes = list(self.v_net.nodes())
        v_bw_agg = np.array([
            sum(edge_demands.get(tuple(sorted([str(v), str(adj)])), 0.0) 
                for adj in self.v_net.adj[v])
            for v in v_nodes
        ]).reshape(-1, 1)
        
        # Betweenness centrality
        bc_dict = nx.betweenness_centrality(self.v_net)
        bc = np.array([bc_dict[node] for node in range(self.v_net.num_nodes)], 
                    dtype=np.float32).reshape(-1, 1)

        # Bandwidth variance
        bw_var = []
        for node in range(self.v_net.num_nodes):
            demands = [self.v_net.links[(node, neighbor)]['bw'] 
                    for neighbor in self.v_net.adj[node]]
            bw_var.append(np.var(demands) if demands else 0.0)
        bw_var = np.array(bw_var, dtype=np.float32).reshape(-1, 1)

        # Concatenate features: all should be 2D arrays now
        v_net_x = np.hstack([resource_features, degrees, total_capacity, v_bw_agg, bc, bw_var])

        # Get virtual network edge index
        v_net_edge_index = self.obs_handler.get_link_index_obs(self.v_net)
        
        #normalize with dataset
        v_net_x = v_net_x / self.norm_vector_v.cpu().numpy()  # convert torch -> numpy

        return {'x': v_net_x, 'edge_index': v_net_edge_index}

    def _get_p_net_obs(self):
        """
        Updated to include extended physical features:
        1. Available resources
        2. Degree
        3. Total capacity
        4. Aggregated bandwidth
        5. Betweenness
        6. Neighbor bandwidth average
        7. Node utilization
        8. Avg link utilization
        9. Link variance
        10. Critical links
        """
        # Resource features
        resource_features = self.obs_handler.get_node_attrs_obs(
            self.p_net, 
            node_attr_types=['resource'], 
            node_attr_benchmarks=self.node_attr_benchmarks
        )
        
        # Physical node degrees
        p_degrees = self.obs_handler.get_node_degree_obs(
            self.p_net, 
            self.obs_handler.get_degree_benchmark(self.p_net)
        )
        
        # Total available capacity
        p_total_capacity = np.sum(resource_features, axis=1, keepdims=True)
        
        # Aggregated available bandwidth (sum of 'bw' from connected edges)
        p_bw_agg = self.obs_handler.get_link_aggr_attrs_obs(
            self.p_net, 
            link_attr_types=['resource'], 
            aggr='sum', 
            link_sum_attr_benchmarks=self.obs_handler.get_link_sum_attr_benchmarks(self.p_net)
        )
        
        # 5. Betweenness centrality for physical network
        bc_dict = nx.betweenness_centrality(self.p_net)
        bc = np.array([bc_dict[node] for node in range(self.p_net.num_nodes)], dtype=np.float32).reshape((-1, 1))

        # 6. Neighbor bandwidth average: for each node, average the 'bw' over its neighboring edges
        neighbor_bw_avg = []
        for node in range(self.p_net.num_nodes):
            neighbors = list(self.p_net.adj[node])
            if neighbors:
                bw_values = []
                for nb in neighbors:
                    # Assumes each edge has a 'bw' attribute
                    bw_values.append(self.p_net.links[(node, nb)]['bw'])
                neighbor_bw_avg.append(np.mean(bw_values))
            else:
                neighbor_bw_avg.append(0.0)
        neighbor_bw_avg = np.array(neighbor_bw_avg, dtype=np.float32).reshape((-1, 1))

        # 7. Node utilization (average % of resources used)
        node_utilization = self.obs_handler.get_node_utilization_obs(self.p_net)
        
        # 8. Average link utilization per node
        p_avg_link_util = self.obs_handler.get_avg_link_utilization_obs(self.p_net)
        
        # 9 & 10. Global link features
        link_variance, critical_links = self.obs_handler.get_link_utilization_global_obs(self.p_net)
        link_variance_feature = np.full((self.p_net.num_nodes, 1), link_variance, dtype=np.float32)
        critical_links_feature = np.full((self.p_net.num_nodes, 1), critical_links, dtype=np.float32)
        
        # Combine features
        p_net_x = np.hstack([
            resource_features,
            p_degrees,
            p_total_capacity,
            p_bw_agg,
            bc,
            neighbor_bw_avg,
            node_utilization,
            p_avg_link_util,
            link_variance_feature,
            critical_links_feature
        ])
        
        # Edge index for the physical network
        edge_index = self.obs_handler.get_link_index_obs(self.p_net)

        # Compute edge attributes; here we use the obs_handler function to get (normalized) link attributes.
        edge_attr = self.obs_handler.get_link_attrs_obs(
            self.p_net,
            link_attr_types=['resource'],
            link_attr_benchmarks=self.obs_handler.get_link_attr_benchmarks(self.p_net)
        )
          
        # Normalize
        p_net_x = p_net_x / self.norm_vector_p.cpu().numpy()

        return {
            'x': p_net_x,
            'edge_index': edge_index,
            'edge_attr': edge_attr
        }

    def get_history_actions(self):
        """
        Return the sequence of previously placed physical node IDs as int64 array.
        This is used by the transformer decoder.
        """
        return np.array(self.selected_p_net_nodes, dtype=np.int64)


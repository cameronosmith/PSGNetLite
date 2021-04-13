from abc import ABC, abstractmethod

import torch
from torch_geometric.nn    import max_pool_x
from torch_geometric.utils import add_self_loops
from torch_scatter         import scatter_mean
from torch_sparse          import coalesce

from LabelProp import LP_clustering
from VAE       import VAE

# Module performs affinity clustering on a graph and label propagation 
# clustering to coarsen the graph
class AffinityConditionedAggregation(torch.nn.Module, ABC):

    # Takes in tensor of node pairs and returns an affinity tensor and a 
    # threshold tensor to filter the affinites with. Also returns any loss
    # items to pass back to the training layer as a dict.
    # x is the list of graph nodes and row, col are the tensors of the adj. list
    @abstractmethod
    def affinities_and_thresholds(self, x, row, col):
        pass

    # Filters edge index based on base method's affinity thresholding and 
    # coarsens graph to produce next level's nodes
    def forward(self, x, edge_index, batch, device="cuda"):

        row, col = edge_index

        # Collect affinities/thresholds to filter edges 

        affinities, threshold, losses = self.affinities_and_thresholds(x,row,col)
        filtered_edge_index = edge_index[:, affinities > threshold]
        filtered_edge_index = add_self_loops(filtered_edge_index,
                                             num_nodes=x.size(0))[0].to(device)

        # Coarsen graph with filtered adj. list to produce next level's nodes

        node_labels    = LP_clustering(x.size(0), filtered_edge_index, 40)
        cluster_labels = node_labels.unique(return_inverse=True,sorted=False)[1]

        coarsened_x, coarsened_batch = max_pool_x(cluster_labels, x, batch)
        coarsened_edge_index = coalesce(cluster_labels[filtered_edge_index],
                              None, coarsened_x.size(0), coarsened_x.size(0))[0]

        return (coarsened_x, coarsened_edge_index, coarsened_batch,
                                                         cluster_labels, losses)

class P2AffinityAggregation(AffinityConditionedAggregation):

    def __init__(self, node_feat_size, v2=3.5 ):
        super().__init__()

        self.v2 = v2
        self.node_pair_vae = VAE( in_features=2*node_feat_size )

    # Note question to ask: why reconstructing difference of nodes versus
    # concatenating them, as many different node pairs can have similar
    # differences? Is it just to make it symmetric? Try both
    # Currently using the concatenation but can switch to difference below.

    def affinities_and_thresholds(self, x, row, col):

        # Affinities as function of vae reconstruction of node pairs
        #_, recon_loss, kl_loss = self.node_pair_vae( torch.abs(x[row]-x[col]) )
        _, recon_loss, kl_loss = self.node_pair_vae( torch.cat((x[row],x[col]),dim=1) )
        edge_affinities = 1/(1 + self.v2*recon_loss )

        losses = {"recon_loss":recon_loss.mean(), "kl_loss":kl_loss.mean()}

        return edge_affinities, .5, losses

class P1AffinityAggregation(AffinityConditionedAggregation):

    # P1 is a zero-parameter affinity clustering algorithm which operates on
    # the similarity of features
    def affinities_and_thresholds(self, x, row, col):

        # Norm of difference for every node pair on grid
        edge_affinities = torch.linalg.norm( x[row]-x[col], dim=1).to(x.device)

        # Inverse mean affinities for each node to threshold each edge with
        inv_mean_affinity = 1/scatter_mean(edge_affinities, row.to(x.device))
        affinity_thresh   = torch.min(inv_mean_affinity[row],
                                      inv_mean_affinity[col])

        return edge_affinities, affinity_thresh, {}

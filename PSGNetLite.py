from types import SimpleNamespace

import torch

from torch_geometric.nn    import max_pool_x, GraphConv
from torch_geometric.data  import Data,Batch
from torch_geometric.utils import grid, to_dense_batch

from pytorch_prototyping.pytorch_prototyping import FCBlock

from RDN import RDN

from Affinities import P1AffinityAggregation, P2AffinityAggregation

class PSGNetLite(torch.nn.Module):
    def __init__(self,imsize):

        super().__init__()

        node_feat_size   = 32
        num_graph_layers = 2

        self.spatial_edges,self.spatial_coords = grid(imsize,imsize,device="cuda")

        # Conv. feature extractor to map pixels to feature vectors
        self.rdn = RDN(SimpleNamespace(G0=node_feat_size-2,RDNkSize=3,n_colors=3,
                               RDNconfig=(4,3,16),scale=[2],no_upsampling=True))

        # Affinity modules: for now just one of P1 and P2 
        self.affinity_aggregations = torch.nn.ModuleList([
            P1AffinityAggregation(), P2AffinityAggregation(node_feat_size)
        ])

        # Node transforms: function applied on aggregated node vectors
        self.node_transforms = torch.nn.ModuleList([
            FCBlock(hidden_ch=128,
                    num_hidden_layers=3,
                    in_features =node_feat_size,
                    out_features=node_feat_size,
                    outermost_linear=True)
                                 for _ in range(len(self.affinity_aggregations))
        ])

        # Graph convolutional layers to apply after each graph coarsening
        self.graph_convs = torch.nn.ModuleList([
            GraphConv(node_feat_size, node_feat_size)
                                 for _ in range(len(self.affinity_aggregations))
        ])

        # Maps cluster vector to constant pixel color
        self.node_to_rgb  = FCBlock(hidden_ch=128,
                                    num_hidden_layers=3,
                                    in_features =node_feat_size,
                                    out_features=3,
                                    outermost_linear=True)

    def forward(self,img):

        # Collect image features with rdn

        im_feats = self.rdn(img.permute(0,3,1,2))

        coords_added_im_feats = torch.cat([
                  self.spatial_coords.unsqueeze(0).repeat(im_feats.size(0),1,1),
                  im_feats.flatten(2,3).permute(0,2,1)
                                          ],dim=2)

        ### Run image feature graph through affinity modules

        graph_in = Batch.from_data_list([Data(x,self.spatial_edges)
                                                for x in coords_added_im_feats])

        x, edge_index, batch = graph_in.x, graph_in.edge_index, graph_in.batch

        clusters, all_losses = [], [] # clusters just used for visualizations

        for pool, conv, transf in zip(self.affinity_aggregations,
                                      self.graph_convs, self.node_transforms):
            batch_uncoarsened = batch

            x, edge_index, batch, cluster, losses = pool(x, edge_index, batch)
            x = conv(x, edge_index)
            x = transf(x)

            clusters.append( (cluster, batch_uncoarsened) )
            all_losses.append(losses)

        # Render into image (just constant coloring from each cluster for now)
        # First map each node to pixel and then uncluster pixels

        nodes_rgb = self.node_to_rgb(x)
        img_out   = nodes_rgb
        for cluster,_ in reversed(clusters): img_out = img_out[cluster]

        return to_dense_batch(img_out,graph_in.batch)[0], clusters, all_losses

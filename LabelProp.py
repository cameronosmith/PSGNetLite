# This file just contains a label propagation clustering function

import torch
from torch_sparse  import SparseTensor
from torch_scatter import scatter_max

# Takes in number of nodes, the graph's edge index (adj. list), number of LP
# iterations, and returns the label for each node 
def LP_clustering(num_nodes,edge_index,edge_weight,num_iter=10,device="cuda"):

    # Sparse adjacency matrix
    adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
           sparse_sizes=(num_nodes, num_nodes),value=edge_weight).t().to(device)

    # Each node starts with its index as its label
    x = SparseTensor.eye(num_nodes).to(device)

    # Run LP for num_iter iterations
    for _ in range(num_iter):

        # Add each node's neighbors' labels to its list of neighbor nodes
        out = adj_t @ x

        # Argmax of each row to assign new label to each node
        row, col, value = out.coo()
        argmax = scatter_max(value, row, dim_size=num_nodes)[1]
        new_labels = col[argmax]

        x = SparseTensor(row=torch.arange(num_nodes).to(device), col=new_labels,
                            sparse_sizes=(num_nodes, num_nodes),
                            value=torch.ones(num_nodes).to(device))

    return new_labels

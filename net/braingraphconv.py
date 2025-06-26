import torch
import torch.nn.functional as F
from torch.nn import Parameter
from net.brainmsgpassing import MyMessagePassing
from torch_geometric.utils import add_remaining_self_loops,softmax

from torch_geometric.typing import (OptTensor)

from net.inits import uniform


class MyNNConv(MyMessagePassing):
    def __init__(self, in_channels, out_channels, nn, normalize=False, bias=True,
                 **kwargs):
        super(MyNNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.nn = nn
        #self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
#        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, edge_weight=None, pseudo= None, size=None):
        """"""
        edge_weight = edge_weight.squeeze()
        if size is None and torch.is_tensor(x):
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, 1, x.size(0))


        if pseudo is None:
            pseudo = torch.zeros((edge_index.size(1), self.nn[0].in_features), device=x.device)

        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)

        # Propagate raw node features and per-edge weight matrices. The actual
        # feature transformation (x_j * W_e) will be done inside `message`,
        # ensuring that the tensor fed to `scatter` has the correct number of
        # elements (one per edge), while the tensor used for indexing (`x`) is
        # still of size `[num_nodes, in_channels]`.

        return self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight, weight=weight)

    def message(self, edge_index_i, size_i, x_j, edge_weight, weight, ptr: OptTensor):
        # Apply the learned, edge-specific weight matrices to the source-node
        # features.
        x_j = torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

        edge_weight = softmax(edge_weight, edge_index_i, ptr, size_i)
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


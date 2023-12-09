# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch_geometric.nn.dense import dense_diff_pool
from torch_geometric.utils import to_dense_adj

class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)
        
        # initialize the assignment vector, for 8x8 patches, we have 64 patches
        # print("in channel size:", in_channels)
        # 320
        self.s = nn.Parameter(torch.rand(512, 16 * 16, 8 * 8))  # hardcoding 512 batch size for now

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()            
        # print("before reshaping x.shape:", x.shape)
        # before reshaping x.shape: torch.Size([512, 320, 4, 4])
        x = x.reshape(B, C, -1, 1).contiguous()
        # print("after reshaping x.shape:", x.shape)
        # after reshaping x.shape: torch.Size([512, 320, 16, 1])
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        # print("DygraphConv2d edge.shape:", edge_index.shape)
        # DygraphConv2d edge_index.shape: torch.Size([2, 512, 16, 16])
        

        
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        # print("resulting x shape, after graph conv:", x.shape)
        # x = x.squeeze(-1) # squeeze the last dimension so now x is B x F x N (512 * 640 * 256)
        # # permute x so that it is B x N x F (512 * 16 * 640)
        # x = x.permute(0, 2, 1).contiguous()
        


        # resulting x shape, after graph conv: torch.Size([512, 640, 16, 1])
        # print("reshaping x:", x.reshape(B, -1, H, W).contiguous().shape)
        # reshaping x: torch.Size([512, 640, 16, 16])




        # edge_index: (2, batch_size, num_points, k)
        # given batch_size of such instance, and the number of points, there are k neighbors for each point in the batch. 
        # compute the adjacency matrix for each batch

        # get the one dimensional batch_size * num_points * k batch vector which indicates what batch each edge belongs to
        
        # _, batch_size, num_points, k = edge_index.shape

        # batch_vector = torch.arange(batch_size).unsqueeze(1).unsqueeze(1).repeat(1, num_points, k).reshape(-1).to(x.device)
        # print("batch_vector size:", batch_vector.shape)

        # flatten the edge_index tensor to 2 dimensional, following the order by batch_size, num_points, k
        # edge_index = edge_index.reshape(2, -1)
        # print("edge_index shape:", edge_index.shape)
        
        # adj = to_dense_adj(edge_index, batch_vector, batch_size=batch_size)
        # print("DygraphConv2d adj.shape:", adj.shape)
        
        # somehow apply dense_diff_pool here?

        # pool the graph into 8x8 patches based on the assignment vector
        # turn into [512, 640, 64, 1] tensor

        # turn x back
        # x = x.unsqueeze(-1)

        # print("x shape after dense_diff_pool:", x.shape)
        # print("reshaping x:", x.reshape(B, -1, H, W).contiguous().shape)
    
        return x.reshape(B, -1, H, W).contiguous()


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x
        # print("grapher module x.shape:", x.shape)
        # grapher module x.shape: torch.Size([512, 320, 4, 4])
        x = self.fc1(x)
        # print("grapher module x.shape after fc1:", x.shape)
        # grapher module x.shape after fc1: torch.Size([512, 320, 4, 4])
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x, relative_pos)
        # print("grapher module x.shape after graph_conv:", x.shape) 
        # grapher module x.shape after graph_conv: torch.Size([512, 640, 4, 4])

        x = self.fc2(x)
        # print("grapher module x.shape after fc2:", x.shape)
        # grapher module x.shape after fc2: torch.Size([512, 320, 4, 4])

        x = self.drop_path(x) + _tmp
        return x

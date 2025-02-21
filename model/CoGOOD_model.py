import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv

class GCN(torch.nn.Module):
    def __init__(self, out_channels, aggr='mean'):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(384, 384, aggr=aggr)
        self.conv2 = GraphConv(384, 384, aggr=aggr)
        self.conv3 = GraphConv(384, 128, aggr=aggr)
        self.linear2 = torch.nn.Linear(384, out_channels)
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.dropout(x)
        y = F.relu(self.conv2(x, edge_index, edge_weight))
        # x = self.dropout(x)
        # y = F.relu(self.conv3(x, edge_index, edge_weight))
        #y = F.relu(self.conv3(x, edge_index, edge_weight))
        #x = self.dropout(y)
        x = F.sigmoid(self.linear2(y))
        return x, y

class MLP(torch.nn.Module):
    def __init__(self, out_channels):
        super(MLP, self).__init__()
        self.conv1 = GraphConv(384, 384)
        self.conv2 = GraphConv(384, 384)
        self.conv3 = GraphConv(64, 32)
        self.linear1 = torch.nn.Linear(384, 128)
        self.linear2 = torch.nn.Linear(128, 64)
        self.linear3 = torch.nn.Linear(384, out_channels)
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # x = F.relu(self.linear1(x))
        # x = self.dropout(x)
        # y = F.relu(self.linear2(x))
        # x = self.dropout(y)
        y = x
        x = F.sigmoid(self.linear3(x))
        return x, y
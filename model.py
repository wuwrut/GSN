import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn

class TypeInferModel(nn.Module):
    def __init__(self, embedding_num):
        super(TypeInferModel, self).__init__()

        self.embedding = nn.Embedding(embedding_num, 128)

        self.c1 = tgnn.GCNConv(128, 256)
        self.c2 = tgnn.GCNConv(256, 512)
        #self.c3 = tgnn.SGConv(512, 512)
        #self.c4 = tgnn.SGConv(512, 512)

        self.norm1 = nn.BatchNorm1d(256)
        self.norm2 = nn.BatchNorm1d(512)
        #self.norm3 = nn.BatchNorm1d(512)
        #self.norm4 = nn.BatchNorm1d(512)

        self.dense1 = nn.Linear(512, 4096)
        self.dense2 = nn.Linear(4096, 6)

    def forward(self, x, a):
        emb = self.embedding(x)

        emb = torch.squeeze(emb)
        a = torch.squeeze(a)

        y = self.c1(emb, a)
        y = self.norm1(y)
        y = F.leaky_relu(y)

        y = self.c2(y, a)
        y = self.norm2(y)
        y = F.leaky_relu(y)

        '''
        y1 = self.c3(y, a)
        y1 = self.norm3(y1)
        y1 = F.leaky_relu(y1)

        y1 = self.c4(y1, a)
        y = y + y1
        y = self.norm4(y)
        y = F.leaky_relu(y)
        '''

        y = self.dense1(y)
        y = F.leaky_relu(y)
        y = self.dense2(y)

        return y

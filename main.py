import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn


class TypeInferModel(nn.Module):
    def __init__(self):
        super(TypeInferModel, self).__init__()

        self.c1 = tgnn.GCNConv(128, 256)
        self.c2 = tgnn.GCNConv(256, 512)

        self.norm1 = nn.BatchNorm1d(256)
        self.norm2 = nn.BatchNorm1d(512)

        self.dense1 = nn.Linear(512, 4096)
        self.dense2 = nn.Linear(4096, 6)

    def forward(self, x, a):
        y = self.c1(x, a)
        y = self.norm1(y)
        y = F.leaky_relu(y)

        y = self.c2(y, a)
        y = self.norm2(y)
        y = F.leaky_relu(y)

        y = self.dense1(y)
        y = F.leaky_relu(y)
        y = self.dense2(y)

        return F.softmax(y, dim=1)


device = torch.device('cpu')#'cuda' if torch.cuda.is_available() else 'cpu')
model = TypeInferModel()#.to(device)

x = torch.rand((10, 128))
a = torch.tensor([[0,1,2,3,4,5,6,7,8,9],
                  [1,2,3,4,5,6,7,8,9,0]])

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
cross_loss = nn.CrossEntropyLoss()

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(x, a)

    predicted_types = out[0:2,:]
    real_types = torch.tensor([[1,0,0,0,0,0],[0,0,0,1,0,0]])

    loss = cross_loss(predicted_types, real_types)
    loss.backward()
    optimizer.step()
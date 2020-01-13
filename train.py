import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import GraphLoader
from model import TypeInferModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TypeInferModel(128).train().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

dataset = GraphLoader('train.pckl')
loader = DataLoader(dataset, 1, shuffle=True, pin_memory=True)

for epoch in range(3):
    for (a, x), (yi, yv) in loader:
        a, x = a.to(device), x.to(device)
        yi, yv = yi.to(device), yv.to(device)
        out = model(x, a)

        predicted = out[torch.squeeze(yi)].view(-1, 6)
        real = torch.squeeze(yv).view(-1)

        optimizer.zero_grad()
        loss = F.cross_entropy(predicted, real)#, reduction='sum')
        loss.backward()
        optimizer.step()

        print("epoch: {}, loss: {}".format(epoch, loss))


torch.save(model.state_dict(), 'trained_model')

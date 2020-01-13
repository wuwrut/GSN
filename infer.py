import torch
from torch.utils.data import DataLoader
from model import TypeInferModel
from dataset import GraphLoader


device = torch.device('cpu')#'cuda' if torch.cuda.is_available() else 'cpu')
model = TypeInferModel(128)
model.load_state_dict(torch.load('gcnconv_model_mean_loss'))

for p in model.parameters():
    p.requires_grad = False

model = model.eval().to(device)

dataset = GraphLoader('test.pckl')
loader = DataLoader(dataset, 1)

correct_nodes = 0
total_nodes = 0
correct_var_types = 0
total_vars = 0

for (a, x), (yi, yv) in loader:
    a, x = a.to(device), x.to(device)
    yi, yv = yi.to(device), yv.to(device)
    out = model(x, a)

    predicted = torch.argmax(out[torch.squeeze(yi)].view(-1, 6), dim=1)
    real = torch.squeeze(yv).view(-1)

    real_var_mask = x[0, torch.squeeze(yi)] < 20
    real_var_indices = torch.masked_select(torch.squeeze(yi), real_var_mask)

    predicted_var_types = torch.argmax(out[real_var_indices].view(-1, 6), dim=1)
    real_var_types = torch.masked_select(torch.squeeze(yv), real_var_mask)

    correct_nodes += torch.sum(predicted == real)
    total_nodes += predicted.shape[0]

    correct_var_types += torch.sum(predicted_var_types == real_var_types)
    total_vars += real_var_types.shape[0]

print("Accuracy (all nodes): {} ({}/{})".format(correct_nodes / float(total_nodes), correct_nodes, total_nodes))
print("Accuracy (variables): {} ({}/{})".format(correct_var_types / float(total_vars), correct_var_types, total_vars))

a = 6
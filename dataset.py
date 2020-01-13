import torch
import torch.nn as nn
import torch.utils.data as data
import torch_geometric.data as gdata
import pickle


class GraphLoader(data.Dataset):
    def __init__(self, filename):
        with open(filename, 'rb') as f:
            (As, Xs, Ys, node_lut, type_lut, var_lut) = pickle.load(f)

        self.As = [torch.tensor(A, dtype=torch.long).t() for A in As]
        self.Xs = [torch.tensor(X, dtype=torch.long) for X in Xs]

        indexes, values = [], []

        for y in Ys:
            tmp1, tmp2 = [], []

            for (k, v) in y.items():
                tmp1.append(k)
                tmp2.append(v if v <= type_lut['any'] else type_lut['any'])

            indexes.append(torch.tensor(tmp1))
            values.append(torch.tensor(tmp2))

        self.Y_indexes = indexes
        self.Y_vals = values
        #self.emb = embedding_dict

    def __len__(self):
        return len(self.As)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self.As[idx], self.Xs[idx]), (self.Y_indexes[idx], self.Y_vals[idx])

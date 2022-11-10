import torch
import torch.nn as nn


class R_Embedding(nn.Module):
    def __init__(self, dataset, parameter):
        super(R_Embedding, self).__init__()
        self.device = parameter['device']
        self.es = parameter['embed_dim']
        self.rel2id = dataset['rel2id']

        num_rel = len(self.rel2id)
        self.norm_vector = nn.Embedding(num_rel, self.es*self.es)

        identity = torch.zeros(self.es, self.es)
        for i in range(min(self.es, self.es)):
            identity[i][i] = 1
        identity = identity.view(self.es * self.es)
        for i in range(num_rel):
            self.norm_vector.weight.data[i] = identity

    def forward(self, triples):
        rel_emb = [[[self.rel2id[t[1]]] for t in batch] for batch in triples]
        rel_emb = torch.LongTensor(rel_emb).to(self.device)
        return self.norm_vector(rel_emb)

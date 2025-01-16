
import torch

class GCNConv(torch.nn.Module):
    def __init__(
            self, 
            input_dim, 
            output_dim, 
            learn_weight=True,
            layer_norm=False, 
            add_self=False, 
            normalize_embedding=False,
            dropout=0.0,
            activation='relu', 
            bias=True
        ):
        super(GCNConv,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learn_weight = learn_weight
        self.layer_norm = layer_norm
        self.add_self = add_self
        self.normalize_embedding = normalize_embedding
        self.dropout = dropout
        self.activation = activation

        if self.learn_weight:
            self.fc = torch.nn.Linear(input_dim, output_dim, bias=bias)
        else:
            self.output_dim = input_dim
        
        if dropout > 0.0:
            self.dropout_layer = torch.nn.Dropout(p=dropout)
        
        if self.layer_norm:
            self.layer_norm_layer = torch.nn.LayerNorm(output_dim)

        if self.activation=='relu':
            self.act_layer=torch.nn.ReLU()
        elif self.activation=='lrelu':
            self.act_layer=torch.nn.LeakyReLU(0.1)
        else:
            self.act_layer=torch.nn.Identity()

    def forward(self, x, adj, mask):
        # y = torch.matmul(adj, x)
        y = torch.bmm(adj, x)
        if self.add_self:
            y += x
        if self.learn_weight:
            y = self.fc(y)
        if self.normalize_embedding:
            y = torch.nn.functional.normalize(y, p=2, dim=2)
        if self.layer_norm:
            y = self.layer_norm_layer(y)
        if self.dropout > 0.0:
            y = self.dropout_layer(y)
        
        y = self.act_layer(y)

        return y
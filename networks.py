import torch
import torch.nn as nn

#  if game in ['Tandem', 'Balduzzi', 'Hamiltonian']:
class PGNet(nn.Module):
    def __init__(self, hyper_params):
        super(self).__init__()
        input_dim, output_dim = hyper_params["input_dim"], hyper_params["output_dim"]
        self.net = torch.nn.Sequential(torch.nn.Linear(input_dim, hyper_params['num_nodes']),
            torch.nn.ReLU(),
            torch.nn.Linear(hyper_params['num_nodes'], hyper_params['num_nodes']),
            torch.nn.Linear(hyper_params['num_nodes'], output_dim),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, y):
        return self.net(y)

#  else
class NonPGNet(nn.Module):
    def __init__(self, hyper_params):
        super(self).__init__()
        input_dim, output_dim = hyper_params["input_dim"], hyper_params["output_dim"]
        self.net = torch.nn.Sequential(torch.nn.Linear(input_dim, hyper_params['num_nodes']),
            torch.nn.Tanh(),
            torch.nn.Linear(hyper_params['num_nodes'], hyper_params['num_nodes']),
            torch.nn.Tanh(),
            torch.nn.Linear(hyper_params['num_nodes'], hyper_params['num_nodes']),
            torch.nn.Tanh(),
            torch.nn.Linear(hyper_params['num_nodes'], output_dim),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, y):
        return self.net(y)
import numpy as np
import torch
from torch.nn import CosineSimilarity

from cola.networks import NonPGNet, PGNet
from cola.utils import load_checkpoint, update_th
from cola.hparams import get_hparams
from cola.games import (
    ultimatum, tandem, matching_pennies, chicken_game,
    ipd, hamiltonian_game, balduzzi,
)

game = 'Matching Pennies'  # ["Matching Pennies", "IPD", "Ultimatum", "Chicken Game", "Tandem", "Balduzzi", "Hamiltonian"]

hyper_params = get_hparams(game)
interval = hyper_params['interval']
network_type = PGNet if game in ['Tandem', 'Balduzzi', 'Hamiltonian'] else NonPGNet

if game == 'Ultimatum':
    dims, Ls = ultimatum()
elif game == 'Tandem':
    dims, Ls = tandem()
elif game == 'Matching Pennies':
    dims, Ls = matching_pennies()
elif game == 'Chicken Game':
    dims, Ls = chicken_game()
elif game == 'IPD':
    dims, Ls = ipd(hyper_params['gamma'])
elif game == 'Hamiltonian':
    dims, Ls = hamiltonian_game()
elif game == 'Balduzzi':
    dims, Ls = balduzzi()

# Load trained COLA networks (produced by exp_training.py)
k_net_long = network_type(hyper_params=hyper_params)
h_net_long = network_type(hyper_params=hyper_params)
k_net_long = load_checkpoint('k_net_long.pth', k_net_long)
h_net_long = load_checkpoint('h_net_long.pth', h_net_long)

# ## Calculating similarity between COLA and HOLA

cos = CosineSimilarity(dim=0, eps=1e-15)
alpha = hyper_params['alpha']
beta = hyper_params['beta']

r1 = -interval
r2 = interval

num_samples = 1000
for algo in ['hola1', 'hola4', 'hola6', 'sos']:
    total_cola_dot = 0
    for m in range(num_samples):
        theta_x = (r1 - r2) * torch.rand(hyper_params['output_dim'], requires_grad=True) + r2
        theta_y = (r1 - r2) * torch.rand(hyper_params['output_dim'], requires_grad=True) + r2
        betas = torch.ones((1, 1)) * beta
        th = [theta_x, theta_y]

        # COLA
        if hyper_params['input_dim'] == 10:
            inp = torch.cat([theta_x.unsqueeze(0), theta_y.unsqueeze(0)], dim=1)
        else:
            inp = torch.cat([theta_x.unsqueeze(0), theta_y.unsqueeze(0)], dim=1)

        x_comp = k_net_long(inp).squeeze(0)
        y_comp = h_net_long(inp).squeeze(0)

        x_comp_cola = x_comp
        y_comp_cola = y_comp
        cola_cons = torch.cat([x_comp_cola, y_comp_cola])

        if algo == 'hola1':
            _, _, grads_lola = update_th(th=th, Ls=Ls, alpha=alpha, algo='higher_order_lola', order=1, beta=beta)
            x_comp = -grads_lola[0]
            y_comp = -grads_lola[1]

        elif algo == 'hola4':
            _, _, grads_hola2 = update_th(th=th, Ls=Ls, alpha=alpha, algo='higher_order_lola', order=2, beta=beta)
            x_comp = -grads_hola2[0]
            y_comp = -grads_hola2[1]

        elif algo == 'hola6':
            _, _, grads_hola3 = update_th(th=th, Ls=Ls, alpha=alpha, algo='higher_order_lola', order=4, beta=beta)
            x_comp = -grads_hola3[0]
            y_comp = -grads_hola3[1]

        else:
            _, _, grads_sos = update_th(th=th, Ls=Ls, alpha=alpha, algo=algo, beta=beta)
            x_comp = -grads_sos[0]
            y_comp = -grads_sos[1]

        cons_grad = torch.cat([x_comp, y_comp])

        total_cola_dot += cos(cola_cons, cons_grad)

    total_cola_dot /= num_samples

    print('Dot Product COLA ' + algo + ': ', total_cola_dot.item())
    print('\n')

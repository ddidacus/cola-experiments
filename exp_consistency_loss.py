import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns
from torch.nn import CosineSimilarity

from .networks import NonPGNet, PGNet
from .utils import get_gradient, init_th, load_checkpoint, update_th
from .hparams import get_hparams
from .games import (
    ultimatum, tandem, matching_pennies, matching_pennies_batch,
    chicken_game, chicken_game_batch, ipd, ipd_batched,
    hamiltonian_game, balduzzi,
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

# ## Calculating the Consistency Loss over an area

cos = CosineSimilarity(dim=0, eps=1e-15)

alpha = hyper_params['alpha']
beta = hyper_params['beta']

r1 = -interval
r2 = interval

num_samples = 100
for beta in [beta]:  # 0.02, 0.03, 0.04, 0.05, 0.06, 0.07
    print('Beta: ', beta)
    for algo in ['cola', 'hola1', 'hola2', 'hola3', 'sos', 'cgd']:
        total_loss_dot = 0
        total_loss_norm = 0
        total_loss = 0
        total_loss_squared_diff = 0
        total_cola_dot = 0
        for m in range(num_samples):
            theta_x = (r1 - r2) * torch.rand(hyper_params['output_dim'], requires_grad=True) + r2
            theta_y = (r1 - r2) * torch.rand(hyper_params['output_dim'], requires_grad=True) + r2
            betas = torch.ones((1, 1)) * beta
            th = [theta_x, theta_y]

            if algo == 'cola':
                if hyper_params['input_dim'] == 10:
                    inp = torch.cat([theta_x.unsqueeze(0), theta_y.unsqueeze(0)], dim=1)
                else:
                    inp = torch.cat([theta_x.unsqueeze(0), theta_y.unsqueeze(0)], dim=1)

                th_prime = [th[0], th[1]]

                x_comp = k_net_long(inp).squeeze(0)
                y_comp = h_net_long(inp).squeeze(0)
                th_delta_y = [th[0], th[1] + (beta) * y_comp]
                th_delta_x = [th[0] + (beta) * x_comp, th[1]]
                loss_delta_y = Ls(th_delta_y)[0]
                loss_delta_x = Ls(th_delta_x)[1]
                incons_gradX_loss_delta_y = -get_gradient(loss_delta_y, th[0])
                incons_gradY_loss_delta_x = -get_gradient(loss_delta_x, th[1])

                x_comp_cola = x_comp
                y_comp_cola = y_comp
                cola_cons = torch.cat([x_comp_cola, y_comp_cola])

            elif algo == 'hola1':
                _, _, grads_lola = update_th(th=th, Ls=Ls, alpha=alpha, algo='higher_order_lola', order=1, beta=beta)
                x_comp = -grads_lola[0]
                y_comp = -grads_lola[1]
                th_delta_y = [th[0], th[1] + beta * y_comp]
                th_delta_x = [th[0] + beta * x_comp, th[1]]
                loss_delta_y = Ls(th_delta_y)[0]
                loss_delta_x = Ls(th_delta_x)[1]
                incons_gradX_loss_delta_y = -get_gradient(loss_delta_y, th[0])
                incons_gradY_loss_delta_x = -get_gradient(loss_delta_x, th[1])

            elif algo == 'hola2':
                _, _, grads_hola2 = update_th(th=th, Ls=Ls, alpha=alpha, algo='higher_order_lola', order=2, beta=beta)
                x_comp = -grads_hola2[0]
                y_comp = -grads_hola2[1]
                th_delta_y = [th[0], th[1] + beta * y_comp]
                th_delta_x = [th[0] + beta * x_comp, th[1]]
                loss_delta_y = Ls(th_delta_y)[0]
                loss_delta_x = Ls(th_delta_x)[1]
                incons_gradX_loss_delta_y = -get_gradient(loss_delta_y, th[0])
                incons_gradY_loss_delta_x = -get_gradient(loss_delta_x, th[1])

            elif algo == 'hola3':
                _, _, grads_hola3 = update_th(th=th, Ls=Ls, alpha=alpha, algo='higher_order_lola', order=4, beta=beta)
                x_comp = -grads_hola3[0]
                y_comp = -grads_hola3[1]
                th_delta_y = [th[0], th[1] + beta * y_comp]
                th_delta_x = [th[0] + beta * x_comp, th[1]]
                loss_delta_y = Ls(th_delta_y)[0]
                loss_delta_x = Ls(th_delta_x)[1]
                incons_gradX_loss_delta_y = -get_gradient(loss_delta_y, th[0])
                incons_gradY_loss_delta_x = -get_gradient(loss_delta_x, th[1])

            else:
                _, _, grads_sos = update_th(th=th, Ls=Ls, alpha=alpha, algo=algo, beta=beta)
                x_comp = -grads_sos[0]
                y_comp = -grads_sos[1]
                th_delta_y = [th[0], th[1] + beta * y_comp]
                th_delta_x = [th[0] + beta * x_comp, th[1]]
                loss_delta_y = Ls(th_delta_y)[0]
                loss_delta_x = Ls(th_delta_x)[1]
                incons_gradX_loss_delta_y = -get_gradient(loss_delta_y, th[0])
                incons_gradY_loss_delta_x = -get_gradient(loss_delta_x, th[1])

            incons_grad = torch.cat([incons_gradX_loss_delta_y, incons_gradY_loss_delta_x])
            cons_grad = torch.cat([x_comp, y_comp])

            incons_grad_norm = F.normalize(incons_grad, dim=0)
            cons_grad_norm = F.normalize(cons_grad, dim=0)

            total_loss_dot += cos(cons_grad, incons_grad)
            total_loss_norm += ((cons_grad_norm - incons_grad_norm) ** 2).sum()
            total_loss += torch.norm(cons_grad - incons_grad)
            total_loss_squared_diff += ((cons_grad - incons_grad) ** 2).sum()

        total_loss_dot /= num_samples
        total_loss_norm /= num_samples
        total_loss /= num_samples
        total_loss_squared_diff /= num_samples

        print('Total Loss DOT ' + algo + ': ', total_loss_dot.item())
        print('Total Loss (normalized updates) ' + algo + ': ', total_loss_norm.item())
        print('Total Loss ' + algo + ': ', total_loss.item())
        print('Total Loss Squared Diff ' + algo + ': ', total_loss_squared_diff.item())

        print('\n')

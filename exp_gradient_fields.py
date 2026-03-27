import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from .networks import NonPGNet, PGNet
from .utils import get_gradient, load_checkpoint, update_th
from .hparams import get_hparams
from .cola import find_local_min_colav2
from .games import (
    matching_pennies, matching_pennies_batch,
    ultimatum, tandem, chicken_game, ipd, hamiltonian_game, balduzzi,
)

sns.set_theme(style="ticks")
fontsize = 30
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

# ## Gradient fields for LOLA and COLA

grain = 20
dims_batch, Ls_batch = matching_pennies_batch(batch_size=grain ** 2)
x_comps, y_comps, errors, th = find_local_min_colav2(
    Ls=Ls_batch, grain=grain, interval=interval,
    hyper_params=hyper_params, k_net_long=k_net_long, h_net_long=h_net_long)

x_comps_cola_pt = x_comps
y_comps_cola_pt = y_comps
x_comps_cola = x_comps.detach().numpy()
y_comps_cola = y_comps.detach().numpy()
errors_cola = errors.detach().numpy()

x, y = np.meshgrid(np.linspace(-interval, interval, grain), np.linspace(-interval, interval, grain))

plt.quiver(x, y, x_comps_cola, y_comps_cola)
plt.xlabel('Agent 1', fontsize=fontsize, labelpad=20)
plt.ylabel('Agent 2', fontsize=fontsize, labelpad=10)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title(f"COLA on {game}", fontsize=fontsize, pad=20)
plt.show()
plt.clf()

ticks = np.linspace(-interval, interval, grain).round(2)

ax = sns.heatmap(errors_cola, annot=True, linewidth=0.5, xticklabels=ticks, yticklabels=ticks)
plt.title("Errors")
plt.show()
plt.clf()

ax = sns.heatmap(x_comps_cola, annot=True, linewidth=0.5, xticklabels=ticks, yticklabels=ticks)
plt.title("X Components")
plt.show()
plt.clf()

ax = sns.heatmap(y_comps_cola, annot=True, linewidth=0.5, xticklabels=ticks, yticklabels=ticks)
plt.title("Y Components")
plt.show()
plt.clf()

# LOLA gradient field
alpha = hyper_params['alpha']
beta = hyper_params['beta']
algorithm = 'higher_order_lola'

x_comps = torch.zeros([grain, grain], dtype=torch.float)
y_comps = torch.zeros([grain, grain], dtype=torch.float)
errors = torch.zeros([grain, grain], dtype=torch.float)
lspace = torch.linspace(-interval, interval, grain, requires_grad=True)

for i in range(grain):
    for j in range(grain):
        theta_0 = lspace[i].unsqueeze(-1)
        theta_1 = lspace[j].unsqueeze(-1)
        th = [theta_0, theta_1]
        _, _, grads = update_th(th=th, Ls=Ls, alpha=alpha, algo=algorithm, order=1, beta=beta)
        x_comp = -grads[0]
        y_comp = -grads[1]
        i_index, j_index = int(i), int(j)
        x_comps[j_index, i_index] = x_comp
        y_comps[j_index, i_index] = y_comp

        th_delta_y = [th[0], th[1] + alpha * y_comp]
        th_delta_x = [th[0] + alpha * x_comp, th[1]]
        loss_delta_y = Ls(th_delta_y)[0]
        loss_delta_x = Ls(th_delta_x)[1]
        incons_gradX_loss_delta_y = -get_gradient(loss_delta_y, th[0])
        incons_gradY_loss_delta_x = -get_gradient(loss_delta_x, th[1])
        incons_grad = torch.cat([incons_gradX_loss_delta_y, incons_gradY_loss_delta_x])
        cons_grad = torch.cat([x_comp, y_comp])
        total_loss = torch.norm(F.normalize(cons_grad, dim=0) - F.normalize(incons_grad, dim=0))
        errors[j_index, i_index] = total_loss

x, y = np.meshgrid(np.linspace(-interval, interval, grain), np.linspace(-interval, interval, grain))

x_comps_lola_pt = x_comps
y_comps_lola_pt = y_comps
x_comps_lola = x_comps.detach().numpy()
y_comps_lola = y_comps.detach().numpy()
errors_lola = errors.detach().numpy()
plt.streamplot(x, y, x_comps_lola, y_comps_lola, color='r', linewidth=1,
               density=1, arrowstyle='->', arrowsize=1)
plt.figure(figsize=(15, 8))
plt.quiver(x, y, x_comps_lola, y_comps_lola)
plt.title('LOLA')
plt.show()
plt.clf()

# HOLA4 gradient field
alpha = 10
beta = 10
algorithm = 'higher_order_lola'

x_comps = torch.zeros([grain, grain], dtype=torch.float)
y_comps = torch.zeros([grain, grain], dtype=torch.float)
errors = torch.zeros([grain, grain], dtype=torch.float)
lspace = torch.linspace(-interval, interval, grain, requires_grad=True)

for i in range(grain):
    for j in range(grain):
        theta_0 = lspace[i].unsqueeze(-1)
        theta_1 = lspace[j].unsqueeze(-1)
        th = [theta_0, theta_1]
        _, _, grads = update_th(th=th, Ls=Ls, alpha=alpha, beta=beta, algo=algorithm, order=4)
        x_comp = -grads[0]
        y_comp = -grads[1]
        i_index, j_index = int(i), int(j)
        x_comps[j_index, i_index] = x_comp
        y_comps[j_index, i_index] = y_comp

        th_delta_y = [th[0], th[1] + beta * y_comp]
        th_delta_x = [th[0] + beta * x_comp, th[1]]
        loss_delta_y = Ls(th_delta_y)[0]
        loss_delta_x = Ls(th_delta_x)[1]
        incons_gradX_loss_delta_y = -get_gradient(loss_delta_y, th[0])
        incons_gradY_loss_delta_x = -get_gradient(loss_delta_x, th[1])
        incons_grad = torch.cat([incons_gradX_loss_delta_y, incons_gradY_loss_delta_x])
        cons_grad = torch.cat([x_comp, y_comp])
        total_loss = torch.norm(F.normalize(cons_grad, dim=0) - F.normalize(incons_grad, dim=0))
        errors[j_index, i_index] = total_loss

x, y = np.meshgrid(np.linspace(-interval, interval, grain), np.linspace(-interval, interval, grain))

x_comps_hola6_pt = x_comps
y_comps_hola6_pt = y_comps
x_comps_hola6 = x_comps.detach().numpy()
y_comps_hola6 = y_comps.detach().numpy()
errors_hola6 = errors.detach().numpy()

plt.figure(figsize=(15, 8))
plt.quiver(x, y, x_comps_hola6, y_comps_hola6)
plt.xlabel('Agent 1', fontsize=fontsize, labelpad=20)
plt.ylabel('Agent 2', fontsize=fontsize, labelpad=10)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title(f"HOLA4 on {game}", fontsize=fontsize, pad=20)
plt.show()
plt.clf()

x_comps_hola6_flat = x_comps_hola6_pt.reshape(-1, 1)
y_comps_hola6_flat = y_comps_hola6_pt.reshape(-1, 1)
print(x_comps_hola6_flat.size())

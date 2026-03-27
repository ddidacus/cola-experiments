import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import torch.optim as optim
import torch.nn as nn
import seaborn as sns
from scipy import stats
from torch.nn import CosineSimilarity

from .networks import NonPGNet, PGNet
from .utils import init_th, load_checkpoint, smooth, update_th
from .hparams import get_hparams
from .cola import cola_v2, nn_batched_pretrain_colav2
from .games import (
    ultimatum, tandem, matching_pennies, matching_pennies_batch,
    chicken_game, chicken_game_batch, ipd, ipd_batched,
    hamiltonian_game, balduzzi,
)

cos = CosineSimilarity(dim=1, eps=1e-15)
markers = ['o', 'v', '^', '*', 's', 'D']
colors = ['blue', 'red', 'green', 'purple']
sns.set_theme(style="ticks")
fontsize = 30
game = 'Matching Pennies'  # ["Matching Pennies", "IPD", "Ultimatum", "Chicken Game", "Tandem", "Balduzzi", "Hamiltonian"]

hyper_params = get_hparams(game)
interval = hyper_params['interval']
network_type = PGNet if game in ['Tandem', 'Balduzzi', 'Hamiltonian'] else NonPGNet

# ## Training the networks plus running on Game

load_from_file = False
if game == 'Ultimatum':
    dims, Ls = ultimatum()
    Ls_batch = Ls
elif game == 'Tandem':
    dims, Ls = tandem()
    Ls_batch = Ls
elif game == 'Matching Pennies':
    dims, Ls = matching_pennies()
    dims_batch, Ls_batch = matching_pennies_batch(batch_size=hyper_params['batch_size'])
elif game == 'Chicken Game':
    dims, Ls = chicken_game()
    dims_batch, Ls_batch = chicken_game_batch(batch_size=hyper_params['batch_size'])
elif game == 'IPD':
    dims, Ls = ipd(hyper_params['gamma'])
    dims, Ls_batch = ipd_batched(hyper_params['gamma'])
elif game == 'Hamiltonian':
    dims, Ls = hamiltonian_game()
    Ls_batch = Ls
elif game == 'Balduzzi':
    dims, Ls = balduzzi()
    Ls_batch = Ls

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (12, 8),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)

th_list = []
total_losses_out = np.zeros((int(hyper_params['num_innerloop_long']),))
for m, algorithm in enumerate(['COLA_long', 'LOLA', 'HOLA2', 'CGD']):  # 'lola', 'cgd', 'nl' ,'sos', 'HOLA3', 'HOLA4'
    losses_out = np.zeros((hyper_params['num_runs'], hyper_params['num_epochs']))
    losses_out_2 = np.zeros((hyper_params['num_runs'], hyper_params['num_epochs']))
    th = init_th(dims, hyper_params['std'])

    if algorithm == 'COLA_long':
        h_net_long = network_type(hyper_params=hyper_params)
        k_net_long = network_type(hyper_params=hyper_params)
        if load_from_file:
            h_net_long = load_checkpoint('h_net_long.pth', h_net_long)
            k_net_long = load_checkpoint('k_net_long.pth', k_net_long)
        else:
            adam_h_long = optim.Adam(h_net_long.parameters(), lr=hyper_params['lr'])
            adam_k_long = optim.Adam(k_net_long.parameters(), lr=hyper_params['lr'])
            scheduler_h_long = optim.lr_scheduler.StepLR(adam_h_long, 750, gamma=hyper_params['lr_scheduler'])
            scheduler_k_long = optim.lr_scheduler.StepLR(adam_k_long, 750, gamma=hyper_params['lr_scheduler'])
            total_losses_out, total_losses_norm, total_losses_dot = nn_batched_pretrain_colav2(
                Ls_batch, interval=interval, hyper_params=hyper_params,
                net1=h_net_long, net2=k_net_long,
                adam1=adam_h_long, adam2=adam_k_long,
                scheduler1=scheduler_h_long, scheduler2=scheduler_k_long,
                num_innerloop=hyper_params['num_innerloop_long'],
                beta=hyper_params['beta'], total_losses_out=total_losses_out)

    for i in range(hyper_params['num_runs']):
        for k in range(hyper_params['num_epochs']):
            if algorithm == 'COLA_long':
                th, losses, _ = cola_v2(th, Ls, hyper_params['alpha'], beta=hyper_params['beta'], k_net=k_net_long, h_net=h_net_long)
                if game == 'IPD':
                    losses_out[i, k] = (1 - hyper_params['gamma']) * losses[0].data.numpy()
                    losses_out_2[i, k] = (1 - hyper_params['gamma']) * losses[1].data.numpy()
                losses_out[i, k] = losses[0].data.numpy()
                losses_out_2[i, k] = losses[1].data.numpy()
            elif algorithm == 'HOLA1':
                th, losses, _ = update_th(th, Ls, hyper_params['alpha'], algo='higher_order_lola', order=1, beta=hyper_params['beta'])
                losses_out[i, k] = losses[0].data.numpy()
                losses_out_2[i, k] = losses[1].data.numpy()
            elif algorithm == 'HOLA2':
                th, losses, _ = update_th(th, Ls, hyper_params['alpha'], algo='higher_order_lola', order=2, beta=hyper_params['beta'])
                losses_out[i, k] = losses[0].data.numpy()
                losses_out_2[i, k] = losses[1].data.numpy()
            elif algorithm == 'HOLA3':
                th, losses, _ = update_th(th, Ls, hyper_params['alpha'], algo='higher_order_lola', order=3, beta=hyper_params['beta'])
                losses_out[i, k] = losses[0].data.numpy()
                losses_out_2[i, k] = losses[1].data.numpy()
            elif algorithm == 'HOLA4':
                th, losses, _ = update_th(th, Ls, hyper_params['alpha'], algo='higher_order_lola', order=4, beta=hyper_params['beta'])
                losses_out[i, k] = losses[0].data.numpy()
                losses_out_2[i, k] = losses[1].data.numpy()
            elif algorithm == 'HOLA5':
                th, losses, _ = update_th(th, Ls, hyper_params['alpha'], algo='higher_order_lola', order=2, beta=hyper_params['beta'])
                losses_out[i, k] = losses[0].data.numpy()
                losses_out_2[i, k] = losses[1].data.numpy()
            elif algorithm == 'CGD':
                th, losses, _ = update_th(th, Ls, hyper_params['alpha'], algo='cgd', beta=hyper_params['beta'])
                losses_out[i, k] = losses[0].data.numpy()
                losses_out_2[i, k] = losses[1].data.numpy()
            elif algorithm == 'SOS':
                th, losses, _ = update_th(th, Ls, hyper_params['alpha'], algo='sos', beta=hyper_params['beta'])
                losses_out[i, k] = losses[0].data.numpy()
                losses_out_2[i, k] = losses[1].data.numpy()
        if algorithm == 'COLA_long':
            th_list.append(th)
        print(th, 'th ' + algorithm)
        th = init_th(dims, hyper_params['std'])

    mean = np.mean((losses_out), axis=0)
    dev = stats.mstats.sem((losses_out), axis=0)

    plt.plot(np.arange(hyper_params['num_epochs']), mean, markers[m] + '-', markersize=10.0, markevery=20)
    plt.fill_between(np.arange(hyper_params['num_epochs']), mean - dev, mean + dev, alpha=0.25)

plt.title(game, fontsize=fontsize, pad=20)
plt.xlabel('Learning Step', fontsize=fontsize, labelpad=20)
plt.xticks(fontsize=20)
plt.ylabel('Average Loss', fontsize=fontsize, labelpad=10)
plt.yticks(fontsize=20)
plt.legend(['COLA:' + str(hyper_params['beta']),
            'LOLA:' + str(hyper_params['beta']),
            'HOLA2:' + str(hyper_params['beta']),
            'CGD:' + str(hyper_params['beta']),
            # 'SOS:'+str(hyper_params['beta']),
            ], loc='upper left', frameon=True, framealpha=0.75, ncol=3, fontsize=20)
plt.show()
plt.clf()

total_losses = np.log(total_losses_out)

plt.plot(smooth(total_losses, hyper_params['smoothing']), label="LA-Rate:" + str((hyper_params['beta'])))
plt.title(game, fontsize=fontsize, pad=20)
plt.xlabel('Learning Step', fontsize=fontsize, labelpad=20)
plt.ylabel('Log of Consistency Loss', fontsize=fontsize, labelpad=10)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
plt.clf()

hyper_params['state_dict'] = h_net_long.state_dict()
torch.save(hyper_params, 'h_net_long.pth')
hyper_params['state_dict'] = k_net_long.state_dict()
torch.save(hyper_params, 'k_net_long.pth')
np.save('total_losses.npy', total_losses)

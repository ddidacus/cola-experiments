import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns
from torch.nn import CosineSimilarity
import argparse
import random
import json
import os
import yaml

from cola.networks import NonPGNet, PGNet
from cola.utils import get_gradient, init_th, load_checkpoint, update_th
from cola.games import (
    ultimatum, tandem, matching_pennies, matching_pennies_batch,
    chicken_game, chicken_game_batch, ipd, ipd_batched,
    hamiltonian_game, balduzzi,
)

def init_game(game, hyper_params):
    if game == 'Ultimatum':
        dims, Ls = ultimatum()
    elif game == 'Tandem':
        dims, Ls = tandem()
    elif game == 'Matching Pennies':
        dims, Ls = matching_pennies()
    elif game == 'Chicken Game':
        dims, Ls = chicken_game()
    elif game == 'IPD':
        dims, Ls = ipd(hyper_params, gamma=hyper_params['gamma'])
    elif game == 'Hamiltonian':
        dims, Ls = hamiltonian_game()
    elif game == 'Balduzzi':
        dims, Ls = balduzzi()
    return dims, Ls

def run(game, hyper_params, results_dir):
    interval = hyper_params['interval']
    network_type = PGNet if game in ['Tandem', 'Balduzzi', 'Hamiltonian'] else NonPGNet

    dims, Ls = init_game(game, hyper_params)

    # Load trained COLA networks (smallest LA-rate model)
    la_rates = hyper_params.get('la_rates', [hyper_params['beta']])
    best_la_rate = min(la_rates)
    k_net_long = network_type(hyper_params=hyper_params)
    h_net_long = network_type(hyper_params=hyper_params)
    k_net_long = load_checkpoint(f'{results_dir}/k_net_la{best_la_rate}.pth', k_net_long)
    h_net_long = load_checkpoint(f'{results_dir}/h_net_la{best_la_rate}.pth', h_net_long)

    # ## Calculating the Consistency Loss over an area

    cos = CosineSimilarity(dim=0, eps=1e-15)

    alpha = hyper_params['alpha']
    beta = hyper_params['beta']

    r1 = -interval
    r2 = interval

    num_samples = 100
    results = {}
    for beta in [beta]:
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

            results[algo] = {
                "dot": total_loss_dot.item(),
                "norm": total_loss_norm.item(),
                "l2": total_loss.item(),
                "squared_diff": total_loss_squared_diff.item(),
            }

            print(f'Total Loss DOT {algo}: {results[algo]["dot"]}')
            print(f'Total Loss (normalized updates) {algo}: {results[algo]["norm"]}')
            print(f'Total Loss {algo}: {results[algo]["l2"]}')
            print(f'Total Loss Squared Diff {algo}: {results[algo]["squared_diff"]}\n')

    with open(os.path.join(results_dir, "consistency_loss.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate consistency loss for COLA and baselines.')
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--results_dir", type=str, required=True, help="Path to run folder with trained weights")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config, "r") as f:
        hyper_params = yaml.safe_load(f)

    hyper_params["output_dim"] = hyper_params["input_dim"] // 2
    hyper_params.pop("algorithms", None)

    game = hyper_params["game"]

    print("[-] Hyper parameters:")
    for k, v in hyper_params.items():
        print(f"\t{k} : \t{v}")

    run(game, hyper_params, args.results_dir)

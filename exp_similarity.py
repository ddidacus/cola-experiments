import numpy as np
import torch
from torch.nn import CosineSimilarity
import argparse
import random
import json
import os
import yaml

from cola.networks import NonPGNet, PGNet
from cola.utils import load_checkpoint, update_th
from cola.games import (
    ultimatum, tandem, matching_pennies, chicken_game,
    ipd, hamiltonian_game, balduzzi,
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

def run(game, hyper_params, results_dir, output_dir):
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

    # ## Calculating similarity between COLA and HOLA

    cos = CosineSimilarity(dim=0, eps=1e-15)
    alpha = hyper_params['alpha']
    beta = hyper_params['beta']

    r1 = -interval
    r2 = interval

    num_samples = 1000
    results = {}
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
        results[algo] = total_cola_dot.item()
        print(f'Dot Product COLA {algo}: {results[algo]}\n')

    with open(os.path.join(output_dir, "similarity.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Measure similarity between COLA and HOLA/SOS.')
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--results_dir", type=str, required=True, help="Path to run folder with trained weights")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: results_dir)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.results_dir
    os.makedirs(args.output_dir, exist_ok=True)

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

    run(game, hyper_params, args.results_dir, args.output_dir)

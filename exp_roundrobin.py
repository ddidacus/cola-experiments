import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import torch.optim as optim
import seaborn as sns
from scipy import stats
import argparse
import random
import os
import yaml
import shutil
from datetime import datetime

from cola.networks import NonPGNet, PGNet
from cola.utils import init_th, load_checkpoint, smooth, update_th
from cola.cola import nn_batched_pretrain_colav2
from cola.games import (
    ultimatum, tandem, matching_pennies, matching_pennies_batch,
    chicken_game, chicken_game_batch, ipd, ipd_batched,
    hamiltonian_game, balduzzi,
)
from tqdm import tqdm


def init_game(game, hyper_params):
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
        dims, Ls = ipd(hyper_params, gamma=hyper_params['gamma'])
        dims, Ls_batch = ipd_batched(hyper_params, gamma=hyper_params['gamma'])
    elif game == 'Hamiltonian':
        dims, Ls = hamiltonian_game()
        Ls_batch = Ls
    elif game == 'Balduzzi':
        dims, Ls = balduzzi()
        Ls_batch = Ls
    return Ls_batch, dims, Ls


def mixed_update(th, Ls, alpha, hyper_params, k_net, opponent_algo, opponent_beta):
    """Update player 0 with COLA (k_net), player 1 with the opponent algorithm."""
    th_update = [th[0].clone(), th[1].clone()]
    losses = Ls(th_update)

    # COLA direction for player 0
    if hyper_params['input_dim'] == 10:
        inp = torch.clamp(torch.cat(th_update), -hyper_params['interval'], hyper_params['interval'])
    else:
        inp = torch.cat(th_update)
    cola_delta_0 = k_net(inp)

    # Opponent direction for player 1
    if opponent_algo == 'LOLA':
        _, _, opp_grads = update_th(th, Ls, alpha, algo='lola', beta=opponent_beta)
    elif opponent_algo.startswith('HOLA'):
        order = int(opponent_algo[4:])
        _, _, opp_grads = update_th(th, Ls, alpha, algo='higher_order_lola', order=order, beta=opponent_beta)
    elif opponent_algo == 'CGD':
        _, _, opp_grads = update_th(th, Ls, alpha, algo='cgd', beta=opponent_beta)
    elif opponent_algo == 'SOS':
        _, _, opp_grads = update_th(th, Ls, alpha, algo='sos', beta=opponent_beta)
    else:
        raise ValueError(f"Unknown opponent algorithm: {opponent_algo}")

    with torch.no_grad():
        th_update[0] = (th[0].clone() + alpha * cola_delta_0).requires_grad_(True)
        th_update[1] = (th[1].clone() - alpha * opp_grads[1]).requires_grad_(True)

    return th_update, losses


def run(game, hyper_params, results_dir, load_from_file, config_path, seeds, algorithms, run_id=None):
    game_dir = game.lower().replace(" ", "_")
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(results_dir, game_dir, f"roundrobin_{run_id}")
    os.makedirs(results_dir, exist_ok=True)
    shutil.copy2(config_path, os.path.join(results_dir, "config.yaml"))
    with open(os.path.join(results_dir, "seeds.yaml"), "w") as f:
        yaml.dump(seeds, f)

    markers = ['o', 'v', '^', '*', 's', 'D', 'P', 'X']
    fontsize = 30
    sns.set_theme(style="ticks")
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (12, 8),
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    pylab.rcParams.update(params)

    interval = hyper_params['interval']
    network_type = PGNet if game in ['Tandem', 'Balduzzi', 'Hamiltonian'] else NonPGNet
    Ls_batch, dims, Ls = init_game(game, hyper_params)

    algo_names = [name for name, _ in algorithms]
    la_rates = hyper_params.get('la_rates', [hyper_params['beta']])
    cola_algos = [(n, p) for n, p in algorithms if n in ('COLA_long', 'COLA_short')]
    non_cola_algos = [(n, p) for n, p in algorithms if n not in ('COLA_long', 'COLA_short')]

    if not cola_algos:
        print("[!] No COLA algorithms enabled -- nothing to do in round-robin mode.")
        return
    if not non_cola_algos:
        print("[!] No non-COLA algorithms enabled -- nothing to pair against.")
        return

    # Phase 1: Pre-train COLA_long
    h_net_long = None
    k_net_long = None
    if 'COLA_long' in algo_names:
        if load_from_file:
            best_la_rate = min(la_rates)
            h_net_long = network_type(hyper_params=hyper_params)
            k_net_long = network_type(hyper_params=hyper_params)
            h_net_long = load_checkpoint(f"{load_from_file}/h_net_la{best_la_rate}.pth", h_net_long)
            k_net_long = load_checkpoint(f"{load_from_file}/k_net_la{best_la_rate}.pth", k_net_long)
        else:
            best_la_rate = min(la_rates)
            for la_rate in la_rates:
                print(f"[-] COLA_long pre-training with LA-Rate={la_rate}...")
                h_net = network_type(hyper_params=hyper_params)
                k_net = network_type(hyper_params=hyper_params)
                adam_h = optim.Adam(h_net.parameters(), lr=hyper_params['lr'])
                adam_k = optim.Adam(k_net.parameters(), lr=hyper_params['lr'])
                scheduler_h = optim.lr_scheduler.StepLR(adam_h, 750, gamma=hyper_params['lr_scheduler'])
                scheduler_k = optim.lr_scheduler.StepLR(adam_k, 750, gamma=hyper_params['lr_scheduler'])
                total_losses_out = np.zeros((int(hyper_params['num_innerloop_long']),))
                total_losses_out, _, _ = nn_batched_pretrain_colav2(
                    Ls_batch, interval=interval, hyper_params=hyper_params,
                    net1=h_net, net2=k_net,
                    adam1=adam_h, adam2=adam_k,
                    scheduler1=scheduler_h, scheduler2=scheduler_k,
                    num_innerloop=hyper_params['num_innerloop_long'],
                    beta=la_rate, total_losses_out=total_losses_out)

                total_losses = np.log(total_losses_out)
                plt.plot(smooth(total_losses, hyper_params['smoothing']), label=f"LA-Rate: {la_rate}")

                if la_rate == best_la_rate:
                    h_net_long = h_net
                    k_net_long = k_net

                hyper_params['state_dict'] = h_net.state_dict()
                torch.save(hyper_params, f"{results_dir}/h_net_la{la_rate}.pth")
                hyper_params['state_dict'] = k_net.state_dict()
                torch.save(hyper_params, f"{results_dir}/k_net_la{la_rate}.pth")
                del hyper_params['state_dict']

            plt.title(game, fontsize=fontsize, pad=20)
            plt.xlabel('Learning Step', fontsize=fontsize, labelpad=20)
            plt.ylabel('Log of Consistency Loss', fontsize=fontsize, labelpad=10)
            plt.legend(fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.savefig(f"{results_dir}/log_consistency_loss.png", bbox_inches='tight', dpi=300)
            plt.clf()
            print(f"[-] Using COLA_long model pre-trained with LA-Rate={best_la_rate}")

    # Phase 1b: Pre-train COLA_short (hardcoded 800 steps)
    h_net_short = None
    k_net_short = None
    if 'COLA_short' in algo_names:
        if load_from_file:
            h_net_short = network_type(hyper_params=hyper_params)
            k_net_short = network_type(hyper_params=hyper_params)
            h_net_short = load_checkpoint(f"{load_from_file}/h_net_short.pth", h_net_short)
            k_net_short = load_checkpoint(f"{load_from_file}/k_net_short.pth", k_net_short)
        else:
            best_la_rate = min(la_rates)
            print(f"[-] COLA_short pre-training (800 steps) with LA-Rate={best_la_rate}...")
            h_net_short = network_type(hyper_params=hyper_params)
            k_net_short = network_type(hyper_params=hyper_params)
            adam_h = optim.Adam(h_net_short.parameters(), lr=hyper_params['lr'])
            adam_k = optim.Adam(k_net_short.parameters(), lr=hyper_params['lr'])
            scheduler_h = optim.lr_scheduler.StepLR(adam_h, 750, gamma=hyper_params['lr_scheduler'])
            scheduler_k = optim.lr_scheduler.StepLR(adam_k, 750, gamma=hyper_params['lr_scheduler'])
            total_losses_out = np.zeros((800,))
            total_losses_out, _, _ = nn_batched_pretrain_colav2(
                Ls_batch, interval=interval, hyper_params=hyper_params,
                net1=h_net_short, net2=k_net_short,
                adam1=adam_h, adam2=adam_k,
                scheduler1=scheduler_h, scheduler2=scheduler_k,
                num_innerloop=800,
                beta=best_la_rate, total_losses_out=total_losses_out)

            hyper_params['state_dict'] = h_net_short.state_dict()
            torch.save(hyper_params, f"{results_dir}/h_net_short.pth")
            hyper_params['state_dict'] = k_net_short.state_dict()
            torch.save(hyper_params, f"{results_dir}/k_net_short.pth")
            del hyper_params['state_dict']
            print(f"[-] COLA_short pre-training complete")

    # Phase 2: Round-robin training -- COLA vs each opponent
    pair_results = []

    for cola_name, cola_params in cola_algos:
        k_net = k_net_long if cola_name == 'COLA_long' else k_net_short

        for m, (opp_name, opp_params) in enumerate(non_cola_algos):
            pair_name = f"{cola_name}_vs_{opp_name}"
            pair_dir = os.path.join(results_dir, pair_name)
            os.makedirs(pair_dir, exist_ok=True)

            print(f"[-] Training pair: {pair_name}...")
            losses_p0 = np.zeros((hyper_params['num_runs'], hyper_params['num_epochs']))
            losses_p1 = np.zeros((hyper_params['num_runs'], hyper_params['num_epochs']))

            for i in range(hyper_params['num_runs']):
                torch.manual_seed(seeds[i])
                random.seed(seeds[i])
                np.random.seed(seeds[i] % (2**31))
                th = init_th(dims, hyper_params['std'])

                for k in tqdm(range(hyper_params['num_epochs']), desc=f"{pair_name} run {i+1}"):
                    th, losses = mixed_update(
                        th, Ls, hyper_params['alpha'], hyper_params,
                        k_net=k_net,
                        opponent_algo=opp_name, opponent_beta=opp_params['beta'])
                    if game == 'IPD':
                        losses_p0[i, k] = (1 - hyper_params['gamma']) * losses[0].data.numpy()
                        losses_p1[i, k] = (1 - hyper_params['gamma']) * losses[1].data.numpy()
                    else:
                        losses_p0[i, k] = losses[0].data.numpy()
                        losses_p1[i, k] = losses[1].data.numpy()

            mean_p0 = np.mean(losses_p0, axis=0)
            dev_p0 = stats.mstats.sem(losses_p0, axis=0)
            mean_p1 = np.mean(losses_p1, axis=0)
            dev_p1 = stats.mstats.sem(losses_p1, axis=0)
            pair_results.append((pair_name, cola_name, opp_name, opp_params,
                                 mean_p0, dev_p0, mean_p1, dev_p1))

            # Per-pair plot: both players' loss curves
            epochs = np.arange(hyper_params['num_epochs'])
            plt.plot(epochs, mean_p0, marker='o', markersize=10, markevery=20,
                     label=f"{cola_name} (P0)")
            plt.fill_between(epochs, mean_p0 + dev_p0, mean_p0 - dev_p0,
                             alpha=0.25, label='_nolegend_')
            plt.plot(epochs, mean_p1, marker='v', markersize=10, markevery=20,
                     label=f"{opp_name} (P1)")
            plt.fill_between(epochs, mean_p1 + dev_p1, mean_p1 - dev_p1,
                             alpha=0.25, label='_nolegend_')

            plt.title(f"{game}: {cola_name} vs {opp_name}", fontsize=fontsize, pad=20)
            plt.xlabel('Learning Step', fontsize=fontsize, labelpad=20)
            plt.ylabel('Average Loss', fontsize=fontsize, labelpad=10)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(loc='upper left', frameon=True, framealpha=0.75, fontsize=20)
            plt.savefig(f"{pair_dir}/avg_loss.png", bbox_inches='tight', dpi=300)
            plt.clf()

    # Summary plots
    if not pair_results:
        return

    epochs = np.arange(hyper_params['num_epochs'])
    cmap = plt.cm.get_cmap('tab10', max(len(pair_results), 1))

    # Summary: COLA's loss (player 0) across all opponents
    for m, (pair_name, cola_name, opp_name, opp_params,
            mean_p0, dev_p0, mean_p1, dev_p1) in enumerate(pair_results):
        plt.plot(epochs, mean_p0, marker=markers[m % len(markers)], markersize=10, markevery=20,
                 color=cmap(m), label=f"{cola_name} vs {opp_name}: {opp_params['beta']}")
        plt.fill_between(epochs, mean_p0 + dev_p0, mean_p0 - dev_p0,
                         alpha=0.25, color=cmap(m), label='_nolegend_')

    plt.title(f"{game} - COLA Loss (Round Robin)", fontsize=fontsize, pad=20)
    plt.xlabel('Learning Step', fontsize=fontsize, labelpad=20)
    plt.ylabel('Average Loss', fontsize=fontsize, labelpad=10)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='upper left', frameon=True, framealpha=0.75, ncol=2, fontsize=20)
    plt.savefig(f"{results_dir}/summary_cola_loss.png", bbox_inches='tight', dpi=300)
    plt.clf()

    # Summary: Opponent's loss (player 1) across all pairs
    for m, (pair_name, cola_name, opp_name, opp_params,
            mean_p0, dev_p0, mean_p1, dev_p1) in enumerate(pair_results):
        plt.plot(epochs, mean_p1, marker=markers[m % len(markers)], markersize=10, markevery=20,
                 color=cmap(m), label=f"{opp_name} vs {cola_name}: {opp_params['beta']}")
        plt.fill_between(epochs, mean_p1 + dev_p1, mean_p1 - dev_p1,
                         alpha=0.25, color=cmap(m), label='_nolegend_')

    plt.title(f"{game} - Opponent Loss (Round Robin)", fontsize=fontsize, pad=20)
    plt.xlabel('Learning Step', fontsize=fontsize, labelpad=20)
    plt.ylabel('Average Loss', fontsize=fontsize, labelpad=10)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='upper left', frameon=True, framealpha=0.75, ncol=2, fontsize=20)
    plt.savefig(f"{results_dir}/summary_opponent_loss.png", bbox_inches='tight', dpi=300)
    plt.clf()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Round-robin: COLA vs each opponent algorithm.')
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--alpha", type=float, default=None, help="Override alpha from config")
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--run_id", type=str, default=None, help="Run folder name (default: timestamp)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        hyper_params = yaml.safe_load(f)

    hyper_params["output_dim"] = hyper_params["input_dim"] // 2
    hyper_params["num_runs"] = args.num_runs

    if args.alpha is not None:
        hyper_params["alpha"] = args.alpha

    game = hyper_params["game"]

    all_algorithms = hyper_params.pop("algorithms")
    algorithms = [[a["name"], a] for a in all_algorithms if a.get("enabled", False)]

    random.seed(42)
    seeds = [random.randint(0, 2**32 - 1) for _ in range(args.num_runs)]

    print("[-] Hyper parameters:")
    for k, v in hyper_params.items():
        print(f"\t{k} : \t{v}")

    cola_names = [a[0] for a in algorithms if a[0] in ('COLA_long', 'COLA_short')]
    opp_names = [a[0] for a in algorithms if a[0] not in ('COLA_long', 'COLA_short')]
    print(f"[-] COLA algorithms: {cola_names}")
    print(f"[-] Opponents: {opp_names}")
    print(f"[-] Pairs: {[f'{c} vs {o}' for c in cola_names for o in opp_names]}")
    print(f"[-] Seeds: {seeds}")

    run(game, hyper_params, results_dir=args.results_dir, load_from_file=args.load_path,
        config_path=args.config, seeds=seeds, algorithms=algorithms, run_id=args.run_id)

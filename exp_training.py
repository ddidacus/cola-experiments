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
import torch.nn.functional as F
from cola.utils import get_gradient, init_th, load_checkpoint, smooth, update_th
from cola.cola import cola_v2, nn_batched_pretrain_colav2
from cola.games import (
    ultimatum, tandem, matching_pennies, matching_pennies_batch,
    chicken_game, chicken_game_batch, ipd, ipd_batched,
    hamiltonian_game, balduzzi,
)
from tqdm import tqdm

GAMES = [
    "Matching Pennies",
    "IPD",
    "Ultimatum",
    "Chicken Game",
    "Tandem",
    "Balduzzi",
    "Hamiltonian"
]

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

def compute_consistency(algo_name, algo_params, Ls, hyper_params, h_net=None, k_net=None):
    """Compute mean consistency loss for an algorithm over random point samples."""
    interval = hyper_params['interval']
    beta = algo_params['beta']
    alpha = hyper_params['alpha']
    r1 = -interval
    r2 = interval
    num_samples = 100

    total_loss = 0
    for _ in range(num_samples):
        theta_x = (r1 - r2) * torch.rand(hyper_params['output_dim'], requires_grad=True) + r2
        theta_y = (r1 - r2) * torch.rand(hyper_params['output_dim'], requires_grad=True) + r2
        th = [theta_x, theta_y]

        if algo_name in ('COLA_long', 'COLA_short'):
            inp = torch.cat([theta_x.unsqueeze(0), theta_y.unsqueeze(0)], dim=1)
            x_comp = k_net(inp).squeeze(0)
            y_comp = h_net(inp).squeeze(0)
        elif algo_name.startswith('HOLA'):
            order = int(algo_name[4:])
            _, _, grads = update_th(th=th, Ls=Ls, alpha=alpha, algo='higher_order_lola', order=order, beta=beta)
            x_comp = -grads[0]
            y_comp = -grads[1]
        elif algo_name == 'LOLA':
            _, _, grads = update_th(th=th, Ls=Ls, alpha=alpha, algo='lola', beta=beta)
            x_comp = -grads[0]
            y_comp = -grads[1]
        elif algo_name == 'CGD':
            _, _, grads = update_th(th=th, Ls=Ls, alpha=alpha, algo='cgd', beta=beta)
            x_comp = -grads[0]
            y_comp = -grads[1]
        elif algo_name == 'SOS':
            _, _, grads = update_th(th=th, Ls=Ls, alpha=alpha, algo='sos', beta=beta)
            x_comp = -grads[0]
            y_comp = -grads[1]
        else:
            return None

        th_delta_y = [th[0], th[1] + beta * y_comp]
        th_delta_x = [th[0] + beta * x_comp, th[1]]
        loss_delta_y = Ls(th_delta_y)[0]
        loss_delta_x = Ls(th_delta_x)[1]
        incons_gradX = -get_gradient(loss_delta_y, th[0])
        incons_gradY = -get_gradient(loss_delta_x, th[1])

        incons_grad = torch.cat([incons_gradX, incons_gradY])
        cons_grad = torch.cat([x_comp, y_comp])

        incons_grad_norm = F.normalize(incons_grad, dim=0)
        cons_grad_norm = F.normalize(cons_grad, dim=0)
        total_loss += ((cons_grad_norm - incons_grad_norm) ** 2).sum().item()

    return total_loss / num_samples

def run(game, hyper_params, results_dir, load_from_file, config_path, seeds, algorithms, run_id=None):
    game_dir = game.lower().replace(" ", "_")
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(results_dir, game_dir, run_id)
    os.makedirs(results_dir, exist_ok=True)
    shutil.copy2(config_path, os.path.join(results_dir, "config.yaml"))
    with open(os.path.join(results_dir, "seeds.yaml"), "w") as f:
        yaml.dump(seeds, f)

    # markers = ['o', 'v', '^', '*', 's', 'D']
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

    th_list = []
    algo_names = [name for name, _ in algorithms]
    la_rates = hyper_params.get('la_rates', [hyper_params['beta']])
    scatter_data = []

    # Phase 1: Pre-train COLA for each LA-rate
    h_net_long = None
    k_net_long = None
    if 'COLA_long' in algo_names:
        if load_from_file:
            h_net_long = network_type(hyper_params=hyper_params)
            k_net_long = network_type(hyper_params=hyper_params)
            h_net_long = load_checkpoint(f"{load_from_file}/h_net_long.pth", h_net_long)
            k_net_long = load_checkpoint(f"{load_from_file}/k_net_long.pth", k_net_long)
        else:
            best_la_rate = min(la_rates)
            for la_rate in la_rates:
                print(f"[-] COLA pre-training with LA-Rate={la_rate}...")
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

                # Plot consistency loss curve for this LA-rate
                total_losses = np.log(total_losses_out)
                plt.plot(smooth(total_losses, hyper_params['smoothing']), label=f"LA-Rate: {la_rate}")

                # Keep the model with the lowest LA-rate for training
                if la_rate == best_la_rate:
                    h_net_long = h_net
                    k_net_long = k_net
                    total_losses_best = total_losses

                # Save weights for each LA-rate
                hyper_params['state_dict'] = h_net.state_dict()
                torch.save(hyper_params, f"{results_dir}/h_net_la{la_rate}.pth")
                hyper_params['state_dict'] = k_net.state_dict()
                torch.save(hyper_params, f"{results_dir}/k_net_la{la_rate}.pth")
                del hyper_params['state_dict']

            # Finalize consistency loss plot
            plt.title(game, fontsize=fontsize, pad=20)
            plt.xlabel('Learning Step', fontsize=fontsize, labelpad=20)
            plt.ylabel('Log of Consistency Loss', fontsize=fontsize, labelpad=10)
            plt.legend(fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.savefig(f"{results_dir}/log_consistency_loss.png", bbox_inches='tight', dpi=300)
            plt.clf()

            np.save(f"{results_dir}/total_losses.npy", total_losses_best)
            print(f"[-] Using COLA model pre-trained with LA-Rate={best_la_rate} for training")

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

    # Phase 2: Training
    for m, (algorithm, algo_params) in enumerate(algorithms):
        print(f"[-] Training {algorithm}...")
        losses_out = np.zeros((hyper_params['num_runs'], hyper_params['num_epochs']))
        losses_out_2 = np.zeros((hyper_params['num_runs'], hyper_params['num_epochs']))
        th = init_th(dims, hyper_params['std'])

        # Over multiple runs
        for i in range(hyper_params['num_runs']):
            torch.manual_seed(seeds[i])
            random.seed(seeds[i])
            np.random.seed(seeds[i] % (2**31))
            for k in tqdm(range(hyper_params['num_epochs'])):
                if algorithm == 'COLA_long':
                    th, losses, _ = cola_v2(th, Ls, hyper_params['alpha'], hyper_params=hyper_params, beta=algo_params['beta'], k_net=k_net_long, h_net=h_net_long)
                    if game == 'IPD':
                        losses_out[i, k] = (1 - hyper_params['gamma']) * losses[0].data.numpy()
                        losses_out_2[i, k] = (1 - hyper_params['gamma']) * losses[1].data.numpy()
                    else:
                        losses_out[i, k] = losses[0].data.numpy()
                        losses_out_2[i, k] = losses[1].data.numpy()
                elif algorithm == 'COLA_short':
                    th, losses, _ = cola_v2(th, Ls, hyper_params['alpha'], hyper_params=hyper_params, beta=algo_params['beta'], k_net=k_net_short, h_net=h_net_short)
                    if game == 'IPD':
                        losses_out[i, k] = (1 - hyper_params['gamma']) * losses[0].data.numpy()
                        losses_out_2[i, k] = (1 - hyper_params['gamma']) * losses[1].data.numpy()
                    else:
                        losses_out[i, k] = losses[0].data.numpy()
                        losses_out_2[i, k] = losses[1].data.numpy()
                elif algorithm == 'LOLA':
                    th, losses, _ = update_th(th, Ls, hyper_params['alpha'], algo='lola', beta=algo_params['beta'])
                    losses_out[i, k] = losses[0].data.numpy()
                    losses_out_2[i, k] = losses[1].data.numpy()
                elif algorithm.startswith('HOLA'):
                    order = int(algorithm[4:])
                    th, losses, _ = update_th(th, Ls, hyper_params['alpha'], algo='higher_order_lola', order=order, beta=algo_params['beta'])
                    losses_out[i, k] = losses[0].data.numpy()
                    losses_out_2[i, k] = losses[1].data.numpy()
                elif algorithm == 'CGD':
                    th, losses, _ = update_th(th, Ls, hyper_params['alpha'], algo='cgd', beta=algo_params['beta'])
                    losses_out[i, k] = losses[0].data.numpy()
                    losses_out_2[i, k] = losses[1].data.numpy()
                elif algorithm == 'SOS':
                    th, losses, _ = update_th(th, Ls, hyper_params['alpha'], algo='sos', beta=algo_params['beta'])
                    losses_out[i, k] = losses[0].data.numpy()
                    losses_out_2[i, k] = losses[1].data.numpy()
            if algorithm in ('COLA_long', 'COLA_short'):
                th_list.append(th)
            print(th, 'th ' + algorithm)
            th = init_th(dims, hyper_params['std'])

        # Avg. and std. over runs
        mean = np.mean(losses_out, axis=0)
        dev = stats.mstats.sem(losses_out, axis=0)
        scatter_data.append((algorithm, algo_params, np.var(losses_out[:, -1])))

        # Plot mean losses (accumulate on same figure, matching notebook)
        plt.plot(np.arange(hyper_params['num_epochs']), mean, marker=markers[m % len(markers)], markersize=10.0, markevery=20, label=f"{algorithm}: {algo_params['beta']}")
        plt.fill_between(x=np.arange(hyper_params['num_epochs']), y1=mean + dev, y2=mean - dev, alpha=0.25, label='_nolegend_')

    # Finalize avg loss plot
    plt.title(game, fontsize=fontsize, pad=20)
    plt.xlabel('Learning Step', fontsize=fontsize, labelpad=20)
    plt.xticks(fontsize=20)
    plt.ylabel('Average Loss', fontsize=fontsize, labelpad=10)
    plt.yticks(fontsize=20)
    plt.legend(loc='upper left', frameon=True, framealpha=0.75, ncol=3, fontsize=20)
    plt.savefig(f"{results_dir}/avg_loss.png", bbox_inches='tight', dpi=300)
    plt.clf()

    # Phase 3: Variance over Consistency scatter plot
    if len(scatter_data) > 1:
        print("[-] Computing variance over consistency plot...")
        consistencies = []
        variances = []
        labels = []
        for algo_name, algo_params, var in scatter_data:
            if algo_name == 'COLA_long':
                h_net, k_net = h_net_long, k_net_long
            elif algo_name == 'COLA_short':
                h_net, k_net = h_net_short, k_net_short
            else:
                h_net, k_net = None, None
            cons = compute_consistency(algo_name, algo_params, Ls, hyper_params, h_net=h_net, k_net=k_net)
            if cons is not None:
                consistencies.append(cons)
                variances.append(var)
                labels.append(f"{algo_name}: {algo_params['beta']}")

        if consistencies:
            # Assign a distinct color per unique algorithm name
            unique_algos = list(dict.fromkeys(l.split(":")[0] for l in labels))
            cmap = plt.cm.get_cmap('tab10', max(len(unique_algos), 1))
            algo_color = {name: cmap(i) for i, name in enumerate(unique_algos)}
            colors = [algo_color[l.split(":")[0]] for l in labels]

            log_consistencies = np.log(np.array(consistencies))
            log_variances = np.log(np.array(variances))

            fig, ax = plt.subplots(figsize=(12, 8))
            for i, label in enumerate(labels):
                ax.scatter(log_consistencies[i], log_variances[i], s=100, zorder=5,
                           color=colors[i], label=label)
                ax.annotate(label, (log_consistencies[i], log_variances[i]),
                            textcoords="offset points", xytext=(8, 5), fontsize=14)
            ax.set_xlabel('Log Consistency Loss', fontsize=fontsize, labelpad=20)
            ax.set_ylabel('Log Variance', fontsize=fontsize, labelpad=10)
            ax.set_title(f"{game} - Variance over Consistency", fontsize=fontsize, pad=20)
            ax.tick_params(axis='both', labelsize=20)
            ax.legend(fontsize=16, loc='best')
            plt.savefig(f"{results_dir}/variance_over_consistency.png", bbox_inches='tight', dpi=300)
            plt.clf()
            print("[-] Variance over consistency plot saved")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='A simple script using argparse.')
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--alpha", type=float, default=None, help="Override alpha (outer learning rate) from config")
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

    # Build algorithm list from config, filtering by enabled flag
    all_algorithms = hyper_params.pop("algorithms")
    algorithms = [[a["name"], a] for a in all_algorithms if a.get("enabled", False)]

    random.seed(42)
    seeds = [random.randint(0, 2**32 - 1) for _ in range(args.num_runs)]

    print("[-] Hyper parameters:")
    for k, v in hyper_params.items():
        print(f"\t{k} : \t{v}")
    print(f"[-] Algorithms: {[a[0] for a in algorithms]}")
    print(f"[-] Seeds: {seeds}")

    run(game, hyper_params, results_dir=args.results_dir, load_from_file=args.load_path,
        config_path=args.config, seeds=seeds, algorithms=algorithms, run_id=args.run_id)

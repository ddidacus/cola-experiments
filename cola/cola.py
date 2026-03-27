import torch
import numpy as np
from .utils import get_gradient

def cola_v2(th, Ls, alpha, hyper_params, beta=0.1, k_net=None, h_net=None):
    th_update = [th[0].clone(), th[1].clone()]
    n = len(th_update)
    losses = Ls(th_update)
    # There is a one-to-one correspondence between the
    # indeces of the losses and parameters
    # Meaning th[i] is the parameters of the agent which is affected by losses[i]

    if hyper_params['input_dim'] == 10:
      inp = torch.clamp(torch.cat(th_update), -hyper_params['interval'], hyper_params['interval'])
    elif hyper_params['input_dim'] == 2:
      inp = torch.cat(th_update)

    # APPROXIMATE GRADIENT #
    ########################
    delta_x = k_net(inp)
    delta_y = h_net(inp)
    ##########################

    grads = [delta_x, delta_y]
    # Update theta
    with torch.no_grad():
        for i in range(n):
            th_update[i] += alpha* grads[i]
    return th_update, losses, grads

def nn_batched_pretrain_colav2(Ls, interval, hyper_params={}, net1=None, 
                               net2=None, adam1=None, adam2=None, scheduler1=None, scheduler2=None,
                               num_innerloop=100, beta=0.1, total_losses_out=None):
    total_losses = []
    total_losses_norm = []
    total_losses_dot = []
    r1 = -interval
    r2 = interval
    for m in range(num_innerloop):
        # Discover Neighbourhood
        betas = torch.rand(hyper_params['batch_size'], 1, requires_grad=True)
        theta_0 = (r1 - r2) * torch.rand(hyper_params['batch_size'], hyper_params['output_dim'], requires_grad=True) + r2
        theta_1 = (r1 - r2) * torch.rand(hyper_params['batch_size'], hyper_params['output_dim'], requires_grad=True) + r2
        th = [theta_0, theta_1]
 
        if hyper_params['input_dim'] == 10:
          inp = torch.cat([theta_0, theta_1], dim=1)
        else:
          inp = torch.cat([theta_0, theta_1], dim=1)
        
        net2_output =  net2(inp)
        net1_output =  net1(inp)
        delta_x =  net2_output
        delta_y =  net1_output

        th_delta_y_EXP = [th[0], th[1] + (beta*delta_y)]
        th_delta_x_EXP = [th[0] + (beta*delta_x), th[1]]

        loss_delta_y_EXP = Ls(th_delta_y_EXP)[0]
        loss_delta_x_EXP = Ls(th_delta_x_EXP)[1]

        gradX_loss_delta_y_EXP = -get_gradient(loss_delta_y_EXP.sum(), th[0])
        gradY_loss_delta_x_EXP = -get_gradient(loss_delta_x_EXP.sum(), th[1])
 
        loss_k = (delta_x - gradX_loss_delta_y_EXP)**2
        loss_h = (delta_y - gradY_loss_delta_x_EXP)**2

        loss_k_norm = torch.norm(delta_x - gradX_loss_delta_y_EXP, dim=1)
        loss_h_norm = torch.norm(delta_y - gradY_loss_delta_x_EXP, dim=1)

        pred_grad = torch.cat([delta_x, delta_y], dim=1)
        target_grad = torch.cat([gradX_loss_delta_y_EXP, gradY_loss_delta_x_EXP], dim=1)

        total_loss_dot = -cos(pred_grad, target_grad)
        total_loss = loss_h + loss_k 
        total_loss_norm = loss_h_norm + loss_k_norm
        total_loss.sum(1).mean().backward()

        if m % 1 == 0:
            total_losses_out[m] = total_loss.mean().unsqueeze(-1)
 
        adam1.step()
        adam2.step()
        adam1.zero_grad()
        adam2.zero_grad()
        scheduler1.step()
        scheduler2.step()
    return total_losses_out, None, None

def find_local_min_colav2(Ls, grain, interval, hyper_params, k_net_long, h_net_long):
    x_comps = torch.zeros([grain, grain], dtype=torch.float)
    y_comps = torch.zeros([grain, grain], dtype=torch.float)
    errors = torch.zeros([grain, grain], dtype=torch.float)
    lspace = torch.linspace(-interval, interval, grain, requires_grad=True)
    grid_x, grid_y = torch.meshgrid(lspace, lspace)
    orig_size = grid_x.size(0)
    grid_x_flat = grid_x.reshape(-1)
    grid_y_flat = grid_y.reshape(-1)

    theta_0 = grid_x_flat.unsqueeze(-1)
    theta_1 = grid_y_flat.unsqueeze(-1)

    betas = torch.ones(theta_0.size()) * hyper_params['beta']
    th = [theta_0, theta_1]

    inp = torch.cat([theta_0, theta_1], dim=-1)

    delta_x = k_net_long(inp)
    delta_y = h_net_long(inp)

    th_delta_y = [th[0], th[1] + (hyper_params['beta']*delta_y)]
    th_delta_x = [th[0] + (hyper_params['beta']*delta_x), th[1]]

    loss_delta_y = Ls(th_delta_y)[0]
    loss_delta_x = Ls(th_delta_x)[1]

    gradX_loss_delta_y = -get_gradient(loss_delta_y.sum(), th[0])
    gradY_loss_delta_x = -get_gradient(loss_delta_x.sum(), th[1])

    loss_k = (delta_x - gradX_loss_delta_y)**2
    loss_h = (delta_y - gradY_loss_delta_x)**2

    total_loss = loss_h + loss_k
    x_comps = delta_x.reshape(orig_size, orig_size).transpose(0,1)
    y_comps = delta_y.reshape(orig_size, orig_size).transpose(0,1)
    errors = total_loss.reshape(orig_size, orig_size)
    local_min_coords = torch.argmin((torch.abs(delta_x) + torch.abs(delta_y)))

    local_min_theta_0 = grid_x_flat[local_min_coords.item()].unsqueeze(-1)
    local_min_theta_1 = grid_y_flat[local_min_coords.item()].unsqueeze(-1)

    local_min_theta = [local_min_theta_0, local_min_theta_1]
    x_comps_cola = x_comps.detach().numpy()
    y_comps_cola = y_comps.detach().numpy()   
    ind = np.unravel_index(np.argmin(x_comps_cola**2+y_comps_cola**2, axis=None), x_comps_cola.shape)

    return x_comps, y_comps, errors, local_min_theta
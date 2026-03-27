import torch

def get_gradient(function, param):
    grad = torch.autograd.grad(function, param, create_graph=True, allow_unused=True)[0]
    return grad

def init_th(dims, std):
    th = []
    for i in range(len(dims)):
        if std > 0:
            init = torch.nn.init.normal_(torch.empty(
                dims[i], requires_grad=True), std=std) #.to(device)
        else:
            init = torch.zeros(dims[i], requires_grad=True) #.to(device)
        th.append(init)
    return th

def load_checkpoint(filepath, model):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def smooth(scalars, weight): # Weight between 0 and 1
    last = scalars[0]   # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
      smoothed_val = last * weight + (1 - weight) * point # Calculate smoothed value
      smoothed.append(smoothed_val)                       # Save it
      last = smoothed_val                                 # Anchor the last smoothed value
    return smoothed

def get_hessian(th, grad_L, diag=True, off_diag=True):
  n = len(th)
  H = []
  for i in range(n):
    row_block = []
    for j in range(n):
      if (i == j and diag) or (i != j and off_diag):
        block = [torch.unsqueeze(get_gradient(grad_L[i][i][k], th[j]), dim=0) 
                  for k in range(len(th[i]))]
        row_block.append(torch.cat(block, dim=0))
      else:
        row_block.append(torch.zeros(len(th[i]), len(th[j])))
    H.append(torch.cat(row_block, dim=1))
  return torch.cat(H, dim=0)

def update_th(th, Ls, alpha, algo, a=0.5, b=0.1, gam=1, ep=0.1, lss_lam=0.1, order=1, beta=0.1):
  th_update = [th[0].clone(), th[1].clone()]
  n = len(th_update)
  losses = Ls(th_update)
  grad_L = [[get_gradient(losses[j], th_update[i]) for j in range(n)] for i in range(n)]
  
  if algo == 'la':
    terms = [sum([torch.dot(grad_L[j][i], grad_L[j][j].detach())
                for j in range(n) if j != i]) for i in range(n)]
    grads = [grad_L[i][i]-alpha*get_gradient(terms[i], th_update[i]) for i in range(n)]
  
  elif algo == 'lola':

    terms = [sum([torch.dot(grad_L[j][i], grad_L[j][j])
                for j in range(n) if j != i]) for i in range(n)]
    grads = [grad_L[i][i]-beta*get_gradient(terms[i], th_update[i]) for i in range(n)]

  elif algo == 'higher_order_lola':
      innerloop = []
      delta_x_list = []
      delta_y_list = []
      inp = torch.cat(th_update)

      delta_x = -grad_L[0][0]
      delta_y = -grad_L[1][1]
      delta_x_list.append(delta_x.unsqueeze(0))
      delta_y_list.append(delta_y.unsqueeze(0))
      for i in range(1, order+1):
        th_delta_y = [th_update[0], th_update[1] + (beta)*delta_y]
        th_delta_x = [th_update[0] + (beta)*delta_x, th_update[1]]

        loss_delta_y = Ls(th_delta_y)[0]
        loss_delta_x = Ls(th_delta_x)[1]

        delta_x =  -torch.cat(
            [get_gradient(loss_delta_y, th_update[0])])
        delta_y =  -torch.cat(
            [get_gradient(loss_delta_x, th_update[1])])
      grads = [-delta_x, -delta_y]

  elif algo == 'sga':
    xi = torch.cat([grad_L[i][i] for i in range(n)])
    ham = torch.dot(xi, xi.detach())
    H_t_xi = [get_gradient(ham, th[i]) for i in range(n)]
    H_xi = [get_gradient(sum([torch.dot(grad_L[j][i], grad_L[j][j].detach())
              for j in range(n)]), th[i]) for i in range(n)]
    A_t_xi = [H_t_xi[i]/2-H_xi[i]/2 for i in range(n)]
    
    # Compute lambda (sga with alignment)
    dot_xi = torch.dot(xi, torch.cat(H_t_xi))
    dot_A = torch.dot(torch.cat(A_t_xi), torch.cat(H_t_xi))
    d = sum([len(th[i]) for i in range(n)])
    lam = torch.sign(dot_xi*dot_A/d+ep)
    grads = [grad_L[i][i]+lam*A_t_xi[i] for i in range(n)]
  
  elif algo == 'sos':
    terms = [sum([torch.dot(grad_L[j][i], grad_L[j][j].detach())
                for j in range(n) if j != i]) for i in range(n)]
    xi_0 = [grad_L[i][i]-beta*get_gradient(terms[i], th_update[i]) for i in range(n)]
    chi = [get_gradient(sum([torch.dot(grad_L[j][i].detach(), grad_L[j][j])
              for j in range(n) if j != i]), th_update[i]) for i in range(n)]
    dot = torch.dot(-beta*torch.cat(chi), torch.cat(xi_0))

    p1 = 1 if dot >= 0 else min(1, -a*torch.norm(torch.cat(xi_0))**2/dot)
    xi = torch.cat([grad_L[i][i] for i in range(n)])
    xi_norm = torch.norm(xi)

    p2 = xi_norm**2 if xi_norm < b else 1
    p = min(p1, p2)
    grads = [xi_0[i]-p*beta*chi[i] for i in range(n)]
  
  elif algo == 'cgd': # Slow implementation (matrix inversion)
    dims = [len(th_update[i]) for i in range(n)]
    xi = torch.cat([grad_L[i][i] for i in range(n)])
    H_o = get_hessian(th_update, grad_L, diag=False)
    grad = torch.matmul(torch.inverse(torch.eye(sum(dims))+alpha*H_o), xi)
    grads = [grad[sum(dims[:i]):sum(dims[:i+1])] for i in range(n)]

  elif algo == 'nl': # Naive Learning
    grads = [grad_L[i][i] for i in range(n)]

  with torch.no_grad():
      for i in range(n):
          th_update[i] -= alpha * grads[i]

  return th_update, losses, grads
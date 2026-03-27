import torch

def tandem():
    dims = [1, 1]

    def Ls(th):
        x, y = th
        # Tandem loss (quadratic loss for moving forward + linear penalty for pedalling backwards)
        L_1 = (x + y)**2 - 2.0 * x
        L_2 = (x + y)**2 - 2.0 * y
        return [L_1, L_2]
    return dims, Ls

def tandem_cubed():
    dims = [1, 1]

    def Ls(th):
        x, y = th
        # Tandem loss (quadratic loss for moving forward + linear penalty for pedalling backwards)
        L_1 = (x + y)**4 - 2.0 * x
        L_2 = (x + y)**4 - 2.0 * y
        return [L_1, L_2]
    return dims, Ls


def ultimatum():
  dims = [1, 1]
  def Ls(th):
    x, y = th
    p_fair = torch.sigmoid(x)
    p_accept = torch.sigmoid(y)
    L_1 = -(5*p_fair + 8*(1-p_fair)*p_accept)
    L_2 = -(5*p_fair + 2*(1-p_fair)*p_accept)
    return [L_1, L_2]
  return dims, Ls


def balduzzi():
  dims = [1, 1]
  def Ls(th):
    x, y = th
    L_1 = 0.5*(x**2) + 10*x*y
    L_2 = 0.5*(y**2) - 10*x*y
    return [L_1, L_2]
  return dims, Ls


def hamiltonian_game():
  dims=[1, 1]
  def Ls(th):
    x, y = th
    L_1 = x*y
    L_2 = -x*y
    return [L_1, L_2]
  return dims, Ls


def matching_pennies():
  dims = [1, 1]
  payout_mat_1 = torch.Tensor([[1,-1],[-1,1]])
  payout_mat_2 = -payout_mat_1
  def Ls(th):
    p_1, p_2 = torch.sigmoid(th[0]), torch.sigmoid(th[1])
    x, y = torch.cat([p_1, 1-p_1]), torch.cat([p_2, 1-p_2])
    L_1 = torch.matmul(torch.matmul(x, payout_mat_1), y)
    L_2 = torch.matmul(torch.matmul(x, payout_mat_2), y)
    return [L_1, L_2]
  return dims, Ls


def matching_pennies_batch(batch_size=128):
  dims = [1, 1]
  payout_mat_1 = torch.Tensor([[1,-1],[-1,1]])
  payout_mat_2 = -payout_mat_1
  payout_mat_1 = payout_mat_1.reshape((1, 2, 2)).repeat(batch_size, 1, 1)
  payout_mat_2 = payout_mat_2.reshape((1, 2, 2)).repeat(batch_size, 1, 1)
  def Ls(th):
    p_1, p_2 = torch.sigmoid(th[0]), torch.sigmoid(th[1])
    x, y = torch.cat([p_1, 1-p_1], dim=-1), torch.cat([p_2, 1-p_2], dim=-1)
    L_1 = torch.matmul(torch.matmul(x.unsqueeze(1), payout_mat_1), y.unsqueeze(-1))
    L_2 = torch.matmul(torch.matmul(x.unsqueeze(1), payout_mat_2), y.unsqueeze(-1))
    return [L_1.squeeze(-1), L_2.squeeze(-1)]
  return dims, Ls


def chicken_game():
  dims = [1, 1]
  payout_mat_1 = torch.Tensor([[0, -1],[1, -100]])
  payout_mat_2 = torch.Tensor([[0, 1],[-1, -100]])
  def Ls(th):
    p_1, p_2 = torch.sigmoid(th[0]), torch.sigmoid(th[1])
    x, y = torch.cat([p_1, 1-p_1]), torch.cat([p_2, 1-p_2])
    L_1 = -torch.matmul(torch.matmul(x, payout_mat_1), y)
    L_2 = -torch.matmul(torch.matmul(x, payout_mat_2), y)
    return [L_1, L_2]
  return dims, Ls


def chicken_game_batch(batch_size=128):
  dims = [1, 1]
  payout_mat_1 = torch.Tensor([[0, -1],[1, -100]])
  payout_mat_2 = torch.Tensor([[0, 1],[-1, -100]])
  payout_mat_1 = payout_mat_1.reshape((1, 2, 2)).repeat(batch_size, 1, 1)
  payout_mat_2 = payout_mat_2.reshape((1, 2, 2)).repeat(batch_size, 1, 1)
  def Ls(th):
    p_1, p_2 = torch.sigmoid(th[0]), torch.sigmoid(th[1])
    x, y = torch.cat([p_1, 1-p_1], dim=-1), torch.cat([p_2, 1-p_2], dim=-1)
    L_1 = -torch.matmul(torch.matmul(x.unsqueeze(1), payout_mat_1), y.unsqueeze(-1))
    L_2 = -torch.matmul(torch.matmul(x.unsqueeze(1), payout_mat_2), y.unsqueeze(-1))
    return [L_1.squeeze(-1), L_2.squeeze(-1)]
  return dims, Ls
  

def ipd_batched(hyper_params, gamma=0.96):
  dims = [5, 5]
  payout_mat_1 = torch.Tensor([[-1,-3],[0,-2]])
  payout_mat_2 = payout_mat_1.T
  payout_mat_1 = payout_mat_1.reshape((1, 2, 2)).repeat(hyper_params['batch_size'], 1, 1)
  payout_mat_2 = payout_mat_2.reshape((1, 2, 2)).repeat(hyper_params['batch_size'], 1, 1)
  def Ls(th):
    p_1_0 = torch.sigmoid(th[0][:, 0:1])
    p_2_0 = torch.sigmoid(th[1][:, 0:1])
    p = torch.cat([p_1_0*p_2_0, p_1_0*(1-p_2_0), (1-p_1_0)*p_2_0, (1-p_1_0)*(1-p_2_0)], dim=-1)
    p_1 = torch.reshape(torch.sigmoid(th[0][:, 1:5]), (hyper_params['batch_size'], 4, 1))
    p_2 = torch.reshape(torch.sigmoid(th[1][:, 1:5]), (hyper_params['batch_size'], 4, 1))
    P = torch.cat([p_1*p_2, p_1*(1-p_2), (1-p_1)*p_2, (1-p_1)*(1-p_2)], dim=-1)
    x = torch.eye(4).reshape((1, 4, 4))
    eyes = x.repeat(hyper_params['batch_size'], 1, 1)


    M = -torch.matmul(p.unsqueeze(1), torch.inverse(torch.eye(4)-gamma*P))
    L_1 = torch.matmul(M, torch.reshape(payout_mat_1, (hyper_params['batch_size'], 4, 1)))
    L_2 = torch.matmul(M, torch.reshape(payout_mat_2, (hyper_params['batch_size'], 4, 1)))
    return [L_1.squeeze(-1), L_2.squeeze(-1)]
  return dims, Ls


def ipd(hyper_params, gamma=0.96):
  dims = [5, 5]
  payout_mat_1 = torch.Tensor([[-1,-3],[0,-2]])
  payout_mat_2 = payout_mat_1.T
  def Ls(th):
    p_1_0 = torch.sigmoid(th[0][0:1])
    p_2_0 = torch.sigmoid(th[1][0:1])
    p = torch.cat([p_1_0*p_2_0, p_1_0*(1-p_2_0), (1-p_1_0)*p_2_0, (1-p_1_0)*(1-p_2_0)])
    p_1 = torch.reshape(torch.sigmoid(th[0][1:5]), (4, 1))
    p_2 = torch.reshape(torch.sigmoid(th[1][1:5]), (4, 1))
    P = torch.cat([p_1*p_2, p_1*(1-p_2), (1-p_1)*p_2, (1-p_1)*(1-p_2)], dim=1)
    M = -torch.matmul(p, torch.inverse(torch.eye(4)-gamma*P))
    L_1 = torch.matmul(M, torch.reshape(payout_mat_1, (4, 1)))
    L_2 = torch.matmul(M, torch.reshape(payout_mat_2, (4, 1)))
    return [L_1, L_2]
  return dims, Ls
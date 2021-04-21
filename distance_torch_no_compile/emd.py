import torch

def expanded_pairwise_distances(x, y):
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences * differences, -1)
    return distances

def approximate_earth_mover_distance(x, y, it=100):
    '''
    starting form tf implementation https://github.com/MilesGrey/emd/blob/master/loss_functions/emd.py
    '''
    distance_matrix = expanded_pairwise_distances(x, y)
    k = torch.exp(distance_matrix)
    km = k * distance_matrix
    u = torch.ones(y.shape) / y.shape[1]
    u = u.to('cuda' if x.is_cuda else 'cpu')
    for _ in range(it):
        u = x / ((y / (u @ k)) @ k)
        v = y / (u @ k)
    return torch.sum(u * (v @ km)) / y.shape[0]
    

if __name__ == "__main__":
    x = torch.tensor([
            [2., 2., 1.],
            [2., 1., 2.],
            [2., 2., 2.],
    ])
    y = torch.tensor([
            [2., 2., 1.],
            [2., 1., 2.],
            [2., 2., 2.],
    ])

    print('chamfer loss torch (cpu):', approximate_earth_mover_distance(x, y))
    print('chamfer loss torch (cuda):', approximate_earth_mover_distance(x.cuda(), y.cuda()))
    

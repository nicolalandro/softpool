import torch

def expanded_pairwise_distances(x, y):
    '''
    Input: x is a bxNxd matrix
           y is an optional bxMxd matirx
    Output: dist is a bxNxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    differences = x.unsqueeze(2) - y.unsqueeze(1)
    distances = torch.sum(differences * differences, -1)
    return distances

def chamfer_distance(x, y):
    '''
    input x and y are bxNxM matrix, b: batch, N:number of point, M: point dim (ex. 2 for 2D or 3 for 3D)
    output is a bx1 Matrix with the value of the chamfer distance for each sample of the batch
    '''
    dist_vec = expanded_pairwise_distances(x, y)
    min_distances = torch.topk(dist_vec, k=1, dim=2, largest=False).values
    chamfer = torch.sum(min_distances, dim=1) / torch.tensor(x.shape[1])
    
    return chamfer

class ChamferLoss(torch.nn.Module):
   def forward(self, x, y):
        chamfer = chamfer_distance(x, y)

        return torch.sum(chamfer)

if __name__ == "__main__":
    x = torch.tensor([
        [
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 1., 0.],
        ],
        [
            [1., 1., 0.],
            [1., 2., 0.],
            [0., 1., 0.],
        ]
    ])
    y = torch.tensor([
        [
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 1., 0.],
        ],
        [
            [1., 1., 0.],
            [1., 2., 0.],
            [0., 1., 0.],
        ]
    ])
    chamfer = ChamferLoss()

    print('chamfer loss torch:', chamfer(x, y))
    
    # import sys   
    # sys.path.append("../distance/chamfer/")
    # import dist_chamfer as cd
    # CD = cd.chamferDist()
    # dist1, dist2, _, _= CD(x, y)
    # print('orig', dist1)

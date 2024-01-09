import torch

a = torch.tensor(
    [
        [1., 2., 7., 8., 10.],
        [3., 4., 1., 4., 9.]
    ]
)
b = torch.tensor(
    [3., 4., 1., 4., 9.]
)
#
# print(a.tolist())

print(torch.mean(a, dim=0))

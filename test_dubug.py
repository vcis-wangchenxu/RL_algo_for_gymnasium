import torch

states = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
actions = torch.tensor([[0], [1], [2]])

print(states.max(1)[1])
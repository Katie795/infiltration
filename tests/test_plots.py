from AI4Flood.misc_functions import *
from AI4Flood.generate_csv_files import *

nx = 10
ny = 10
length = 1
depth = 1
dx = length / nx
dy = depth / ny

x, y, xx, yy = generate_csv_files(nx, nx, length=length, depth=depth, name="test")
ones = torch.ones((nx, ny), dtype=torch.float64)
zeros = torch.zeros((nx, ny), dtype=torch.float64)
grad_x = torch.stack((ones, zeros), dim=0)
grad_y = torch.stack((zeros, ones), dim=0)

def test_grads():
    torch.testing.assert_close(calculate_dx(x, dx), ones)
    torch.testing.assert_close(calculate_dx(y, dx), zeros)
    torch.testing.assert_close(calculate_dy(x, dy), zeros)
    torch.testing.assert_close(calculate_dy(y, dy), ones)


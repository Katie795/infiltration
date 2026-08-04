# Tests for Richard's solver.
# 1: Tests whether padded coordinates line up with original coordinates

import torch
from AI4Flood.richards_solver import *
from AI4Flood.generate_csv_files import *

nx = 10
length = 10
t = 0

### generate the coordinates ###

x, y, xx, yy = generate_csv_files(nx, nx, length=10, depth=10, name="horung_")

### tests ###
def test_coordinates():
    """Test whether x_padded and y_padded align with x and y"""
    assert torch.equal(x, xx[1:-1, 1:-1])
    assert torch.equal(y, yy[1:-1, 1:-1])
    #assert torch.equal(x[-1, -1], torch.tensor(length)) # test coords correspond to length


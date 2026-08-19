from AI4Flood.misc_functions import jacobi, dd
import torch

#TODO: does this work for tensors of shape (1, 1, n, m)

def test_jacobi():
    A = torch.tensor([[2.,1.],[5.,7.]])
    b = torch.tensor([11., 13.])
    x0 = torch.tensor([1.,1.])
    #assert jacobi(A, b, x0, maxiterations=1) == torch.tensor([5.0, 1.1429])
    return jacobi(A, b, x0, maxiterations=1)

def test_jacobi():
    A = torch.tensor([[2.,1.],[5.,7.]])
    b = torch.tensor([11., 13.])
    x0 = torch.tensor([1.,1.])
    #assert jacobi(A, b, x0, maxiterations=1) == torch.tensor([5.0, 1.1429])
    return jacobi(A, b, x0, maxiterations=1)


def test_jacobi_iterations():
    A = torch.tensor([[10.,-1., 2., 0. ],
                      [-1.,11., -1., 3.,],
                      [2., -1., 10., -1., ],
                      [0., 3., -1., 8.]])
    b = torch.tensor([6., 25., -11., 15.])
    x0 = torch.tensor([0.,0.,0.,0.])

    return jacobi(A, b, x0, maxiterations=100) # should be [1, 2, -1, 1]

def test_dd():
    A1 = torch.tensor([[2.,1.],[5.,7.]])
    A2 = torch.tensor([[10.,-1., 2., 0. ],
                      [-1.,11., -1., 3.,],
                      [2., -1., 10., -1., ],
                      [0., 3., -1., 8.]])
    A3 = torch.tensor([[10.,-1., 2., 0. ],
                      [-1.,11., -1., 3.,],
                      [2., -1., 10., -1., ],
                      [-5., 3., -1., 8.]])
    print(dd(A1))
    print(dd(A2))
    print(dd(A3))

def test_jacobi_error():
    A = torch.tensor([[10.,-1., 2., 0. ],
                          [-1.,11., -1., 3.,],
                          [2., -1., 10., -1., ],
                          [-5., 3., -1., 8.]])
    b = torch.tensor([6., 25., -11., 15.])
    x0 = torch.tensor([0.,0.,0.,0.])

    return jacobi(A, b, x0, maxiterations=100) # should be [1, 2, -1, 1]

def test_jacobi_unsqueezed():
    A = torch.tensor([[[[2.,1.],[5.,7.]]]])

    b = torch.tensor([[[11., 13.]]])
    x0 = torch.tensor([[[1.,1.]]])
    #assert jacobi(A, b, x0, maxiterations=1) == torch.tensor([5.0, 1.1429])
    return jacobi(A, b, x0, maxiterations=1)

print(test_jacobi_unsqueezed())
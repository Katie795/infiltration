import torch.nn as nn
import torch
import torch.nn.functional as F

class AI4SWE(nn.Module):
    """docstring for two_step"""
    def __init__(self, w1, w2, w3, wm, bias_initializer, dx, beta, diag):
        super(AI4SWE, self).__init__()
        self.xadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.yadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.cmm  = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)

        self.diff.weight.data = w1
        self.xadv.weight.data = w2
        self.yadv.weight.data = w3
        self.cmm.weight.data = wm

        self.diff.bias.data = bias_initializer
        self.xadv.bias.data = bias_initializer
        self.yadv.bias.data = bias_initializer
        self.cmm.bias.data = bias_initializer
        
        self.dx = dx
        self.beta = beta
        self.diag = diag

    def boundary_condition_u(self, values_u, values_uu):
        # boundary conditions are applied using padding.
        # with linear elements, the padding is one element thick
        ny = values_u.shape[2] 
        nx = values_u.shape[3]
        nny = values_uu.shape[2] # this is just ny + 2 (ny with padding)
        nnx = values_uu.shape[3] # this is just nx + 2 (with padding)

        values_uu[0,0,1:nny-1,1:nnx-1] = values_u[0,0,:,:] # insert u values
        values_uu[0,0,:,0] =  values_uu[0,0,:,1]*0 # set zero boundary conditions
        values_uu[0,0,:,nx+1] = values_uu[0,0,:,nx]*0
        values_uu[0,0,0,:] = values_uu[0,0,1,:] # transfer dirichlet values of v to padding
        values_uu[0,0,ny+1,:] = values_uu[0,0,ny,:]
        # zero on left and right boundaries, fixed on upper and lower boundaries 
        return values_uu   

    def boundary_condition_v(self, values_v, values_vv):
        ny = values_v.shape[2]
        nx = values_v.shape[3]
        nny = values_vv.shape[2]
        nnx = values_vv.shape[3]

        values_vv[0,0,1:nny-1,1:nnx-1] = values_v[0,0,:,:]
        values_vv[0,0,:,0] =  values_vv[0,0,:,1] # transfer dirichlet values of v to padding
        values_vv[0,0,:,nx+1] = values_vv[0,0,:,nx]
        values_vv[0,0,0,:] = values_vv[0,0,1,:]*0
        values_vv[0,0,ny+1,:] = values_vv[0,0,ny,:]*0
        return values_vv

    def boundary_condition_eta(self, values_h, values_hp):

        ny = values_h.shape[2]
        nx = values_h.shape[3]
        nny = values_hp.shape[2]
        nnx = values_hp.shape[3]

        # homogeneous neumann boundary conditions
        values_hp[0,0,1:nny-1,1:nnx-1] = values_h[0,0,:,:]
        values_hp[0,0,:,nx+1] = values_hp[0,0,:,nx]#0 # zero bcs on all sides for   
        values_hp[0,0,:,0] = values_hp[0,0,:,1]#0 
        values_hp[0,0,ny+1,:] = values_hp[0,0,ny,:]#0  
        values_hp[0,0,0,:] = values_hp[0,0,1,:]#0 
        return values_hp   

    def PG_vector(self, values_uu, values_vv, values_u, values_v, k3):
        # PG vector is the petrov galerkin stabilisation term
        # This is very important for stabilising the solution!!
        k_u = 0.25 * self.dx * torch.abs(1/2 * (self.dx**-2) * (torch.abs(values_u) * self.dx + torch.abs(values_v) * self.dx) * self.diff(values_uu)) / \
            (1e-03  + (torch.abs(self.xadv(values_uu)) * (self.dx**-2) + torch.abs(self.yadv(values_uu)) * (self.dx**-2)) / 2)

        k_v = 0.25 * self.dx * torch.abs(1/2 * (self.dx**-2) * (torch.abs(values_u) * self.dx + torch.abs(values_v) * self.dx) * self.diff(values_vv)) / \
            (1e-03  + (torch.abs(self.xadv(values_vv)) * (self.dx**-2) + torch.abs(self.yadv(values_vv)) * (self.dx**-2)) / 2)

        k_uu = F.pad(torch.minimum(k_u, k3) , (1, 1, 1, 1), mode='constant', value=0)
        k_vv = F.pad(torch.minimum(k_v, k3) , (1, 1, 1, 1), mode='constant', value=0)

        k_x = 0.5 * (k_u * self.diff(values_uu) + self.diff(values_uu * k_uu) - values_u * self.diff(k_uu))
        k_y = 0.5 * (k_v * self.diff(values_vv) + self.diff(values_vv * k_vv) - values_v * self.diff(k_vv))
        return k_x, k_y

    def PG_scalar(self, values_hh, values_h, values_u, values_v, k3):
        k_u = 0.25 * self.dx * torch.abs(1/2 * (self.dx**-2) * (torch.abs(values_u) * self.dx + torch.abs(values_v) * self.dx) * self.diff(values_hh)) / \
            (1e-03 + (torch.abs(self.xadv(values_hh)) * (self.dx**-2) + torch.abs(self.yadv(values_hh)) * (self.dx**-2)) / 2)  
      
        k_uu = F.pad(torch.minimum(k_u, k3) , (1, 1, 1, 1), mode='constant', value=0)
        return 0.5 * (k_u * self.diff(values_hh) + self.diff(values_hh * k_uu) - values_h * self.diff(k_uu))        

    def forward(self, values_u, values_uu, values_v, values_vv, values_H, values_h, values_hp, b_u, b_uu, b_v, b_vv, dt, rho, k1, k2, k3, eta1_p, source_h, dif_values_h, dif_values_hh, values_hh, values_hhp):
        
        ### STEP 1: SOLVE FOR VELOCITY ###
        values_uu = self.boundary_condition_u(values_u, values_uu) # update the boundary conditions with dirichlet values
        values_vv = self.boundary_condition_v(values_v, values_vv)

        # k_x and k_y are the diffusion coefficients in the x and y directions, respectively
        [k_x,k_y] = self.PG_vector(values_uu, values_vv, values_u, values_v, k3) # update PG stabilisation coefficients

        # b is used for the correction step?
        b_u = (k_x * dt - values_u * self.xadv(values_uu) * dt - values_v * self.yadv(values_uu) * dt) * 0.5 + values_u
        b_v = (k_y * dt - values_u * self.xadv(values_vv) * dt - values_v * self.yadv(values_vv) * dt) * 0.5 + values_v
        b_u = b_u - self.xadv(self.boundary_condition_eta(values_h,values_hp)) * dt
        b_v = b_v - self.yadv(self.boundary_condition_eta(values_h,values_hp)) * dt

        b_uu = self.boundary_condition_u(b_u, b_uu)       
        b_vv = self.boundary_condition_v(b_v, b_vv)

        # sigma is the roughness coefficient (equation 16)
        sigma_q = (b_u**2 + b_v**2)**0.5 * 0.055**2 / (torch.maximum( k1,
            self.dx*self.cmm(self.boundary_condition_eta(values_H+values_h,values_hp))*0.01+(values_H+values_h)*0.99 )**(4/3)) 

        # Equation 22 and 23
        b_u = b_u / (1 + sigma_q * dt / rho)
        b_v = b_v / (1 + sigma_q * dt / rho) # equation 22 and 23?

        [k_x, k_y] = self.PG_vector(b_uu, b_vv, b_u, b_v, k3)
        
        # forward euler time step (equation 27?)
        values_u = values_u + k_x * dt - b_u * self.xadv(b_uu) * dt - b_v * self.yadv(b_uu) * dt   
        values_v = values_v + k_y * dt - b_u * self.xadv(b_vv) * dt - b_v * self.yadv(b_vv) * dt 
        values_u = values_u - self.xadv(self.boundary_condition_eta(values_h,values_hp)) * dt
        values_v = values_v - self.yadv(self.boundary_condition_eta(values_h,values_hp)) * dt     

        # bed roughness coefficient - equation 16
        sigma_q = (values_u**2 + values_v**2)**0.5 * 0.055**2 / (torch.maximum( k1,
            self.dx*self.cmm(self.boundary_condition_eta(values_H+values_h,values_hp))*0.01+(values_H+values_h)*0.99 )**(4/3))

        # equation 22?
        values_u = values_u / (1 + sigma_q * dt / rho)
        values_v = values_v / (1 + sigma_q * dt / rho)

        values_uu = self.boundary_condition_u(values_u,values_uu)
        values_vv = self.boundary_condition_v(values_v,values_vv)

        ### STEP 2: SOLVE FOR HEIGHT ###

        eta1 = torch.maximum(k2,(values_H + values_h)) # why is values_H used here?
        eta2 = torch.maximum(k1,(values_H + values_h))

        # Right hand side of equation 24, called b because it is Ax = b in Jacobi iteration.
        b = self.beta * rho * (-self.xadv(self.boundary_condition_eta(eta1,eta1_p)) * values_u - \
                           self.yadv(self.boundary_condition_eta(eta1,eta1_p)) * values_v - \
                           eta1 * self.xadv(values_uu) - eta1 * self.yadv(values_vv) + \
                           self.PG_scalar(self.boundary_condition_eta(eta1,eta1_p), eta1, values_u, values_v, k3) - \
                           self.cmm(self.boundary_condition_eta(dif_values_h,dif_values_hh)) / dt + source_h) / (dt * eta2)   
        values_h_old = values_h.clone()

        # Jacobi interation for equation 25
        for i in range(2):
            values_hh = values_hh - (-self.diff(self.boundary_condition_eta(values_hh,values_hhp))  + self.beta * rho / (dt**2 * eta2) * values_hh) / \
                    (self.diag + self.beta * rho / (dt**2 * eta2)) + b / (self.diag + self.beta * rho / (dt**2 * eta2))
        
        values_h = values_h + values_hh # update values_h
        
        dif_values_h = values_h - values_h_old 
        
        ### STEP 3: CORRECT VELOCITY ### - equation 27
        values_u = values_u - self.xadv(self.boundary_condition_eta(values_hh,values_hhp)) * dt / rho # equation 28
        values_v = values_v - self.yadv(self.boundary_condition_eta(values_hh,values_hhp)) * dt / rho 

        return values_u, values_v, values_h, values_hh, dif_values_h, sigma_q
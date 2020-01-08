# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 09:51:24 2018

@author: sfmt4368
"""

import torch
from torch.autograd import grad
import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D 
#matplotlib inline

# fix random seeds
npr.seed(2019)
torch.manual_seed(2019)
# define material parameters
E = 30000
nu = 0.3

# define domain and collocation points
x_dom = 0., 1., 21
y_dom = 0., 1., 21
# create points
lin_x = np.linspace(x_dom[0], x_dom[1], x_dom[2])
lin_y = np.linspace(y_dom[0], y_dom[1], y_dom[2])
dom = np.zeros((x_dom[2]*y_dom[2],2))
c = 0
for x in np.nditer(lin_x):
    tb = y_dom[2]*c
    te = tb + y_dom[2]
    c += 1
    dom[tb:te,0] = x
    dom[tb:te,1] = lin_y

# define constitutive matrix
C = torch.empty((3,3))
C = torch.tensor([[E/(1-nu**2), E*nu/(1-nu**2), 0],[E*nu/(1-nu**2), E/(1-nu**2), 0],[0, 0, E/(2*(1+nu))]])

# define boundary conditions
# penalty parameteris for BCs
bc_d_tol = 10e7
# Dirichlet x, y, dir, val
bc_d1 = []    
bc_d2 = []
bc_d2_pts_idx = np.where(dom[:,1] == 0)
bc_d2_pts = dom[bc_d2_pts_idx,:][0]
for (x, y) in bc_d2_pts:
    bc_d2.append((np.array([x, y]), 1, 0.))
    bc_d1.append((np.array([x, y]), 1, 0.))
# von Neumann boundary conditions
# applied force
bc_n1 = []
bc_n1_pts_idx = np.where(dom[:,1] == 1.0)
bc_n1_pts = dom[bc_n1_pts_idx,:][0]
for (x, y) in bc_n1_pts:
    bc_n1.append((np.array([x, y]), 1, 1000.))
# free boundaryies  
bc_n2 = []
bc_n3 = []
for x in np.nditer(lin_x):
    bc_n2.append((np.array([0., x]), 1, 0.))
    bc_n3.append((np.array([0., x]), 1, 0.))
    
# convert numpy BCs to torch
def ConvBCsToTensors(bc_d):
    size_in_1 = len(bc_d)
    size_in_2 = len(bc_d[0][0])
    bc_in = torch.empty(size_in_1, size_in_2, dtype=torch.float)
    c = 0
    for bc in bc_d:
        bc_in[c,:] = torch.from_numpy(bc[0])
        c += 1
    return bc_in

# mean squared loss function
def mse_loss(tinput, target):
    return torch.sum((tinput - target) ** 2) / tinput.data.nelement()

# Pytorch neural network
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = torch.tanh(self.linear1(x))
        y = self.linear2(h_relu)
        return y

# D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in, H, D_out = 2, 101, 2

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)


# prepare inputs and outputs for training the model
dtype = torch.float
# inputs
x = torch.from_numpy(dom).float()
x.requires_grad_(True)
N = x.size()[0]
# get tensor inputs and outputs for boundary conditions
# Dirichlet
# boundary 1
bc_d1_x = ConvBCsToTensors(bc_d1)
bc_d1_y = torch.from_numpy(np.array([bc_d1[0][2]])).float()
# boundary 2
bc_d2_x = ConvBCsToTensors(bc_d2)
bc_d2_y = torch.from_numpy(np.asarray([i[2] for i in bc_d2])).float()
# von Neumann
# applied forces
bc_n1_x = ConvBCsToTensors(bc_n1).float()
bc_n1_x.requires_grad_(True)
bc_n1_y = torch.from_numpy(np.asarray([i[2] for i in bc_n1])).float()
# free boundary
bc_n2_x = ConvBCsToTensors(bc_n2).float()
bc_n2_x.requires_grad_(True)
bc_n2_y = torch.from_numpy(np.asarray([i[2] for i in bc_n2])).float()
bc_n3_x = ConvBCsToTensors(bc_n3).float()
bc_n3_x.requires_grad_(True)
bc_n3_y = torch.from_numpy(np.asarray([i[2] for i in bc_n3])).float()

# prepare inputs for testing the model
# define domain
x_dom_test = 0., 1., 21
y_dom_test = 0., 1., 21
# create points
x_space = np.linspace(x_dom_test[0], x_dom_test[1], x_dom_test[2])
y_space = np.linspace(y_dom_test[0], y_dom_test[1], y_dom_test[2])

# Plate in plane stress
def KinematicEquation(primal_pred, x):
    d_primal_x_pred = grad(primal_pred[:,0].unsqueeze(1),x,torch.ones(x.size()[0], 1, dtype=torch.float),create_graph=True,retain_graph=True)[0]
    d_primal_y_pred = grad(primal_pred[:,1].unsqueeze(1),x,torch.ones(x.size()[0], 1, dtype=torch.float),create_graph=True,retain_graph=True)[0]
    eps_x = d_primal_x_pred[:, 0].unsqueeze(1)
    eps_y = d_primal_y_pred[:, 1].unsqueeze(1)
    eps_xy = d_primal_x_pred[:, 1].unsqueeze(1) + d_primal_y_pred[:, 0].unsqueeze(1)
    eps = torch.cat((eps_x, eps_y, eps_xy), 1)
    return eps


def ConstitutiveEquation(eps_pred, C):
    sig = torch.mm(eps_pred, C)
    return sig


def BalanceEquation(sig_pred, x):
    d_sig_x_pred = grad(sig_pred[:,0].unsqueeze(1),x,torch.ones(x.size()[0], 1, dtype=torch.float),create_graph=True,retain_graph=True)[0]
    d_sig_y_pred = grad(sig_pred[:,1].unsqueeze(1),x,torch.ones(x.size()[0], 1, dtype=torch.float),create_graph=True,retain_graph=True)[0]
    d_sig_xy_pred = grad(sig_pred[:,2].unsqueeze(1),x,torch.ones(x.size()[0], 1, dtype=torch.float),create_graph=True,retain_graph=True)[0]
    dsig_dx = d_sig_x_pred[:, 0].unsqueeze(1) + d_sig_xy_pred[:, 1].unsqueeze(1)
    dsig_dy = d_sig_y_pred[:, 1].unsqueeze(1) + d_sig_xy_pred[:, 0].unsqueeze(1)
    return torch.cat((dsig_dx, dsig_dy), 1)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = mse_loss
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-8)
optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0) 
    
for t in range(100):
    # Zero gradients, perform a backward pass, and update the weights.  
    
    def closure():
        # predicion of primary variables
        primal_pred = model(x)
        # evaluate kinematic equations
        eps_pred = KinematicEquation(primal_pred, x)
        # evalute consitutive equations
        sig_pred = ConstitutiveEquation(eps_pred, C)
        # evalute balance equations - y_pred for training the model in the domain
        y_pred = BalanceEquation(sig_pred, x)
        y = torch.zeros_like(y_pred)
        dom_crit = criterion(y_pred,y)
        # treat boundary conditions
        # Dirichlet boundary conditions
        # boundary 1 x - direction
        bc_d1_pred = model(bc_d1_x)
        bc_d1_crit = criterion(bc_d1_pred[:,bc_d1[0][1]],bc_d1_y)
        # boundary 2 y - direction
        bc_d2_pred = model(bc_d2_x)
        bc_d2_crit = criterion(bc_d2_pred[:,bc_d2[0][1]],bc_d2_y)        
        # von Neumann boundary conditions
        # boundary 1
        bc_n1_primal = model(bc_n1_x)
        bc_n1_eps = KinematicEquation(bc_n1_primal, bc_n1_x)
        bc_n1_sig = ConstitutiveEquation(bc_n1_eps, C)
        bc_n1_pred = torch.cat((bc_n1_sig[:, 1], bc_n1_sig[:, 2]))
        bc_n1_crit = criterion(bc_n1_pred, torch.cat((bc_n1_y, torch.zeros_like(bc_n1_y))))
        # boundary 2
        bc_n2_primal = model(bc_n2_x)
        bc_n2_eps = KinematicEquation(bc_n2_primal, bc_n2_x)
        bc_n2_sig = ConstitutiveEquation(bc_n2_eps, C)
        bc_n2_pred = torch.cat((bc_n2_sig[:, 0], bc_n2_sig[:, 2]))
        bc_n2_crit = criterion(bc_n2_pred, torch.cat((bc_n2_y, torch.zeros_like(bc_n2_y))))
        # boundary 3
        bc_n3_primal = model(bc_n3_x)
        bc_n3_eps = KinematicEquation(bc_n3_primal, bc_n3_x)
        bc_n3_sig = ConstitutiveEquation(bc_n3_eps, C)
        bc_n3_pred = torch.cat((bc_n3_sig[:, 0], bc_n3_sig[:, 2]))
        bc_n3_crit = criterion(bc_n3_pred, torch.cat((bc_n3_y, torch.zeros_like(bc_n3_y))))
        # Compute and print loss
#        loss = dom_crit + bc_n1_crit + bc_n2_crit + bc_n3_crit + ( bc_d1_crit + bc_d2_crit )  * bc_d_tol
        loss = dom_crit + bc_n1_crit + ( bc_d1_crit + bc_d2_crit )  * bc_d_tol
        print(t, loss.item())
        optimizer.zero_grad()
        loss.backward()
        return loss
    optimizer.step(closure)

# plot results
oShapeX = np.zeros((x_dom_test[2], y_dom_test[2]))
oShapeY = np.zeros((x_dom_test[2], y_dom_test[2]))
surfaceUx = np.zeros((x_dom_test[2], y_dom_test[2]))
surfaceUy = np.zeros((x_dom_test[2], y_dom_test[2]))
defShapeX = np.zeros((x_dom_test[2], y_dom_test[2]))
defShapeY = np.zeros((x_dom_test[2], y_dom_test[2]))
for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        t_tensor = torch.tensor([x,y]).unsqueeze(0)
        tRes = model(t_tensor).detach().numpy()[0]
        oShapeX[i][j] = x
        oShapeY[i][j] = y
        surfaceUx[i][j] = tRes[0]
        surfaceUy[i][j] = tRes[1]
        defShapeX[i][j] = x +  tRes[0] * 2.1
        defShapeY[i][j] = y + tRes[1] * 2.1

fig, axes = plt.subplots(ncols=2)
#X, Y = np.meshgrid(x_space, y_space)
cs2 = axes[0].contourf(defShapeX, defShapeY, surfaceUx, cmap=cm.PuBu_r)
cs1 = axes[1].contourf(defShapeX, defShapeY, surfaceUy, cmap=cm.PuBu_r)
cbar1 = fig.colorbar(cs1, ax=axes[0])
cbar2 = fig.colorbar(cs2, ax=axes[1])
axes[0].set_title("Displacement in x")
axes[1].set_title("Displacement in y")
fig.tight_layout()
for tax in axes:
    tax.set_xlabel('$x$')
    tax.set_ylabel('$y$')
plt.show()

cornerPts = torch.tensor([[0.0,0.0],[1.0,0.],[1.0,1.0],[0.0,1.0]])
cornerPts.requires_grad_(True)
print(model(cornerPts))
print(ConstitutiveEquation(KinematicEquation(model(cornerPts), cornerPts), C))
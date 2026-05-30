import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.path import Path
import scienceplots
from Geometry import Geometry
from Source_Panel_solver import Normal_influence, Velocity_field, pressure, rgeometry, Tangential_influence

plt.style.use(['science', 'grid', 'dark_background'])
plt.rcParams['text.usetex'] = False  # critical override

U = 1
AoA = 0

_x = np.linspace(-0.5, 1.5, 500)
_y = np.linspace(-0.5, 0.5, 500)

xs,ys = rgeometry(AoA/180 * np.pi)

N = len(xs)-1
L = int((N)/2)
geom = Geometry(xc=xs, yc=ys)

xlower = np.zeros((L))
xupper = np.zeros((L))

for i in range(L):
    xlower[i] = geom.x_ctr[i]
    xupper[i] = geom.x_ctr[i+L]


print(xlower)

print(xupper)

A = np.zeros((N, N))

Vt = np.zeros((N))

for j in range(N):
    xj, yj = geom.x_start[j], geom.y_start[j]
    phij = geom.theta[j]
    Sj = geom.length[j]
    for i in range(N):
        if i != j:
            xi, yi = geom.x_ctr[i], geom.y_ctr[i]
            phii = geom.theta[i]

            A[i, j] = Normal_influence(Sj, xi, yi, phii, phij, xj, yj)

np.fill_diagonal(A, np.pi)

b = np.zeros((N))
for k in range(N):
    b[k] = - U * np.pi * 2 * np.cos(geom.theta[k] + np.pi/2)

lam = np.linalg.solve(A, b)

print(lam)

x,y = np.meshgrid(_x, _y)

u,v = Velocity_field(U, lam, x, y, geom.x_start[:,None,None], geom.y_start[:,None,None], geom.theta[:,None,None], geom.length[:,None,None])

Tinf = 0

for i in range(N):
    xi, yi = geom.x_ctr[i], geom.y_ctr[i]
    phii = geom.theta[i]
    for j in range(N):
        if j != i:
            xj, yj = geom.x_start[j], geom.y_start[j]
            phij = geom.theta[j]
            Sj = geom.length[j]

            Tinf += lam[j] * Tangential_influence(Sj, xi, yi, phii, phij, xj, yj) / (2*np.pi)

    Vt[i] = Tinf + U * np.sin(geom.theta[i] + np.pi/2)
    Tinf = 0

SCp = 1 - Vt**2 / U**2

print("Vt min/max:", Vt.min(), Vt.max())
print("Cp min/max:", SCp.min(), SCp.max())


print(u.shape)
print(v.shape)

xc= np.append(geom.x_end, geom.x_start[0])
yc= np.append(geom.y_end, geom.y_start[0])

speed = np.sqrt(u**2 + v**2)

Cp = pressure(U, speed)

polygon = Path(np.column_stack((geom.x_start, geom.y_start)))
points = np.column_stack((x.ravel(), y.ravel()))
inside = polygon.contains_points(points)

inside = inside.reshape(x.shape)

u = np.ma.array(u, mask=inside)
v = np.ma.array(v, mask=inside)
Cp = np.ma.array(Cp, mask=inside)

fig, axe = plt.subplots(2, 2, figsize=(10,8))
ax = axe[0, 0]
ax.plot(xc, yc)
ax.set_aspect('equal')
ax.set_title('Flow visualization')
ax.streamplot(x, y, u, v, color='white', density=1.2, broken_streamlines=False, arrowstyle='-')

ax = axe[0, 1]
ax.plot(xc, yc)
ax.set_aspect('equal')
levels = np.linspace(-3, 1, 80)
norm = colors.TwoSlopeNorm(vmin=Cp.min(),vcenter=0,  vmax=Cp.max())
c = ax.contourf(x, y, Cp, levels=100, norm=norm, cmap='plasma')
ax.set_title('Pressure Coefficient')
fig.colorbar(c, ax=ax, label=r'$C_p$')

ax = axe[1, 0]
ax.plot(xlower, SCp[:L:], color='red', linestyle='dashdot', linewidth=2, label='lower surface')
ax.plot(xupper, SCp[L:], color='purple', linestyle='dashdot', linewidth=2, label='upper surface')
ax.axhline(0, color='white', linewidth=1.2, alpha=0.9, zorder=10)
ax.axvline(0, color='white', linewidth=1.2, alpha=0.9, zorder=10)
ax.legend(loc='upper right')
ax.set_xlabel('x')
ax.set_ylabel(r'$C_p$')
plt.show()
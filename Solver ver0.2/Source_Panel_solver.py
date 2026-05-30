import numpy as np

def Normal_influence(Sj, xi, yi, phii, phij, xj, yj):

    A = -(xi - xj)*np.cos(phij) - (yi - yj)*np.sin(phij)
    B = (xi - xj)**2 + (yi - yj)**2
    C = np.sin(phii - phij)
    D = -(xi - xj)*np.sin(phii) + (yi - yj)*np.cos(phii)
    E = np.sqrt(B - A**2)

    Iij = (C/2) * np.log((Sj**2 + 2*A*Sj + B)/B) + ((D-A*C)/E) * (np.atan2(Sj+A, E) - np.atan2(A,E))

    return Iij


def Tangential_influence(Sj, xi, yi, phii, phij, xj, yj):

    A = -(xi - xj) * np.cos(phij) - (yi - yj) * np.sin(phij)
    B = (xi - xj) ** 2 + (yi - yj) ** 2
    C = -np.cos(phii - phij)
    D = (xi - xj) * np.cos(phii) + (yi - yj) * np.sin(phii)
    E = np.sqrt(B - A ** 2)

    Jij = (C / 2) * np.log((Sj ** 2 + 2 * A * Sj + B) / B) + ((D - A * C) / E) * (np.atan2(Sj + A, E) - np.atan2(A, E))

    return Jij

def X_influence(x, y, xj, yj, phij, Sj):

    A = -(x - xj)*np.cos(phij) - (y - yj)*np.sin(phij)
    B = (x - xj)**2 + (y - yj)**2
    Cx = -np.cos(phij)
    Dx = (x -xj)
    E = np.sqrt(B - A**2)

    Mxp = (Cx/2) * np.log((Sj**2 + 2*A*Sj + B)/B) + ((Dx-A*Cx)/E) * (np.atan2(Sj+A, E) - np.atan2(A,E))

    return Mxp

def Y_influence(x, y, xj, yj, phij, Sj):
    A = -(x - xj) * np.cos(phij) - (y - yj) * np.sin(phij)
    B = (x - xj) ** 2 + (y - yj) ** 2
    Cy = -np.sin(phij)
    Dy = (y - yj)
    E = np.sqrt(B - A ** 2)

    Myp = (Cy / 2) * np.log((Sj ** 2 + 2 * A * Sj + B) / B) + ((Dy - A * Cy) / E) * (np.atan2(Sj + A, E) - np.atan2(A, E))

    return Myp

def Velocity_field(U, lam, x, y, xj, yj, phij,Sj):

    N = len(lam)

    u = np.zeros_like(x)
    v = np.zeros_like(y)

    u = U + np.sum((lam[:, None, None]/(2*np.pi)) * X_influence(x, y, xj, yj, phij, Sj), axis=0)
    v = np.sum((lam[:, None, None]/(2*np.pi)) * Y_influence(x, y, xj, yj, phij, Sj), axis=0)

    return u,v

def pressure(U, Speed):

    Cp = 1 - Speed**2/U**2

    return Cp

def rgeometry(AoA):
    xc, yc = np.loadtxt("points.txt", unpack=True)

    xs = xc * np.cos(AoA) + yc * np.sin(AoA)
    ys = -xc * np.sin(AoA) + yc * np.cos(AoA)

    return xs, ys





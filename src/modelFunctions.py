import numpy as np


def gauss4term(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4):
    return Gauss(x, a1, b1, c1) + Gauss(x, a2, b2, c2) + Gauss(x, a3, b3, c3) + Gauss(x, a4, b4, c4)


def gauss3term(x, a1, b1, c1, a2, b2, c2, a3, b3, c3):
    return Gauss(x, a1, b1, c1) + Gauss(x, a2, b2, c2) + Gauss(x, a3, b3, c3)


def gauss2term(x, a1, b1, c1, a2, b2, c2):
    return Gauss(x, a1, b1, c1) + Gauss(x, a2, b2, c2)


def Gauss(x, a, b, c):
    return a * np.exp( - (x - b)**2 / 2 / c**2)


def lorentz4term(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4):
    return Lorentz(x, a1, b1, c1) + Lorentz(x, a2, b2, c2) + Lorentz(x, a3, b3, c3) + Lorentz(x, a4, b4, c4)


def lorentz3term(x, a1, b1, c1, a2, b2, c2, a3, b3, c3):
    return Lorentz(x, a1, b1, c1) + Lorentz(x, a2, b2, c2) + Lorentz(x, a3, b3, c3)


def lorentz2term(x, a1, b1, c1, a2, b2, c2):
    return Lorentz(x, a1, b1, c1) + Lorentz(x, a2, b2, c2)


def Lorentz(x, a, b, c):
    return a / ((x - b)**2 + c**2)

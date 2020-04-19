# Michael Stien

import numpy as np

class waveFunction:
    def __init__(self,Lx,Ly,Nx,Ny):
        self.Lx=float(Lx)
        self.Ly=float(Ly)
        self.Nx=float(Nx)
        self.Ny=float(Ny)

    def calc(self,x,y):
        temp = np.sqrt(self.Lx*self.Ly)
        temp = (2.0/temp)
        inSin1 = np.pi*self.Nx*x
        inSin1 = inSin1/self.Lx
        inSin2 = np.pi*self.Ny*y
        inSin2 = inSin2/self.Ly
        temp = temp*np.sin(inSin1)*np.sin(inSin2)
        return temp
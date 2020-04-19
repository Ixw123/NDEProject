"""The following program determines the probability distribution for a wave function using 2d scatter points"""
import numpy
import matplotlib.pyplot as plt
import random

def prob(Lx,Ly,Nx,Ny,size):
    data = []
    Psi = waveFunction(Lx,Ly,Nx,Ny)
    for x in range(size):
        for y in range(size):
            dx=Lx*float(x)/size
            dy=Ly*float(y)/size
            if(random.random()<abs(Psi.calc(dx,dy))):
               point = (dx,dy)
               data.append(point)
    return data
class waveFunction:
    def __init__(self,Lx,Ly,Nx,Ny):
        self.Lx=float(Lx)
        self.Ly=float(Ly)
        self.Nx=float(Nx)
        self.Ny=float(Ny)

    def calc(self,x,y):
        temp = numpy.sqrt(self.Lx*self.Ly)
        temp = (2.0/temp)
        inSin1 = numpy.pi*self.Nx*x
        inSin1 = inSin1/self.Lx
        inSin2 = numpy.pi*self.Ny*y
        inSin2 = inSin2/self.Ly
        temp = temp*numpy.sin(inSin1)*numpy.sin(inSin2)
        return temp

"""Btw Micah the link you provided is for spin our system does not incorporate spin in any way
    The Pauli Equation would however.
    I believe that the way to determine what quantum state it is in is to measure the energy in the area and its movement
    and compare it with known values for the atom you are studying.
    Using the how to page this is all the function would be.
"""
def upSpin(spinState):
    mag=spinState[0]^2 + spinState[1]^2
    return spinState[0]/mag
data = prob(4.0,3.0,2.0,4.0,250)
data = zip(*data)
plt.scatter(*data,s=.1)
plt.show()

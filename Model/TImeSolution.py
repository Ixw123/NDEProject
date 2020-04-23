import numpy as np
import Model.CommonFunctions as MFunc
import Model.WaveFunction as mwf
"""Written by Michael Stein runs the RungeKuttaStep
step size= h
A = Spacial Discretization matrix
b = Eigenvector solution
Todo = Animate
"""
def rungeKutta(A,b,h,Iterations):
    c = np.zeros(len(b))
    for x in range(Iterations):
        (b,c)=rungaKuttaStep(A,b,c,h)
"""
Written by Michael Stein
Solves the RungaKuttaStep step size h for the equation u* = ipAu
Where p is the constant planck/(2pi*m)
c contains the complex part of the solution
while b contains the real part.
kc contains the complex portion
kr contains the real portion
"""

def rungeKuttaStep(A,b,c,h):
    p=1
    kc1= h*(np.matmul(A,p*b))
    kr1= -h*(np.matmul(A,p*c))
    kc2= h*(np.matmul(A,p*(b+(.5*kr1))))
    kr2= -h*(np.matmul(A,p*(c+(.5*kc1))))
    kc3= h*(np.matmul(A,p*(b+(.5*kr2))))
    kr3= -h*(np.matmul(A,p*(b+(.5*kc2))))
    kc4= h*(np.matmul(A,p*(b+kr3)))
    kr4= -h*(np.matmul(A,p*(b+kc3)))
    real = b + (1/6)*(kr1+2rk2+2kr3+kr4)
    complex = b + (1/6)*(kc1+2kc2+2kc3+kc4)
    return (real,complex)

"""
Written by Michael Stein
Gives the Probability Density Function as it changes over time
dataR contains the real portion
dataC contains the complex portion
E is an eigenvalue
h is typically the planck constant
I'm not sure how the distribution will look without the correct constants and units
But I figured I'd make it anyways.
Returns the still frames for each t
"""
def prob(Lx,Ly,Nx,Ny,E,size,Times):
    dataTR = []
    dataTC = []
    h=1
    Psi = mwf.waveFunction(Lx,Ly,Nx,Ny)
    for t in range(Times):
        realE= -cos(t*(E/h))
        complexE = -sin(t*(E/h))
        dataR = []
        dataC = []
        for x in range(size):
            for y in range(size):
                dx=Lx*float(x)/size
                dy=Ly*float(y)/size
                if(random.random()<abs(Psi.calc(dx,dy))*realE):
                    point = (dx,dy)
                    dataR.append(point)
                if(random.random()<abs(Psi.calc(dx,dy))*complexE):
                    point = (dx,dy)
                    dataC.append(point)
        dataTR.append(dataR)
        dataTC.append(dataC)
    return (dataTR,dataTC)

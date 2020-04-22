import numpy as np
import random
import matplotlib.pyplot as plt

import Model.WaveFunction as mwf
# Micah Church
# Implemented from https://math.la.asu.edu/~gardner/QR.pdf

# H = I - (2uu^T)/||u||_2^2
def getHouseHolderMat(u):
    I = np.identity(u.shape[0])
    u2norm = np.linalg.norm(u)
    
    return I - (2*(np.outer(u, u.T)))/u2norm**2

def getHessenBergForm(A):
    # print(A)
    # Loop through each column
    for i in range(A.shape[0] - 2):
        a = A[i, :]
        # print(a)
        # get the magnitude of the a to make the u
        r = np.zeros(a.shape)
        r[0] = A[0,0]
        # print(A[i+1][i+1:])
        # print(A[i+1:, 0])
        r[1] = np.linalg.norm(A[i+1:, 0])
        # print("r", r)
        u = a - r
        H = getHouseHolderMat(u)
        # print(H)
        A = np.dot(np.dot(H, A), H)
        # print(A)
        # R = np.dot(H, A)
        # print(np.dot(H, A))

    return A
# Implementation from 
# https://rpubs.com/aaronsc32/qr-decomposition-householder
# https://math.la.asu.edu/~gardner/QR.pdf
def getQR(A):
    # loop through the columns
    alpha = 1
    sign = 1 if A[0,0] > 0 else -1 
    Q = np.identity(A.shape[0])
    for j in range(A.shape[1] - 1):
        a = A[j:, j]
        e = np.zeros(a.shape[0])
        e[0] = 1
        
        v = a + sign*np.linalg.norm(a)*e
        H = np.identity(A.shape[0])
        H[j:, j:] = getHouseHolderMat(v)

        Q = np.dot(Q, H)
        R = np.dot(H, A)
        A = R

    return Q, R

# https://math.la.asu.edu/~gardner/QR.pdf
def getQREigens(A, tol=1e-15, cntMax=1e4):
    if A.shape[0] == 1:
        raise ValueError("Error the matrix", A, "is a singleton and thus no QR factorization or QR algorithm can be computed!!!!")
    # print(A.shape)
    cnt = 0
    eigenVecs = np.identity(A.shape[0])
    # offDiagInds = [[i, j] for i in range(A.shape[0], A.shape[1] if i != j)]
    offDiagInds = np.array(~np.eye(A.shape[0], dtype=bool))
    # print(offDiagInds)
    while abs(sum(A[offDiagInds[:, 0], offDiagInds[:, 1]])) > tol and cnt < cntMax:
        Q, R = getQR(A)
        # print(Q)
        eigenVecs = np.dot(eigenVecs, Q)
        A = np.dot(R, Q)
        cnt += 1
    # print("Q:", Q, "R:", R, "A:", A, "EigenVecs:", eigenVecs)

    # print(list(set(np.diag(np.round(A, 8)))))

    if len(list(set(np.diag(np.round(A, 8))))) < A.shape[0]:
        print("A", A)
        print(Q, Q)
        print("offDiag sum", abs(sum((A[offDiagInds[:, 0], offDiagInds[:, 1]]))), "cnt", cnt)
        print("WARNING: Mulitplicity of eigenvalue/s is greater than 1, eigenvectors for that value may be off!!!!")

    print("Off diagonal abs sum is", abs(sum(A[offDiagInds[:, 0], offDiagInds[:, 1]])), "tol is", tol)

    return np.diag(A), eigenVecs

# Micah Church: Implemented from 
# https://stackoverflow.com/questions/47463827/solving-1d-schr%C3%B6dinger-equation-with-numerov-method-python
# and https://en.wikipedia.org/wiki/Numerov%27s_method
# TODO:
# Think of a better way to determine E and V
def numerov(start, end, y0, inputs, a=3, m=1, hBar=1, E=.5):
    outputs = [*y0]
    # dT <= dX. dY
    h = inputs[1] - inputs[0]
    # define potentials
    # a = 3
    # m = 1
    # hBar = 1
    k = np.zeros(inputs.shape[0])
    # Make sure this makes sense
    def V(x, a):
        return 1 if np.abs(x) < a else 0

    for i in range(k.shape[0]):
        k[i] = 2*m*(E-V(inputs[i], a))/hBar**2

    for i in range(2, inputs.shape[0]):
        # g_n = g(x_n) = v[x_n]
        # y_n = psi[n]
        # h = dT
        # y_n+1(1 + h**2/12*g_n+1) = 2*y_n(1 - 5*h**2/12*g_n) - y_n-1(1 + 12*g_n-1)
        # y_n term, ie the leading term
        # 2*y_n(1 - 5*h**2/12*g_n)
        e = 2*outputs[i-1]*(1 - 5*h**2/12*k[i - 1])
        # y_n-1 term, ie the tail of the implicit method
        # adding the h_n+1 term
        s = 2*outputs[i-2]*(1 + 5*h**2/12*k[i - 2])
        # divisor term to isolate and add the y_n+1 term
        d = 1 + h**2/12*k[i]
        outputs.append((e - s)/d)

    return outputs

# Micah Church
# Setting up Apsi = -E2m/hbar * psi
# Mesh: the 2d dimensional coordinate system
# BC is a dictionary with the keys being the coordinates as a tuple and the value being the initial condition
# Spatial descritization based on central differences method
def getSpatialDescritization(mesh, BC, DEBUG_PRINT=False):
    boundryCoords = [list(k) for k in BC.keys()]
    unknowns = mesh.shape[1] * mesh.shape[2] - len(list(BC.keys()))
    a = np.zeros((unknowns, unknowns))
    b = np.zeros((unknowns, 1))
    # -1* a = a
    # u = [u11, u12 ..., u21, u22, ... un-1, 1, un-1, 2, ... un, 1, un, 2, ...]
    psiCnt = 0

    points = {}
    for i in range(1, mesh.shape[1] - 1):
        for j in range(1, mesh.shape[2] - 1):
            if tuple([i, j]) not in list(points.keys()):
                points[tuple([i, j])] = psiCnt
            psiCnt += 1

    psiCnt = 0
    for i in range(1, mesh.shape[1] - 1):
        for j in range(1, mesh.shape[2] - 1):
            a[psiCnt, psiCnt] = 4
            neighbors = np.array([[i - 1, j], [i + 1, j], [i, j + 1], [i, j - 1]])
            for n in neighbors:
                if DEBUG_PRINT: print(n, n[0] in boundryCoords[:][0] and n[1] in boundryCoords[:][1], np.where(boundryCoords == n)[0])
                # Check if the point is on the boundry
                if np.where(boundryCoords == n, True, False).all(axis=1).any():
                    b[psiCnt] += BC[tuple(n)]
                else:
                    a[psiCnt, points[tuple(n)]] = -1
            psiCnt += 1

    psi = np.linalg.solve(a, b)

    return a, psi

# Michael Stien
def prob(Lx,Ly,Nx,Ny,size):
    data = []
    Psi = mwf.waveFunction(Lx,Ly,Nx,Ny)
    for x in range(size):
        for y in range(size):
            dx=Lx*float(x)/size
            dy=Ly*float(y)/size
            if(random.random()<abs(Psi.calc(dx,dy))):
               point = (dx,dy)
               data.append(point)
    return data

"""Btw Micah the link you provided is for spin our system does not incorporate spin in any way
    The Pauli Equation would however.
    I believe that the way to determine what quantum state it is in is to measure the energy in the area and its movement
    and compare it with known values for the atom you are studying.
    Using the how to page this is all the function would be.
"""
def upSpin(spinState):
    mag=spinState[0]^2 + spinState[1]^2
    return spinState[0]/mag

# Implemented from https://en.wikipedia.org/wiki/Power_iteration
def powerItteration(A, x0=None, tol=1e-5, DEBUG_PRINT=False):
    # Intial itteration vector
    bK = np.zeros((A.shape[1], 2))
    if x0 is None:
        bK[:, 0] = np.random.rand(A.shape[1])
    else:
        bK[:, 0] = x0
    # print("Initial", bK[:, 0])
    bK[:, 1] = np.dot(A, bK[:, 0])
    eigenVals = [np.dot(bK[:, 1], bK[:, 0])/np.dot(bK[:, 0], bK[:, 0])]
    # print(np.dot(A, bK[:, 0]), np.linalg.norm(np.dot(A, bK[:, 0])))
    # print("first guess", bK[:, 1])
    # print("first eigenvalue", bK[0, 1], "eigenVector", bK[:, 1]/ bK[0, 1])
    bK[:, 0] = bK[:, 1]
    bK[:, 1] = np.dot(A, bK[:, 0])
    eigenVals.append(np.dot(bK[:, 1], bK[:, 0])/np.dot(bK[:, 0], bK[:, 0]))
    # print("eigenVals", eigenVals)
    cnt = 0
    # print(abs(sum((bK[:, 1] - bK[:, 0])/bK[:, 1])))
    # if DEBUG_PRINT: print(bK[:, 1])
    while abs(np.array(eigenVals[-1]) - np.array(eigenVals[-2])) > tol or cnt > 10000:
        bK[:, 0] = bK[:, 1]
        bK[:, 1] = np.dot(A, bK[:, 0])
        eigenVals.append([np.dot(bK[:, 1], bK[:, 0])/np.dot(bK[:, 0], bK[:, 0])])
        cnt += 1

    # if DEBUG_PRINT: print("final error", abs(sum((bK[:, 1] - bK[:, 0])/bK[:, 1])), "cnt", cnt)

    eigenVal = eigenVals[-1]
    eigenVec = bK[:, 1]/np.linalg.norm(bK[:, 1])

    return eigenVal, eigenVec

# Not really working because it just gets the same values and vectors everytime
# https://math.stackexchange.com/questions/768882/power-method-for-finding-all-eigenvectors
# Implements deflation algortihm for finding eigen values need to check error counts
# Think of a different method that is more stable
def getEigenVectors(A, tol=1e-5, DEBUG_PRINT=False):

    basis = A
    initial = np.identity(A.shape[0])
    x0 = initial[:, 0]
    firstEigenVal, firstEigenVec =  powerItteration(basis, x0=x0, tol=tol, DEBUG_PRINT=DEBUG_PRINT)

    if DEBUG_PRINT: print("First Eigen vectors and value", firstEigenVal, firstEigenVec)

    eigenVals = [firstEigenVal]
    eigenVecs = [firstEigenVec]

    for i in range(1, A.shape[0]):
        # print("old basis", basis)
        # print(eigenVecs[-1], np.outer(eigenVecs[-1], eigenVecs[-1].T))
        basis -= eigenVals[-1]*np.outer(eigenVecs[-1], eigenVecs[-1].T)
        # print("new basis", basis)
        x0 = initial[:, 1]
        eVal, eVec =  powerItteration(basis, x0=x0, tol=tol, DEBUG_PRINT=DEBUG_PRINT)
        # Check to see if the eigenValues are repeated with a different eigenVector
        cnt = 0
        while np.where(eVec == eigenVecs, True, False).all(axis=1).any() or eVal in eigenVals and cnt >= 10000:
            eVal, eVec =  powerItteration(basis, x0=x0*random(), tol=tol, DEBUG_PRINT=DEBUG_PRINT)
            cnt += 1
        eigenVals.append(eVal)
        eigenVecs.append(eVec)

    return eigenVals, eigenVecs
        


# Implement this http://www.robots.ox.ac.uk/~sjrob/Teaching/EngComp/ecl4.pdf

# https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
def gramSchmidt(A, eigenVec):
    # A -= eigenVec[0] / np.linalg.norm(eigenVec)**2 * np.outer(eigenVec, eigenVec.T)
    eigenVecs = [eigenVec]
    for i in range(1, A.shape[0]):
        # print("i is", i)
        vec = A[:, i]
        sum = 0
        for j in range(i):
            # print("j is", j)
            proj = (np.dot(vec, eigenVecs[j])/np.dot(eigenVecs[j], eigenVecs[j])) * eigenVecs[j]
            sum += proj


        # A -= eigenVec[0] / np.linalg.norm(eigenVec)**2 * np.outer(eigenVec, eigenVec.T)
        eigenVecs.append(vec - sum)

    return eigenVecs


def plotA(A):
    x = np.arange(0, A.shape[0])
    # yInds = np.arange(0, A.shape[1])
    # y = np.zeros((A.shape[0], A.shape[1]))
    # print(y.shape)
    y = []
    for rInd in range(A.shape[0]):      
        # values = [i for i in range(A.shape[1]) if A[rInd, i] != 0]
        outputs = np.where(A[rInd] != 0)[0]
        # print(values)  
        y.append(outputs)
        # print(y[rInd])
    # print(x.shape, y.shape)
    # print(y)
    for xe, ye in zip(x, y):
        plt.scatter([xe] * len(ye), ye)
    plt.show()

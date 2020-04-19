import numpy as np
import random

import Model.WaveFunction as mwf

# Micah Church
# Setting up Aphi = -E2m/hbar * phi
# Mesh: the 2d dimensional coordinate system
# BC is a dictionary with the keys being the coordinates as a tuple and the value being the initial condition
def getA(mesh, BC, DEBUG_PRINT=False):
    boundryCoords = [list(k) for k in BC.keys()]
    # if DEBUG_PRINT: print("boundryCoords", boundryCoords)
    # if DEBUG_PRINT: print(mesh.shape[1], mesh.shape[2], len(list(BC.keys())))
    unknowns = mesh.shape[1] * mesh.shape[2] - len(list(BC.keys()))
    # if DEBUG_PRINT: print(unknowns)
    a = np.zeros((unknowns, unknowns))
    b = np.zeros((unknowns, 1))
    # if DEBUG_PRINT: print("a shape", a.shape)
    # for k,v in BC.items():
    #     if DEBUG_PRINT: print(k)
    #     a[k[0], k[1]] = v
    # -1* a = a
    # u = [u11, u12 ..., u21, u22, ... un-1, 1, un-1, 2, ... un, 1, un, 2, ...]
    # inds = np.arange(0, unknowns)
    phiCnt = 0
    # if DEBUG_PRINT: print("Number of variables",mesh.shape[2] - 2)

    points = {}
    for i in range(1, mesh.shape[1] - 1):
        # if DEBUG_PRINT: print("i is", i)
        for j in range(1, mesh.shape[2] - 1):
            if tuple([i, j]) not in list(points.keys()):
                points[tuple([i, j])] = phiCnt
            phiCnt += 1

    phiCnt = 0
    # if DEBUG_PRINT: print(a)
    for i in range(1, mesh.shape[1] - 1):
        # if DEBUG_PRINT: print("i is", i)
        for j in range(1, mesh.shape[2] - 1):
            a[phiCnt, phiCnt] = 4
            neighbors = np.array([[i - 1, j], [i + 1, j], [i, j + 1], [i, j - 1]])
            # if DEBUG_PRINT: print(neighbors)
            for n in neighbors:
                if DEBUG_PRINT: print(n, n[0] in boundryCoords[:][0] and n[1] in boundryCoords[:][1], np.where(boundryCoords == n)[0])
                # Check if the point is on the boundry
                if np.where(boundryCoords == n, True, False).all(axis=1).any():
                    # if DEBUG_PRINT: print("adding", BC[tuple(n)], "to", b[phiCnt])
                    b[phiCnt] += BC[tuple(n)]
                else:
                    # if DEBUG_PRINT: print(points[tuple(n)])
                    a[phiCnt, points[tuple(n)]] = -1
            phiCnt += 1

    if DEBUG_PRINT: print("a is", a)

    if DEBUG_PRINT: print((a == a.T).all())
    phi = np.linalg.solve(a, b)
    print("Solution is", phi)
    plotA(a)

    if DEBUG_PRINT: print("A is", a, "b is", b)

    return a

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
def powerItteration(A, tol=1e-5, DEBUG_PRINT=False):
    # Intial itteration vector
    bK = np.zeros((A.shape[1], 2))
    bK[:, 0] = np.ones(A.shape[1])
    # print("Initial", bK[:, 0])
    bK[:, 1] = np.dot(A, bK[:, 0])/np.linalg.norm(np.dot(A, bK[:, 0]))
    # print(np.dot(A, bK[:, 0]), np.linalg.norm(np.dot(A, bK[:, 0])))
    # print("first guess", bK[:, 1])
    # print("first eigenvalue", bK[0, 1], "eigenVector", bK[:, 1]/ bK[0, 1])
    bK[:, 0] = bK[:, 1]
    bK[:, 1] = np.dot(A, bK[:, 0])/np.linalg.norm(np.dot(A, bK[:, 0]))
    # print("second guess", bK[:, 1])
    cnt = 0
    # print(abs(sum((bK[:, 1] - bK[:, 0])/bK[:, 1])))
    while abs(sum((bK[:, 1] - bK[:, 0])/bK[:, 1])) > tol or cnt > 10000:
        bK[:, 0] = bK[:, 1]
        bK[:, 1] = np.dot(A, bK[:, 0])/np.linalg.norm(np.dot(A, bK[:, 0]))
        cnt += 1

    if DEBUG_PRINT: print("final error", abs(sum((bK[:, 1] - bK[:, 0])/bK[:, 1])), "cnt", cnt)

    eigenVal = bK[0, 1]
    eigenVec = bK[:, 1]/bK[0, 1]

    return eigenVal, eigenVec

# https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
def gramSchmidt(A, eigenVec):
    eigenVecs = [eigenVec]
    for i in range(1, A.shape[0]):
        print("i is", i)
        vec = A[:, i]
        print("vec is", vec)
        sum = 0
        for j in range(i):
            print("j is", j)
            proj = (np.dot(vec, eigenVecs[j])/np.dot(eigenVecs[j], eigenVecs[j])) * eigenVecs[j]
            print("projection of", vec, eigenVecs[j], "is", proj)
            sum += proj
            print("sum is", sum)

        eigenVecs.append(vec - sum)

    return eigenVecs

def plotA(A):
    x = np.arange(0, A.shape[0])
    yInds = np.arange(0, A.shape[1])
    y = np.zeros((A.shape[0]))
    nonZero = np.where(A != 0)
    print(nonZero)

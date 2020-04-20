# Micah Church
# Simple test case to try to find known solution of test case

import numpy as np
import sys
import math 
import matplotlib.pyplot as plt

import Model.CommonFunctions as cf

def main():

    DEBUG_PRINT = False

    xRange = [0, math.pi]
    yRange = [0, math.pi]

    # Descritize some points in space
    n = 4
    m = 4
    x = np.linspace(xRange[0], xRange[1], n)
    y = np.linspace(yRange[0], yRange[1], m)

    mesh = np.array(np.meshgrid(x, y, indexing="ij"))

    if DEBUG_PRINT: print("mesh is", mesh, mesh.shape)
    rowRange = np.arange(0, n)
    colRange = np.arange(0, m)
    # boundry values for quantum well [psi(xRange, 0), psi(xRange, m-1), psi(0, yRange), psi(n - 1, yRange)]
    boundVals = [1, 2, 1, 2]
    boundryConds = np.array([[[0]*m, colRange, boundVals[2]], [[n - 1]*m, colRange, boundVals[3]], \
        [rowRange, [0]*n, boundVals[0]], [rowRange, [m - 1]*n, boundVals[1]]])
    # boundryConds = np.reshape(np.array(boundryConds), (4, n, m, 1))
    if DEBUG_PRINT: print("boundryConds", boundryConds, boundryConds.shape)
    # assuming that all boundry conditions are the same for now
    # boundVal = 0
    BC = {}
    for i in range(boundryConds.shape[0]):
        for j in range(len(boundryConds[i, 0])):
            key = tuple([boundryConds[i, 0][j], boundryConds[i, 1][j]])
            if DEBUG_PRINT: print("key", key)
            value = boundryConds[i, 2]
            if DEBUG_PRINT: print("value", value)
            BC[key] = value

    A, psi = cf.getSpatialDescritization(mesh, BC, DEBUG_PRINT=DEBUG_PRINT)
    for a in A:
        print(a)
    # x = input()

    # Numerov Test
    # xRange = [-10, 10]
    # n1 = 1000
    # x = np.linspace(xRange[0], xRange[1], n1)
    # psi = [0, .1]
    # y = cf.numerov(xRange[0], xRange[1], psi, x)

    # Ask Dziubbek about this
    y = cf.numerov(xRange[0], xRange[1], [0, psi[0]], x)
    plt.plot(x, y)
    plt.show()

    a = np.array([[1, 3], [2, 2]], dtype=np.float64)
    # a = np.array([[1.5, 0, 1], [-.5, .5, -.5], [-.5, 0, 0]])
    # a = np.array([[1, 2], [0, 5]], dtype=np.float64)
    # get dominate eigenvector
    # eVal, eVec = cf.powerItteration(a, tol=1e-10)
    # print(eVal, eVec)
    # get minimum eigenvector
    # eVal2, eVec2 = cf.powerItteration(np.linalg.inv(a), tol=1e-10, DEBUG_PRINT=False)
    # print(eVal2, eVec2)
    # eVecs = cf.gramSchmidt(a, eVec)
    # # print(v1, v2, v3)
    eVals, eVecs = cf.getEigenVectors(A, tol=1e-100, DEBUG_PRINT=False)
    for i in range(len(eVecs)):
        print("eigenValue", eVals[i])
        print("eigenVector", eVecs[i])


if __name__ == "__main__":
    main()
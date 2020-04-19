# Micah Church
import numpy as np

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

def plotA(A):
    x = np.arange(0, A.shape[0])
    yInds = np.arange(0, A.shape[1])
    y = np.zeros((A.shape[0]))
    nonZero = np.where(A != 0)
    print(nonZero)

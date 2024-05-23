import math
# calculate distance between two atoms(type: Atom) in a simulation box of lx*ly*lz


def bondlen(A, B, lx, ly, lz):
    x1x = A.x[0]-B.x[0]
    x1y = A.x[1]-B.x[1]
    x1z = A.x[2]-B.x[2]
    while x1x > lx/2.0 : x1x = x1x-lx
    while x1x < -lx/2.0: x1x = x1x+lx
    while x1y > ly/2.0 : x1y = x1y-ly
    while x1y < -ly/2.0: x1y = x1y+ly
    while x1z > lz/2.0 : x1z = x1z-lz
    while x1z < -lz/2.0: x1z = x1z+lz
    dist = math.sqrt(x1x*x1x+x1y*x1y+x1z*x1z)
    return dist

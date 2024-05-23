import sys
import numpy as np

from cell import Cell
from mpi4py import MPI
comm = MPI.COMM_WORLD

class SimulationBox:
    def __init__(self):
        self.myatom = None
        self.mycell = None
        self.currentstep = None
        # dimension of the cells
        self.ncell = [0, 0, 0]
        self.L = [0.0, 0.0, 0.0]
        # origin of the simulation box
        self.origin = [0.0, 0.0, 0.0]
        self.nmols = 0

    def SetupCell(self, cellsize):
        if comm.Get_rank() == 0:
            print("Initiating setupcell\n")
            print(self.ncell, self.L)
        if self.nmols == 0:
            print("Simulation Box is empty!\n")
            sys.exit()
        for d in range(3):
            self.ncell[d] = int(self.L[d] // cellsize - 1)  # make it a little larger than necessary
            if self.ncell[d] < 3:
                print("self.ncell[", d, "]=", self.ncell[d], "\n",
                      "if self.ncell[i] < 3\n, there will be overcounting of pair "
                      "because looping 27 neighbour will lead to same cells\n", "decrease the cell size please!\n")
                sys.exit()
        if comm.Get_rank() == 0:
            print(self.ncell)
        self.mycell = np.ndarray(shape=self.ncell, dtype=object)

        # //etup NBR cells, Coding here
        # //////////////////////////////////////////
        # //////Yongjian code following/////////////
        # //////////////////////////////////////////
        # //oop of cells
        for i in range(self.ncell[0]):
            for j in range(self.ncell[1]):
                for k in range(self.ncell[2]):
                    # loop of neighbour cells
                    self.mycell[i][j][k] = Cell()
                    self.mycell[i][j][k].nnbr = 27
                    for pi in range(i - 1, i + 2):
                        for qi in range(j - 1, j + 2):
                            for ri in range(k - 1, k + 2):
                                # convert neighbour cells outside the range (periodic boundary condition)
                                p, q, r = pi, qi, ri
                                la = (pi - i + 1) * 9 + (qi - j + 1) * 3 + (ri - k + 1)
                                if la < 0 or la > 26:
                                    print("l:", la, "is wrong")
                                    sys.exit()
                                if p == -1:
                                    p = self.ncell[0] - 1
                                if p == self.ncell[0]:
                                    p = 0
                                if q == -1:
                                    q = self.ncell[1] - 1
                                if q == self.ncell[1]:
                                    q = 0
                                if r == -1:
                                    r = self.ncell[2] - 1
                                if r == self.ncell[2]:
                                    r = 0
                                # convert every neighbour's vector index into a scalor index
                                self.mycell[i][j][k].nbr[la] = p * self.ncell[1] * self.ncell[2] + q * self.ncell[2] + r
                                # printf("l has a value: %d\nnbr[l] has a value:%d\ni=%d j=%d k=%d\n",l,mycell[i][j][k]->nbr[l],i,j,k);
                                # printf("mycell[i][j][k] has %d neighbour\ni j k are:%d %d %d\n", mycell[i][j][k]->nnbr,i,j,k);

                                #print("i j k are ", i, j, k, "\n")

    # /////////////////////////////////////////
    # ///////Yongjian code ends////////////////
    # /////////////////////////////////////////

    def PrintInfo(self):
        print("SimulationBox: self.ncell(%d,%d,%d), L(%f,%f,%f), origin(%f,%f,%f),nmols(%d)\n" % (
            self.ncell[0], self.ncell[1], self.ncell[2], self.L[0], self.L[1], self.L[2], self.origin[0], self.origin[1], self.origin[2], self.nmols))


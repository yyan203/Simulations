from math import fmod
from basic_function import bondlen
from random import random
import numpy as np
from math import exp
from lammps import PyLammps, lammps

from mpi4py import MPI
comm = MPI.COMM_WORLD
split = comm.Split(comm.Get_rank(), comm.Get_rank())

# 2019-05-01  YJ
# throw all atoms into the cells of each frame

def throw_atom2cell(mysystem, iframe, cellsize):

    mysystem[iframe].SetupCell(cellsize)

    originx= mysystem[iframe].origin[0]
    originy= mysystem[iframe].origin[1]
    originz= mysystem[iframe].origin[2]
    if comm.Get_rank() == 0:  print("originx=", originx)
    cellsizex=(float)(mysystem[iframe].L[0]/mysystem[iframe].ncell[0])
    if comm.Get_rank() == 0:  print("cellsizex=", cellsizex)
    cellsizey=(float)(mysystem[iframe].L[1]/mysystem[iframe].ncell[1])
    cellsizez=(float)(mysystem[iframe].L[2]/mysystem[iframe].ncell[2])

    # loop of every atoms of one frame in dump file
    myatom = mysystem[iframe].myatom
    # print(type(myatom), len(mysystem[iframe].myatom), mysystem[iframe].nmols)
    for i in range(mysystem[iframe].nmols):
        x1 = myatom[i].x[0]
        x2 = myatom[i].x[1]
        x3 = myatom[i].x[2]
        # if i == 0: print("myatom:", myatom[i].x)
        # according to the atom's position, allocate them to the right cell. Since cellsizex is float number,
        # to make it safe, when the index from atom's position is calculated from cellsizex, bring it back to within the cell.
        p1=(int)((x1-originx)/cellsizex)
        if p1==mysystem[iframe].ncell[0]: p1 = p1 - 1
        p2=(int)((x2-originy)/cellsizey)
        if p2==mysystem[iframe].ncell[1]: p2 = p2 - 1
        p3=(int)((x3-originz)/cellsizez)
        if p3==mysystem[iframe].ncell[2]: p3 = p3 - 1
        memberid = mysystem[iframe].mycell[p1][p2][p3].nmember
        # print ("p1=%d p2=%d p3=%d\n",p1,p2,p3)
        # allocate the atom's id to the right cell
        # mysystem[iframe]->mycell[p1][p2][p3]->member[memberid]=myatom[i]->id
        mysystem[iframe].mycell[p1][p2][p3].member[memberid] = i
        # print("mycell%d%d%d->member[%d]:%d\n",p1,p2,p3,memberid,mysystem[iframe]->mycell[p1][p2][p3]->member[memberid])

        # if (p1==0 && p2==0 && p3==0){print("mycell-000 has atom(id): %d\n",mysystem[iframe]->mycell[p1][p2][p3]->member[mysystem[iframe]->mycell[p1][p2][p3]->nmember])}
        # if(i==400){print("nmember of mycell[%d][%d][%d] is %d\n",p1,p2,p3,mysystem[iframe]->mycell[p1][p2][p3]->nmember)}

        # increase the nmember of cell when a new atom is added into it.
        mysystem[iframe].mycell[p1][p2][p3].nmember += 1
    if comm.Get_rank() == 0: print("\nAtoms thrown to cells finished!!\n")

# get number density fluctuation by sampling N random points and count atoms belonging to sphere of radius=R locating at these random points

def get_density_fluctuation(mysystem, iframe, sphereR, sampleN):

    count = [ 0.0 for _ in range(sampleN)]
    originx= mysystem[iframe].origin[0]
    originy= mysystem[iframe].origin[1]
    originz= mysystem[iframe].origin[2]
    Lx= mysystem[iframe].origin[0]
    Ly= mysystem[iframe].origin[1]
    Lz= mysystem[iframe].origin[2]

    for i in range(sampleN):
        x1 = random()*Lx + originx
        x2 = random()*Ly + originy
        x3 = random()*Lz + originz
        p1=(int)((x1-originx)/Lx)
        if p1==mysystem[iframe].ncell[0]: p1 = p1 - 1
        p2=(int)((x2-originy)/Ly)
        if p2==mysystem[iframe].ncell[1]: p2 = p2 - 1
        p3=(int)((x3-originz)/Lz)
        if p3==mysystem[iframe].ncell[2]: p3 = p3 - 1
        memberid = mysystem[iframe].mycell[p1][p2][p3].nmember



# 2020-02-24  YJ
# for WWW algorithm implementation
# findn all Si-O bonds, Si-O-Si and O-Si-O bonds using a bond cutoff of
# cutoff Si-O is the distance for Si-O bond, typically 2.2 A


from collections import defaultdict

def get_bond_list(mysystem, iframe, Si_type, O_type, cutoff_Si_O = 2.2):

    mysystem_ = mysystem[iframe]
    myatom = mysystem[iframe].myatom
    resSi = defaultdict(list)
    resO = defaultdict(list)

    lx, ly, lz = mysystem_.L[0], mysystem_.L[1], mysystem_.L[2]
    ncell = []
    # initialize array ncell[3]
    for iii in range(3):
        ncell.append(mysystem_.ncell[iii])

    neigh_O = set() # O atom
    neigh_Si = set() # Si atom
    ################## do analysis
    # loop among different cell
    for cellx in range(ncell[0]):
        for celly in range(ncell[1]):
            for cellz in range(ncell[2]):
                nmember = mysystem_.mycell[cellx][celly][cellz].nmember
                # //loop among all the atoms contained in one cell
                for imember in range(nmember):
                    comparei = mysystem_.mycell[cellx][celly][cellz].member[imember]
                    itype = myatom[comparei].type # get type
                    if itype == Si_type:
                        neigh_O.clear()
                    if itype == O_type:
                        neigh_Si.clear()

                        # loop among all neighbour cells
                    for ii in range(27):
                        c = mysystem_.mycell[cellx][celly][cellz].nbr[ii]
                        ncellx = (int) (c / (ncell[1] * ncell[2]))
                        ncelly = (int) (fmod((int) (c / ncell[2]), ncell[1]))
                        ncellz = (int) (fmod(c, ncell[2]))
                        inmember = mysystem_.mycell[ncellx][ncelly][ncellz].nmember

                        # loop among all the neighbour's atoms
                        for j in range(inmember):
                            comparej = mysystem_.mycell[ncellx][ncelly][ncellz].member[j]
                            jtype = myatom[comparej].type  # get jtype
                            if comparei != comparej:
                                dist = bondlen(myatom[comparei], myatom[comparej], lx, ly, lz)
                                if jtype == Si_type and dist <= cutoff_Si_O:
                                    neigh_Si.add(myatom[comparej].id)
                                if jtype == O_type and dist <= cutoff_Si_O:
                                    neigh_O.add(myatom[comparej].id)

                    if itype == Si_type:
                        # each Si connect to 4 O
                        assert len(neigh_O) == 4
                        resSi[myatom[comparei].id] = []
                        for it in neigh_O:
                            resSi[myatom[comparei].id].append(it)

                    if itype == O_type:
                        # each O connect to 2 Si
                        #if not len(neigh_Si) == 2:
                            #print(neigh_Si)
                        resO[myatom[comparei].id] = []
                        for it in neigh_Si:
                            resO[myatom[comparei].id].append(it)
    return resSi, resO




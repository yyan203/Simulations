
import numpy as np
from Atom import Atom
from mpi4py import MPI
comm = MPI.COMM_WORLD


# read atoms coordination information

def read_lata(mysystem, filename, ifcharge, max_frame_to_read):

    # print(filename)
    f = open(filename)
    frame, nmols, starttime, endtime = -1, 0, 0, 0
    # print("ori", type(mysystem[0]), len(mysystem))

    while True:
        if frame + 1 >= max_frame_to_read:
            f.close()
            print("max frame:", frame, "reached")
            return frame, nmols, starttime, endtime
        line = f.readline()
        if not line:
            break
        t = line.split(' ')
        if t[0] == "ITEM:":
            frame += 1
            # print(t[0], t, frame)
            line = f.readline()
            t = line.split(' ')
            if frame == 0:
                starttime = int(t[0])
            endtime = int(t[0])
            line = f.readline()
            # number of atom
            line = f.readline()
            t = line.split(' ')
            nmols = int(t[0])
            # assign number of atoms
            mysystem[frame].nmols = nmols
            line = f.readline()

            # read box information
            line = f.readline()
            t = line.split(' ')
            mysystem[frame].origin[0] = float(t[0])
            # print("=", mysystem[0].origin[0], float(t[0]), frame)
            mysystem[frame].L[0] = float(t[1]) - float(t[0])
            line = f.readline()
            t = line.split(' ')
            mysystem[frame].origin[1] = float(t[0])
            mysystem[frame].L[1] = float(t[1]) - float(t[0])
            line = f.readline()
            t = line.split(' ')
            mysystem[frame].origin[2] = float(t[0])
            mysystem[frame].L[2] = float(t[1]) - float(t[0])

            line = f.readline()
            # read num atoms
            mysystem[frame].myatom = []
            for i in range(nmols):
                line = f.readline()
                t = line.split(' ')
                if ifcharge:
                    mysystem[frame].myatom.append(Atom(int(t[0]), int(t[1]), float(t[3]), float(t[4]), float(t[5])))
                else:
                    mysystem[frame].myatom.append(Atom(int(t[0]), int(t[1]), float(t[2]), float(t[2]), float(t[4])))
            # print("x=", mysystem[0].myatom[0].x)

    f.close()
    print("LATA file import:", nmols, " atoms!\n")
    return frame, nmols, starttime, endtime

# read data file
# assume single frame
def read_data(mysystem, filename, ifcharge, style, frame = 0):

    # print(filename)
    f = open(filename)
    # import re

    nmols = 0
    linenum = 0
    while True:

        line = f.readline()
        if not line:
            break
        t = line.strip().split()
        #  linenum += 1
        #  if linenum < 10:
        #      print(linenum)
        #      print(t)
        #      #print(len(t))
        if len(t) >= 2 and t[1] == "atoms":
            nmols = int(t[0])
            #print(t)
            mysystem[frame].nmols = nmols
        if len(t) >= 4 and t[3] == "xhi":
            #print(t)
            mysystem[frame].origin[0] = float(t[0])
            mysystem[frame].L[0] = float(t[1]) - float(t[0])
        if len(t) >= 4 and t[3] == "yhi":
            mysystem[frame].origin[1] = float(t[0])
            mysystem[frame].L[1] = float(t[1]) - float(t[0])
        if len(t) >= 4 and t[3] == "zhi":
            mysystem[frame].origin[2] = float(t[0])
            mysystem[frame].L[2] = float(t[1]) - float(t[0])

        if len(t) >= 1 and t[0] == "Atoms":
            # read num atoms
            mysystem[frame].myatom = []
            for i in range(nmols+1):
                line = f.readline()
                t = line.strip().split()
                if len(t) < 6:
                    continue
                if style != "full":
                    if ifcharge:
                        mysystem[frame].myatom.append(Atom(int(t[0]), int(t[1]), float(t[3]), float(t[4]), float(t[5])))
                    else:
                        mysystem[frame].myatom.append(Atom(int(t[0]), int(t[1]), float(t[2]), float(t[2]), float(t[4])))
                else:
                    if ifcharge:
                        # print(t, style)
                        mysystem[frame].myatom.append(Atom(int(t[0]), int(t[2]), float(t[4]), float(t[5]), float(t[6])))
                    else:
                        mysystem[frame].myatom.append(Atom(int(t[0]), int(t[2]), float(t[3]), float(t[4]), float(t[5])))

            # print("x=", mysystem[0].myatom[0].x)

    f.close()
    if comm.Get_rank() == 0: print("DATA file import:", nmols, " atoms!\n")
    return frame, nmols, 0, 0


# read all Si-O bond information -> only Si-O or O-Si are considered, not Si-O-Si or O-Si-O

# assuming data file is "full" style with information of atom: ID  Molecule_ID  type q x y z ......
# assuming type 1 is O and type 2 is Si; only two types of atoms are allowed for now


def read_bond(mysystem, filename, frame=0):

    from collections import defaultdict
    # print(filename)
    f = open(filename)
    resSi = defaultdict(list)
    resO = defaultdict(list)
    nmols = 0  # num of atoms
    nbonds = 0  # num of bonds
    linenum = 0
    atomtype = {}
    while True:

        line = f.readline()
        if not line:
            break
        t = line.strip().split()

        if len(t) >= 2 and t[1] == "atoms":
            nmols = int(t[0])
        if len(t) >= 2 and t[1] == "bonds":
            nbonds = int(t[0])
        if len(t) >= 1 and t[0] == "Atoms":
            # read atom type
            for i in range(nmols+1):
                line = f.readline()
                t = line.strip().split()
                if len(t) < 6:
                    continue
                atomtype[int(t[0])] = int(t[2])

        if len(t) >= 1 and t[0] == "Bonds":
            # read num bonds
            for i in range(nbonds+1):
                line = f.readline()
                t = line.strip().split()
                if len(t) != 4:
                    continue
                if atomtype[int(t[2])] == 2:
                    resSi[int(t[2])].append(int(t[3]))
                    resO[int(t[3])].append(int(t[2]))
                else:
                    resSi[int(t[3])].append(int(t[2]))
                    resO[int(t[2])].append(int(t[3]))
            # print("x=", mysystem[0].myatom[0].x)
    f.close()
    if comm.Get_rank() == 0:  print("DATA file import:", nbonds, " bonds!\n")
    return resSi, resO

# read RDF information and store in a multi-dimentional list

# 1st line is comments
# index distance g(r)_i_j  (i, j vary from 1 to N, N is number of type of atoms)
# 0   0.00   0.000
# -   ----   -----
# 0   0.00   0.000

# return maximum_distance  and  distance_step
def read_RDF(filename, container, type1, type2, line_to_ignore):

    # print(filename)
    f = open(filename)
    linenum, start, end = 0, 0.0, 0.0
    # print("ori", type(mysystem[0]), len(mysystem))

    while True:
        if linenum < line_to_ignore:
            line = f.readline()
            linenum += 1
            continue
        else:
            break

    while True:
        line = f.readline()
        if not line:
            break
        t = line.split(' ')
        if int(t[0]) == 1:
            start = float(t[1])
        if int(t[0]) == 2:
            end = float(t[1])
        maximum_distance = float(t[1])
        np.append(container, float(t[2]))

    f.close()
    print("-------  G(r):", type1, type2, " ------\n")
    return maximum_distance, end - start

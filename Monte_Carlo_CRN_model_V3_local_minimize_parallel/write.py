
import numpy as np
from Atom import Atom


# write "Bonds ..." and "Angles ..." for lammps datafile

def write_bonds(bond_list_Si, bond_list_O):

    # write Si-O bonds

    f = open("BONDFILE.tmp", 'w')
    f.write('\nBonds\n\n')
    countBond = 1
    for i in bond_list_Si:
        for j in bond_list_Si[i]:
            f.write("%7d %2d %6d %6d\n" % (countBond, 1, i, j))
            countBond += 1
    f.write('\n\nAngles\n\n')

    countBond = 1
    # write Angular bonds: O-Si-O and Si-O-Si
    for i in bond_list_Si:
        for j in bond_list_Si[i]:
            for k in bond_list_Si[i]:
                if k > j:
                    f.write("%7d %2d %6d %6d %6d\n" % (countBond, 1, j, i, k))
                    countBond += 1
    for i in bond_list_O:
        for j in bond_list_O[i]:
            for k in bond_list_O[i]:
                if k > j:
                    f.write("%7d %2d %6d %6d %6d\n" % (countBond, 2, j, i, k))
                    countBond += 1
    f.write('\n')
    f.close()
    print("-------  Bond file written  ------\n")
    return True

# write data file for lammps


def write_data(mysystem, frame, bond_list_Si, bond_list_O):

    # write Si-O bonds
    nSi, nO = len(bond_list_Si), len(bond_list_O)
    f = open("DATAFILE.initial", 'w')
    f.write("SiO2 initial configuration\n\n")
    f.write("%10d atoms\n" % (nSi + nO))
    f.write("%10d bonds\n" % (nSi * 4))
    f.write("%10d angles\n" % (nSi * 6 + nO))
    f.write("\n%10d atom types\n" % (2))
    f.write("%10d bond types\n" % (1))
    f.write("%10d angle types\n\n" % (2))
    f.write("%11.6f   %11.6f  xlo xhi\n" % (mysystem[frame].origin[0], mysystem[frame].L[0] + mysystem[frame].origin[0]))
    f.write("%11.6f   %11.6f  ylo yhi\n" % (mysystem[frame].origin[1], mysystem[frame].L[1] + mysystem[frame].origin[1]))
    f.write("%11.6f   %11.6f  zlo zhi\n" % (mysystem[frame].origin[2], mysystem[frame].L[2] + mysystem[frame].origin[2]))
    # f.write("%11.6f   %11.6f  %11.6f  xy xz yz\n" % (0.0, 0.0, 0.0))
    # f.write("%10d dihedrals\n" % 0)
    # f.write("%10d impropers\n" % 0)
    f.write('\n\nMasses\n\n1  15.9994  # O\n2  28.0855  # Si\n\n')
    f.write('\nAtoms\n\n')

    countAtom = 1

    for i in mysystem[frame].myatom:
        f.write("%7d   111  %2d  0.000 %10.5f %10.5f %10.5f\n" % (i.id, i.type, i.x[0], i.x[1], i.x[2]))
        countAtom += 1

    f.write('\nBonds\n\n')

    countBond = 1

    for i in bond_list_Si:
        for j in bond_list_Si[i]:
            f.write("%7d %2d %6d %6d\n" % (countBond, 1, i, j))
            countBond += 1
    f.write('\n\nAngles\n\n')

    countBond = 1
    # write Angular bonds: O-Si-O and Si-O-Si
    for i in bond_list_Si:
        for j in bond_list_Si[i]:
            for k in bond_list_Si[i]:
                if k > j:
                    f.write("%7d %2d %6d %6d %6d\n" % (countBond, 1, j, i, k))
                    countBond += 1
    for i in bond_list_O:
        for j in bond_list_O[i]:
            for k in bond_list_O[i]:
                if k > j:
                    f.write("%7d %2d %6d %6d %6d\n" % (countBond, 2, j, i, k))
                    countBond += 1
    f.write('\n')
    f.close()
    print("-------  Data file written  ------\n")
    return True


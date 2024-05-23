# Implement Vink's method (PRB 67, 2003 245201) to generate random network of SiO2
# reverse Monte-Carlo method
# needed Input files:

# (1) configuration file -- position of each atom  xyz file
# (2) bond configuration file with bond list of each atom Si and O
# (1) assume RDF of each pair has been calculated by other software with a format:

#(1) input file format: lata4olivia
# number of Atoms
# index distance g(r)_i_j  (i, j vary from 1 to N, N is number of type of atoms)
# 0   0.00   0.000
# -   ----   -----
# 0   0.00   0.000
#(2) bond list for each atoms

#(2) choose  R  using the third position when g(r)_i_j== 1.0, following (Wedberg, Molecular Simulation, Vol. 36, No.15, 2010, 1243)


from math import exp, log10
import sys, argparse, pdb
import random as rd
import numpy as np
# from Atom import Atom
from SimulationBox import SimulationBox
from read import read_lata, read_data, read_bond
from write import write_bonds, write_data
# from cell import Cell
# from Functions import throw_atom2cell
import Functions as fun
import random
from mpi4py import MPI
from lammps import PyLammps, lammps
import cluster_analysis as mycluster
import time

from bond_transposition import bondTransposition

MAXDUMP = 2
comm = MPI.COMM_WORLD
split = comm.Split(comm.Get_rank(), comm.Get_rank())

DEBUG_MODE = False

# read single frame bonds information: type atom1-ID  atom2-ID
# read bond info
#  1  1  6660
#  3  6655  8703

def read_bonds(self, filename):
    with open(filename) as f:
        for index, line in enumerate(f):
            t = line.rstrip().split()
            if int(t[1]) not in self.bonds:
                self.bonds[int(t[1])] = set([int(t[2])])
            else:
                self.bonds[int(t[1])].add(int(t[2]))
            if int(t[2]) not in self.bonds:
                self.bonds[int(t[2])] = set([int(t[1])])
            else:
                self.bonds[int(t[2])].add(int(t[1]))
            self.bondnum += 1
    print("Import ", self.bondnum, " bonds!")
    # print(self.bonds)

#  after delete atoms and add new atoms, the maximum ID of atoms might larger than self.atomnum
#  because some ID has no atoms associated
#  assuming there are only  Zn  H  C  N elements


def outputxyz(self, xyzfile):
    Zn, H, C, N = {}, {}, {}, {}
    # nZn, nH, nC, nN = 1, 1, 1, 1
    j = 1
    # print(self.atoms[10].type_,"yes here")
    for i in self.atoms:
        if self.atoms[i].type_ == "Zn":
            Zn[j] = self.atoms[i].id_;
            j += 1
    for i in self.atoms:
        if self.atoms[i].type_ == "H":
            H[j] = self.atoms[i].id_;
            j += 1
    for i in self.atoms:
        if self.atoms[i].type_ == "C":
            C[j] = self.atoms[i].id_;
            j += 1
    for i in self.atoms:
        if self.atoms[i].type_ == "N":
            N[j] = self.atoms[i].id_;
            j += 1

    f = open(xyzfile, 'w')
    f.write('%d\n' % self.atomnum)
    f.write('add benzine to Zif4\n')
    for i in sorted(Zn.keys()):
        xyz = self.atoms[Zn[i]].xyz_
        # print(Zn[i], xyz)
        f.write("%d %s %f %f %f\n" % (i, self.atoms[Zn[i]].type_, xyz[0], xyz[1], xyz[2]))
    for i in sorted(H.keys()):
        xyz = self.atoms[H[i]].xyz_
        # print(xyz)
        f.write("%d %s %f %f %f\n" % (i, self.atoms[H[i]].type_, xyz[0], xyz[1], xyz[2]))
    for i in sorted(C.keys()):
        xyz = self.atoms[C[i]].xyz_
        f.write("%d %s %f %f %f\n" % (i, self.atoms[C[i]].type_, xyz[0], xyz[1], xyz[2]))
    for i in sorted(N.keys()):
        xyz = self.atoms[N[i]].xyz_
        f.write("%d %s %f %f %f\n" % (i, self.atoms[N[i]].type_, xyz[0], xyz[1], xyz[2]))
    f.close()

def outputbond(self, bondfile):
    oldID2newID = {}
    newID2oldID = {}
    j = 1
    newbonds = {}
    bondtype = {"Zn-N": 1, "N-Zn": 1, "H-C": 2, "C-H": 2, "C-N": 3, "N-C": 3, "C-C": 4}
    for i in self.atoms:
        if self.atoms[i].type_ == "Zn":
            oldID2newID[self.atoms[i].id_] = j
            newID2oldID[j] = self.atoms[i].id_
            j += 1
    for i in self.atoms:
        if self.atoms[i].type_ == "H":
            oldID2newID[self.atoms[i].id_] = j
            newID2oldID[j] = self.atoms[i].id_
            j += 1
    for i in self.atoms:
        if self.atoms[i].type_ == "C":
            oldID2newID[self.atoms[i].id_] = j
            newID2oldID[j] = self.atoms[i].id_
            j += 1
    for i in self.atoms:
        if self.atoms[i].type_ == "N":
            oldID2newID[self.atoms[i].id_] = j
            newID2oldID[j] = self.atoms[i].id_
            j += 1
    # print("old2new ID:",oldID2newID)
    j = 1
    while j <= self.maxID:
        if j in self.bonds:
            newJ = oldID2newID[j]
            neighbour = self.bonds[j]
            for k in neighbour:
                newK = oldID2newID[k]
                assert newJ is not newK, "bonds error, same atoms!"
                if newJ < newK:
                    if newJ in newbonds:
                        newbonds[newJ].add(newK)
                    else:
                        newbonds[newJ] = set([newK])
                else:
                    if newK in newbonds:
                        newbonds[newK].add(newJ)
                    else:
                        newbonds[newK] = set([newJ])
        j += 1

    f = open(bondfile, 'w')
    # print(newbonds)
    for i in newbonds:
        for j in newbonds[i]:
            if i < j:
                # print(newID2oldID[i], newID2oldID[j], self.atoms[newID2oldID[i]].type_, self.atoms[newID2oldID[j]].type_ )
                typ = bondtype[self.atoms[newID2oldID[i]].type_ + "-" + self.atoms[newID2oldID[j]].type_]
                f.write("%d %d %d\n" % (int(typ), i, j))
    f.close()

# delete atoms and its associated bonds
def delete_atom(self, atomID):
    assert atomID in self.atoms, "Atom %d does not exist!" % atomID
    neigh = self.bonds.pop(atomID)
    self.bondnum -= len(neigh)
    # print(self.bondnum)
    for i in neigh:
        if len(self.bonds[i]) == 1:
            self.bonds.pop(i)
        else:
            self.bonds[i].remove(atomID)
    self.atoms.pop(atomID)
    self.atomnum -= 1

def add_atom(self, typ, coord):
    self.atomnum += 1
    newid = self.atomnum
    while newid in self.atoms:
        newid += 1
    self.atoms[newid] = Atom(newid, typ, coord[0], coord[1], coord[2])
    self.maxID = max(self.maxID, newid)
    return newid

def add_bond(self, id1, id2):
    self.bondnum += 1
    assert id1 not in self.bonds or id2 not in self.bonds[id1], "Bond %d -> %d already exist!" % (id1, id2)

    if id1 not in self.bonds:
        self.bonds[id1] = set([id2])
    else:
        self.bonds[id1].add(id2)
    if id2 not in self.bonds:
        self.bonds[id2] = set([id1])
    else:
        self.bonds[id2].add(id1)

    # read atoms information and bond information
def parseOptions(comm):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("LATA", help="lata_file(typical format: ID type q x y z ...")
    parser.add_argument("if_data", type=int, default= 0, help="1:if the LATA file is a datafile for lammps")
    parser.add_argument("out", help="output file prefix name")
    parser.add_argument("Cutoff", type=float, help="cutoff to determine Si-O bonds")
    parser.add_argument("Charge", type=int, help="if file contain charge [1/0]")
    parser.add_argument('-c', '--cellsize', type=float, help="cellsize to divide the simulation box")
    parser.add_argument('-fr', '--frame', type=int, help="use nth frame")
    parser.add_argument('-b', '--bond_list', type=int, help="(1: yes; 0: no) whether to read bond from data file FUN# 3")
    parser.add_argument('-in', '--input', type=str, help="lammps input file for FUN# 3")
    parser.add_argument('-indat', '--input_data', type=str, help="data file containing Bonds info for FUN# 3")
    #parser.add_argument('-indat_BKS', '--input_data_BKS', type=str, help="input_data file for BKS relaxation for FUN# 3")
    parser.add_argument('-dtst', '--input_data_style', type=str, help="full(include molecule-ID) or atom(no molecule) "
                                                                      "style of atom information: for FUN# 3")
    parser.add_argument('-sw', '--switch', type=int, help="number of successful bond switch for function 3")
    parser.add_argument('-res', '--restart', type=int, help="0: new start or "
                                                            "Number of successful bond transpositon( > 0):"
                                                            "get this from first number of last row of "
                                                            "FILE:[PE_Press_vs_success_Bond_change.dat]"
                                                            "restart from : restart_original_position.tmp ")
    parser.add_argument('-r_out', '--outter_radius', type=float,
                        help="outter radius cutoff for local relaxation in CRN WWW method）")
    parser.add_argument('-r_in',  '--inner_radius', type=float,
                        help="inner radius cutoff for local relaxation in CRN WWW method）")
    parser.add_argument('-BKS',  '--bks_flag', type=float,
                        help="whether to use BKS to check energy and reject or accept the bond transposition: 1:yes, 0:no）")
    parser.add_argument('-target', '--target_times', type=int, help="target transposition times")
    parser.add_argument('-T', '--temp', type=float, help="temperature for FUN# 3 （CRN WWW method）")
    parser.add_argument('-T_BKS', '--temp_BKS', type=float, help="temperature(BKS relaxation) for FUN# 3 （CRN WWW method）")
    parser.add_argument('-f', '--fun', type=int, help="[FUNCTIONS]\n"
                                                      "1. number of nth neighbor (type Si) from a central atom (type Si)\n"
                                                      "2. density fluctuation using a spherical sample size(radius=R)\n"
                                                      "3. generate CRN from bond switch\n")
    args = None
    try:
        if comm.Get_rank() == 0:
            args = parser.parse_args()
    finally:
        args = comm.bcast(args, root=0)
    if args is None:
        exit(0)
    return args


# Monte Carlo Metropolis

def main():
    rank = comm.Get_rank()
    size = comm.Get_size()
    args = parseOptions(comm)
    if rank == 0:
        print("Choose function:", args.fun)
        print("open files:", args.LATA)
    mysystem = [SimulationBox() for _ in range(MAXDUMP)]  # remember not to use [SimulationBox()] * 10 because this initiate 10 references to the same object
    if comm.Get_rank() == 0: print("sim_box_origin:", mysystem[0].origin)
    if not args.if_data:
        (frame, nmols, starttime, endtime) = read_lata(mysystem, args.LATA, args.Charge, 11)
    else:
        (frame, nmols, starttime, endtime) = read_data(mysystem, args.input_data, args.Charge, args.input_data_style, 0)


    # the following part (not always necessary) is used to generate bond list between Si and O, see the "write_bonds" whose output
    # is the bond list which can be appended to create data file for lammps

    fun.throw_atom2cell(mysystem, 0, args.cellsize)

    # read Si-O bond pair from data file or generate on my own
    if args.bond_list:
        bond_list_Si, bond_list_O = read_bond(mysystem, args.input_data, 0)
    else:
        bond_list_Si, bond_list_O = fun.get_bond_list(mysystem, 0, 2, 1, 2.2)
    if rank == 0:
        print("# of Si:", len(bond_list_Si), "# of O:", len(bond_list_O))
    write_bond_list = True
    if write_bond_list and rank == 0:
        assert write_bonds(bond_list_Si, bond_list_O)
    #exit()
    comm.Barrier()
    bondTran = bondTransposition(mysystem, 0, args.input, bond_list_Si, bond_list_O,
                                 args.temp, args.temp_BKS, args.restart,
                                 args.outter_radius, args.inner_radius, args.bks_flag)

    # the above part is not necessary if a data file with bond information is provided.

    # check_cal = comm.allreduce(check_cal)

    from pickle import load, dump

    write_bond_change = True
    #if rank == 0 and write_bond_change:
    if rank == 0 and write_bond_change and not args.restart:
        f = open("Bond_transposition_history.dat", 'w')
        f.write("# T(K)=%d,  T(K)_BKS=%d\n" % (args.temp, args.temp_BKS))
        f.write("#step  Si1 Si2 Si3 Si4 O2 O1 O4 O5 O3 O6 O7\n")
        f.close()

    if comm.Get_rank() == 0:
        if args.restart:
            ff = open("PE_Press_vs_success_Bond_change.dat", 'a+')
            time_counter = open("Time_step.dat", "a+")
        else:
            time_counter = open("Time_step.dat", "w")
            ff = open("PE_Press_vs_success_Bond_change.dat", 'w')
            ff.write("# T(K)=%d,  T(K)_BKS=%d\n" % (args.temp, args.temp_BKS))
            ff.write("#step  PE(eV) press(GPa) PE(eV)_BKS press(GPa)_BKS volume\n")
            ff.write("%6d  %9.4f  %3.3f   %9.4f %3.3f   %7.1f\n" %
                     (0, bondTran.attribute[1], bondTran.attribute[0]/10000.0,
                      bondTran.attribute[5], bondTran.attribute[7]/10000.0, bondTran.attribute[8]))
            ff.flush()  # save to file

    comm.Barrier()

    # for tracking time elasped
    start_time = time.time()
    comm.Barrier()
    #exit()

    random.seed(27848 + rank * rank)
    #random.seed(27848)
    if args.restart:
        if comm.Get_rank() == 0:
            print("WARNING!!!  restart from a previous running using: restart_original_position.tmp")
            print("If this is not what you want, use Ctrl + D to cancel")
            sys.stdout.flush()
        time.sleep(1)
        with open('cpu.'+str(comm.Get_rank())+'.random_number_state_for_restart.obj', 'rb') as f:
            random.setstate(load(f))

    #try_limit = 300
    try_limit = 500000
    try_times = 1
    #target_change = 9000
    target_change = args.target_times

    #  start try bond transposition
    while True and try_times <= try_limit and bondTran.get_success() < target_change:
        sys.stdout.flush()  # print results to screen
        if comm.Get_rank() == 0 and try_times % 10 == 0:
            print('–––––––>Starting bond transposition [Rounds] = %d, Time(seconds): %d'
                  % (try_times, int(time.time() - start_time)))

        bondTran.generate_random_Si_O_list_for_www()
        if not bondTran.bond_trans_list[0]:
            print("transposition_list not obtained!!")
            print("Try 100 times but cannot find bonds to transpose (maybe due to 3-member ring???)")
            exit()

        # O2 is the central O atom which connect two Si atoms that are involved in the WWW algorithm
        bondTran.bond_transposition_WWW_parallel()

        comm.Barrier()
        bondTran.calculation_flag = comm.allreduce(bondTran.calculation_flag)
        bondTran.cutoff = comm.allreduce(bondTran.cutoff, op=MPI.SUM)
        bondTran.cutoff = bondTran.cutoff / float(comm.Get_size())
        comm.Barrier()
        try_times += 1
        if len(np.nonzero(bondTran.calculation_flag == 1)[0]) == 0:
            if comm.Get_rank() == 0 and DEBUG_MODE:
                print("[All CPU try fail]:", bondTran.calculation_flag)
                print("--------------------- %d seconds --------------- try rounds ––––––––––> |%6d| "
                      % (int(time.time() - start_time), try_times))
            continue
        #if True:
        #    print(bondTran.bond_trans_list)
        #    exit()
        comm.Barrier()
        #if comm.Get_rank() == 0:
            #print("MY_GOD")
        success_cpu = np.nonzero(bondTran.calculation_flag == 1)[0][0]  # now select the first one
        bondTran.bond_trans_list = comm.bcast(bondTran.bond_trans_list, root=success_cpu)
        comm.Barrier
        O2, Si1, Si2, O1, O4, O5, O3, O6, O7, Si3, Si4 = bondTran.bond_trans_list
        #bondTran.bond_list_Si = comm.bcast(bondTran.bond_list_Si, root=success_cpu)
        #bondTran.bond_list_O  = comm.bcast(bondTran.bond_list_O, root=success_cpu)
        bondTran.bond_list_Si[Si1] = [O2, O3, O4, O5]
        bondTran.bond_list_Si[Si2] = [O2, O1, O6, O7]
        bondTran.bond_list_O[O1] = [Si2, Si3]
        bondTran.bond_list_O[O3] = [Si1, Si4]

        # substitute with new globally relaxed bond_transposed configuration
        if comm.Get_rank() == success_cpu:
            #bondTran.lmp.command("read_dump new_SiO2_confi.data 0 x y z replace yes box no")
            #print(bondTran.attribute)
            if bondTran.BKS_minimize_flag:
                bondTran.relax_volume_with_BKS()
                print("BKS pressure minimization, final pressure (GPa): %2.4f, Volume: %8.0f" %
                      (bondTran.get_attribute(7)/10000.0, bondTran.get_attribute(8)))
            bondTran.lmp.command("write_restart restart_original_position.tmp")
            bondTran.lmp.command("write_dump all custom current_SiO2_confi.data id type q x y z")
        comm.Barrier()
        #exit()

        bondTran.attribute = comm.bcast(bondTran.attribute, root=success_cpu)
        if comm.Get_rank() == 0:
            print("\n[Target_#_bond_transpose]: %d  [Successful # of bond_transpose] ===========> |%5d| \n"
                  % (args.target_times, bondTran.get_success() + 1))
            print("Sphere_cutoff:%2.1f, %2.1f #local_atoms: %3d"
                  % (bondTran.cutoff[0], bondTran.cutoff[1], bondTran.cutoff[2]))
            print("[some CPU try success]:", bondTran.calculation_flag)
            print("[Attribute CPU root]:", bondTran.attribute)
        time.sleep(.1)  # wait for 0.2 sec

        # this command must be issued after "restart_original_position.tmp" has been generated
        bondTran.reset_calculation_flag()  # reset all cpu to start from the successful last calculation

        ## store random number state (binary) for restart the simulation
        ## !!! Because the random number status is only dump when a successful bond_transposition is found
        ## Therefore, exact restart is not possible using these random number status
        with open('cpu.'+str(comm.Get_rank())+'.random_number_state_for_restart.obj', 'wb') as f:
            dump(random.getstate(), f)
        if comm.Get_rank() == 0:
            time_counter.write("%9.0f seconds,  %6d valid_step,  %8d trial_step * %d CPUs  "
                               "Sphere_cutoff[out] %2.2f [in] %2.2f average_atom#_in_inner sphere: %d\n"
                               % (time.time() - start_time, bondTran.get_success(), try_times, comm.Get_size(),
                                  bondTran.cutoff[0], bondTran.cutoff[1], bondTran.cutoff[2]))
            time_counter.flush()  # save to file
            if write_bond_change:
                f = open("Bond_transposition_history.dat", 'a+')
                f.write("%6d  %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d\n"
                        % (bondTran.success, Si1, Si2, Si3, Si4, O2, O1, O4, O5, O3, O6, O7))
                f.close()
            # success, PE_keating, Press, PE_BKS, PE_Press, Vol
            ff.write("%6d   %9.4f %3.3f   %9.4f %3.3f   %7.1f\n" %
                     (bondTran.get_success(), bondTran.attribute[2], bondTran.attribute[0]/10000.0,
                      bondTran.attribute[6], bondTran.attribute[7]/10000.0, bondTran.attribute[8]))
            ff.flush()  # save to file

        if comm.Get_rank() == 0:
            if bondTran.get_success() % 10 == 0:
                name_temp = "transposition.data."+str(bondTran.get_success())
                bondTran.lmp.command("write_data " + name_temp)
                name_temp = "lata4olivia."+str(bondTran.get_success())
                bondTran.lmp.command("write_dump all custom " + name_temp + " id type q x y z")
            bondTran.lmp.command("write_data " + "restart_configure_file.data")

    comm.Barrier()
    if comm.Get_rank() == 0:
        ff.close()
        time_counter.close()  # save to file
        print("While loop ends!!")
        print("\n[Job completed!!!], do %d successful bond transposition" % bondTran.get_success())
        # L.write_data("after_switch.data")

    MPI.Finalize()


# yj remote
if __name__ == "__main__":
    main()

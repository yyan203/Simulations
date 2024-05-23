import sys
import time
from math import fmod
import numpy as np
from scipy import optimize
np.set_printoptions(precision=3, suppress=True)
from math import exp
import random
from lammps import lammps, PyLammps
from mpi4py import MPI
from basic_function import bondlen
from collections import defaultdict
comm = MPI.COMM_WORLD
split = comm.Split(comm.Get_rank(), comm.Get_rank())

DEBUG_MODE = False
from SimulationBox import SimulationBox
import Functions as fun

class bondTransposition:
    count = 0
    success = 0
    simBox = None
    temp = 8000
    tempBKS = 8000
    lx = 0
    ly = 0
    lz = 0
    lmp = None  # lammps instance for getting PE at each bond transposition
    L_BKS = None  # lammps instance for getting PE using BKS at each bond transposition
    BKS_minimize_flag = 1  # determine whether to use BKS potential to check the structure
    bond_list_Si = None  # 4-O for each Si;
    bond_list_O = None   # 2-Si for each O
    O_list = None
    numO = 0
    bond_trans_list = None
    cutoff = np.array([12.0, 9.5, 0])  # outter and inner radius to define local sphere AND
    # last number is # of atoms in the inner sphere

    calculation_flag = np.zeros(comm.Get_size(), dtype=int)
    attribute = np.zeros(9)
    # attribute[0]: pressure keating (bar)
    # attribute[1]: previous global Keating PE
    # attribute[2]: current  global Keating PE
    # attribute[3]: previous local  Keating PE
    # attribute[4]: current  local  Keating PE
    # attribute[5]: previous global BKS PE
    # attribute[6]: current  global BKS PE
    # attribute[7]: current  global BKS Pressure (bar)
    # attribute[8]: current  system volumne

    # for new round of calculation
    def reset_calculation_flag(self):
        self.success += 1
        self.calculation_flag = np.zeros(comm.Get_size(), dtype=int)
        self.lmp.close()
        self.lmp = lammps(cmdargs=['-screen', 'lammps.initial.screen.' + str(comm.Get_rank())], comm=split)
        self.lmp.command("read_restart restart_original_position.tmp")
        self.attribute[0] = self.get_thermo(self.lmp, "press")

    # get success number #
    def get_success(self):
        return self.success

    # get success #
    def get_attribute(self, index):  # see self.attribute for index meaning
        return self.attribute[index]

    #
    def __init__(self, mysystem, frameNum, lmp_inputfile, bond_list_Si, bond_list_O, T, T_BKS,
                 restart=0, out_cutoff=12.0, in_cutoff=9.5, BKS_flag=1):
        self.simBox = mysystem[frameNum]
        self.lx, self.ly, self.lz = self.simBox.L
        self.BKS_minimize_flag = BKS_flag
        # restart == number of previous successfull bond transposition
        if restart:
            self.count = 0
            self.lmp = lammps(cmdargs=['-screen', 'lammps.initial.screen.' + str(comm.Get_rank())], comm=split)
            self.lmp.command("read_restart restart_original_position.tmp")
            self.success = restart  # restart: previous number of successful bond transposition
            self.cutoff = np.array([out_cutoff, in_cutoff, 0.0])
        else:
            self.lmp = self.initiate_lammps(lmp_inputfile)

        self.bond_list_Si, self.bond_list_O = bond_list_Si, bond_list_O
        self.O_list = np.array([i for i in self.bond_list_O])
        self.numO = self.O_list.size
        self.temp = T
        self.tempBKS = T_BKS

        # measure energy after shrink or enlarge the sample
        self.lmp.command('reset_timestep 0')
        if not restart:
            self.lmp.command('min_style cg')
            self.lmp.command('minimize 1.0e-15 1.0e-15 10000 100000')

        #  # enlarge simulation box (tune SiO2 density)
        #  enlarge_ratio = 1.1
        #  self.lmp.command('change_box' + ' all x scale ' + str(enlarge_ratio) +  " y scale " +
        #                   str(enlarge_ratio) + " z scale " + str(enlarge_ratio) + " remap")


        self.lmp.command('fix 1 all nve')
        self.lmp.command('run 0')
        self.lmp.command('unfix 1')
        # determine initial energy (from lammps)
        # print('[Attribute]:', self.attribute)
        if comm.Get_rank() == 0:
            self.attribute[1] = self.get_thermo(self.lmp, "pe")
            self.attribute[0] = self.get_thermo(self.lmp, "press")
            self.attribute[8] = self.get_thermo(self.lmp, "vol")
            if self.BKS_minimize_flag:
                self.attribute[5], self.attribute[7] = self.get_BKS_PE(self.lmp)  # return PE and pressure using BKS
            else:
                self.attribute[5], self.attribute[7] = 0, 0  # return PE and pressure using BKS

            print("O atoms # = ", self.numO)
        comm.Barrier()
        self.attribute = comm.allreduce(self.attribute)
        #print("BKS_energy: ", self.attribute[5])
        #print('[Attribute]:', self.attribute)
        #exit()
        if comm.Get_rank() == 0:
            self.lmp.command('reset_timestep 0')
            self.lmp.command("write_dump all custom current_SiO2_confi.data id type q x y z")  # used for BKS relaxation
        """ 
        # if not args.restart:
            for x in range(10):
                print("Current cycle >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ", x)
                L.minimize("1.0e-4 1.0e-4 100 1000")
                currentPE = L.eval("pe")
                if abs(preE - currentPE)/abs(preE) <= 0.0001:
                    break
                preE = currentPE
        L.fix("1 all nve")
        L.run("0")
        if MPI.COMM_WORLD.rank == 0:
            print(">>>>> Potential energy: ", L.eval("pe"))
            initial_PE = L.eval("pe")
            press = L.eval("press")
        L.unfix("1")
        """
        if comm.Get_rank() == 0:
            # record initial status and atomic configuration
            self.lmp.command("reset_timestep 0")
            self.lmp.command("write_restart restart_original_position.tmp")
            self.lmp.command("write_dump all custom current_SiO2_confi.data id type q x y z")  # used for BKS relaxation
            self.lmp.command("log log.lammps.L.cpu.0 append")
        comm.Barrier()
        #else:
        #    L.clear()
        #    L = PyLammps()
        #    L.log("log.lammps.L append")
        #    L.read_restart("restart_original_position.tmp")
        #    # if restart is 1, then restore the random number generator state to continue from previous stop
        #    with open('random_number_state_for_restart.obj', 'rb') as f:
        #        random.setstate(load(f))
        # L_step = int(L.eval("step"))
        #quit()

    def initiate_lammps(self, inputfile):
        assert inputfile
        lmp = lammps(cmdargs=['-screen', 'lammps.initial.screen.' + str(comm.Get_rank())], comm=split)
        lmp.file(inputfile)
        return lmp

    # 2020-02-24  YJ
    # for WWW algorithm implementation
    # findn all Si-O bonds, Si-O-Si and O-Si-O bonds using a bond cutoff of
    # cutoff Si-O is the distance for Si-O bond, typically 2.2 A


    def get_bond_list(self, mybox, Si_type, O_type, cutoff_Si_O = 2.2):

        myatom = mybox.myatom
        resSi = defaultdict(list)
        resO = defaultdict(list)

        lx, ly, lz = mybox.L[0], mybox.L[1], mybox.L[2]
        ncell = []
        # initialize array ncell[3]
        for iii in range(3):
            ncell.append(mybox.ncell[iii])

        neigh_O = set()  # O atom
        neigh_Si = set()  # Si atom
        ################## do analysis
        # loop among different cell
        for cellx in range(ncell[0]):
            for celly in range(ncell[1]):
                for cellz in range(ncell[2]):
                    nmember = mybox.mycell[cellx][celly][cellz].nmember
                    # //loop among all the atoms contained in one cell
                    for imember in range(nmember):
                        comparei = mybox.mycell[cellx][celly][cellz].member[imember]
                        itype = myatom[comparei].type # get type
                        if itype == Si_type:
                            neigh_O.clear()
                        if itype == O_type:
                            neigh_Si.clear()

                            # loop among all neighbour cells
                        for ii in range(27):
                            c = mybox.mycell[cellx][celly][cellz].nbr[ii]
                            ncellx = (int) (c / (ncell[1] * ncell[2]))
                            ncelly = (int) (fmod((int) (c / ncell[2]), ncell[1]))
                            ncellz = (int) (fmod(c, ncell[2]))
                            inmember = mybox.mycell[ncellx][ncelly][ncellz].nmember

                            # loop among all the neighbour's atoms
                            for j in range(inmember):
                                comparej = mybox.mycell[ncellx][ncelly][ncellz].member[j]
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
                            assert len(neigh_Si) == 2
                            resO[myatom[comparei].id] = []
                            for it in neigh_Si:
                                resO[myatom[comparei].id].append(it)
        return resSi, resO

    #Vink's method (PRB 67, 2003 245201)


    def generate_random_Si_O_list_for_www(self):

        try_times = 100
        while True and try_times > 0:
            assert self.O_list.size == self.numO
            # print(numO)
            O2 = self.O_list[random.randint(0, self.numO - 1)]
            # print(bond_list_O[O2])
            Si1 = self.bond_list_O[O2][0]
            Si2 = self.bond_list_O[O2][1]
            diffO1 = set(self.bond_list_Si[Si1]) - set([O2])
            diffO2 = set(self.bond_list_Si[Si2]) - set([O2])
            if len(diffO1) != 3:
                print(self.bond_list_Si[Si1], [O2])
                name_temp = "Debug.Si"+str(Si1)+".O"+str(O2) +".cpu."+str(comm.Get_rank())
                self.lmp.command("write_data " + name_temp)
                assert False, "bond_list error!"
            if len(diffO2) != 3:
                print(self.bond_list_Si[Si2], [O2])
                name_temp = "Debug.Si"+str(Si1)+".O"+str(O2) +".cpu."+str(comm.Get_rank())
                self.lmp.command("write_data " + name_temp)
                assert False, "bond_list error!"
            O1, O4, O5 = random.sample(diffO1, 3)
            O3, O6, O7 = random.sample(diffO2, 3)
            diffS1 = set(self.bond_list_O[O1]) - set([Si1])
            diffS2 = set(self.bond_list_O[O3]) - set([Si2])
            Si3 = random.sample(diffS1, 1)[0]
            Si4 = random.sample(diffS2, 1)[0]
            if len(set(self.bond_list_Si[Si1]) & set(self.bond_list_Si[Si4])) + \
                    len(set(self.bond_list_Si[Si2]) & set(self.bond_list_Si[Si3])) == 0:
                break
            try_times -= 1
        # print("try_times=",try_times)
        if try_times <= 0:
            #return np.array([None, None, None, None, None, None, None, None, None, None, None])
            self.bond_trans_list = np.array([None, None, None, None, None, None, None, None, None, None, None])
            exit(0)
        self.bond_trans_list = np.array([O2, Si1, Si2, O1, O4, O5, O3, O6, O7, Si3, Si4])


    # Vink's method (PRB 67, 2003 245201)
    #
    #            O5           O3 –– Si4      >                  O5    O3 –– Si4
    #             |           |              >>>                  \  /
    #       O4 ––Si1–– O2 –– Si2 –– O7       >>>>>           O4 –– Si1–– O2 –– Si2 –– O7
    #             |           |              >>>                               / \
    #      Si3 ––O1           O6             >                         Si3 –– O1  O6
    #

    #   improved algorithm with fast evaluation of energy change
    def bond_transposition_WWW_parallel(self):
        self.count += 1

        O2, Si1, Si2, O1, O4, O5, O3, O6, O7, Si3, Si4 = self.bond_trans_list

        # Use L2 to try the bond transposition
        # If it is a success then
        #    check if BKS relaxation is successful, if yes
        #        do the same bond transposition in L
        #        replaced L atom position with the relaxed atom position of L2
        #
        # self.attribute[1] = self.get_thermo(self.lmp, "pe")
        L2 = lammps(cmdargs=['-screen', 'lammps.l2.local.screen.' + str(comm.Get_rank())], comm=split)
        #L2 = PyLammps(ptr=lmp)
        #L2 = PyLammps()
        #L2.read_restart("restart_original_position.tmp")
        #L2.log("log.lammps.l2.local append")
        L2.command("read_restart restart_original_position.tmp")
        L2.command("log log.lammps.l2.local" + str(comm.Get_rank()))
        if DEBUG_MODE and comm.Get_rank() == 0:
            print("L2 system contain #atoms:", L2.get_natoms())
        L2_coord = L2.gather_atoms('x', 1, 3)
        if DEBUG_MODE:
            assert (Si1 - 1) * 3 + 2 < len(L2_coord)
            assert (Si2 - 1) * 3 + 2 < len(L2_coord)
        sphcx_s1, sphcy_s1, sphcz_s1 = L2_coord[(Si1 - 1) * 3], L2_coord[(Si1 - 1) * 3 + 1], L2_coord[(Si1 - 1) * 3 + 2]
        sphcx_s2, sphcy_s2, sphcz_s2 = L2_coord[(Si2 - 1) * 3], L2_coord[(Si2 - 1) * 3 + 1], L2_coord[(Si2 - 1) * 3 + 2]
        #if comm.Get_rank() == 0:
        #    print(sphcx_s1, sphcy_s1, sphcz_s1)
        self.define_sphere_group_across_boundary(L2, sphcx_s1, sphcy_s1, sphcz_s1, self.cutoff[0], 'sphere_s1', "in")
        self.define_sphere_group_across_boundary(L2, sphcx_s2, sphcy_s2, sphcz_s2, self.cutoff[0], 'sphere_s2', "in")
        L2.command("group to_delete subtract all sphere_s1 sphere_s2")
        L2.command("group sphere_s1 delete")
        L2.command("group sphere_s2 delete")
        #L2.write_data("before_delete.data")
        L2.command("delete_atoms group to_delete bond yes")
        L2.command("group to_delete delete")

        self.define_sphere_group_across_boundary(L2, sphcx_s1, sphcy_s1, sphcz_s1, self.cutoff[1], 'sphere_s1', "in")
        self.define_sphere_group_across_boundary(L2, sphcx_s2, sphcy_s2, sphcz_s2, self.cutoff[1], 'sphere_s2', "in")
        L2.command("group to_fix subtract all sphere_s1 sphere_s2")
        L2.command("group sphere_s1 delete")
        L2.command("group sphere_s2 delete")
        L2.command("variable to_fix_num equal count(to_fix)")
        # print(L2.groups)
        #num_fix = L2.variables['to_fix_num'].value
        num_fix = L2.extract_variable('to_fix_num', 'to_fix', 0)
        num_total = L2.get_natoms()
        self.cutoff[2] = num_total - num_fix
        if comm.Get_rank() == 0 and num_total - num_fix < 350:
            print("to_fix_atom_num:", num_fix, "local_atom#:", num_total - num_fix)
        if num_total - num_fix < 380:
            #print("After deletion, L system contain #atoms:", self.lmp.get_natoms())
            #print("After deletion, L2 system contain #atoms:", num_total)
            self.cutoff[0] += 0.1
            self.cutoff[1] += 0.1
            #L2.command("write_data DEBUG_NUM_L2_ATOM_ERROR-300.dat." + str(self.success) + str(comm.Get_rank()))
            #if num_total - num_fix < 200:
            #    #assert False, "WARNING for WWW algorithm, # of local atoms must be enough > 200"
            #    L2.command("write_data DEBUG_NUM_L2_ATOM_ERROR-200.dat." + str(self.success) + str(comm.Get_rank()))
        if num_total - num_fix > 410:
            self.cutoff[0] -= 0.1
            self.cutoff[1] -= 0.1


        self.attribute[3] = self.get_thermo(L2, "pe")
        if DEBUG_MODE and comm.Get_rank() == 0:
            print("After deletion, L2 system contain #atoms:", num_total)

        ###########  do the bond transposition
        self.bond_transposition_operation(L2)

        # print("START minimize energy after bond switch")
        # set the skin layer's force to be zero so that they are not moved during minimization
        L2.command("fix skin to_fix setforce 0.0 0.0 0.0")
        # self.lmp.command("log log.minimize append")
        L2.command("log log.minimize.local_l2." + str(comm.Get_rank()) + " append")
        self.attribute[4] = self.minimize_system(L2, 10)

        if comm.Get_rank() == 0 and DEBUG_MODE:
            print("[Local_minimization]  Initial_local_PE(without_transposition): %4.3f" % self.attribute[3],
                  "Final_local_PE(relaxed_E_after_transposition): %4.3f" % self.attribute[4])
            print(self.attribute)

        # print("END local minimize energy after bond switch")
        rand_num = random.random()
        temp_value = (self.attribute[3] - self.attribute[4])/8.617e-5/self.temp
        if temp_value > 100:
            temp_value = 100
        if temp_value < -100:
            temp_value = -100
        poss_num = exp(temp_value)

        if rand_num > poss_num:
            L2.close()
            self.calculation_flag[comm.Get_rank()] = 0
            return
        if DEBUG_MODE:
            print("[CPU]:", comm.Get_rank(), "[rand#] %1.3f" % rand_num, "[possibility] %1.3f" % poss_num, "[Temp(K)]",
                  self.temp, "[Temp(K)]_BKS", self.tempBKS)

        # do global relaxation using Keating potential'
        L_new = lammps(cmdargs=['-screen', 'lammps.l2.local.screen.' + str(comm.Get_rank())], comm=split)
        L_new.command("log log.lammps.Keating")
        L_new.command("read_restart restart_original_position.tmp")
        self.substitute_with_new_local_atom_position(L_new, L2) #change position
        self.bond_transposition_operation(L_new)  # change topology
        #L_new.command("write_data new_SiO2_confi.data.debug." + str(comm.Get_rank()) + ".0")  # used for BKS energy calculation
        L2.close()
        self.attribute[2] = self.attribute[1] + self.attribute[4] - self.attribute[3]
        self.attribute[2] = self.minimize_system(L_new, 10)
        L_new.command("reset_timestep 0")
        L_new.command("write_dump all custom new_SiO2_confi.data." + str(comm.Get_rank())+" id type q x y z")  # used for BKS energy calculation
        #L_new.command("write_data new_SiO2_confi.data.debug." + str(comm.Get_rank()) + ".1")  # used for BKS energy calculation
        L_new.close()


        # if failed, it means BKS does not favor the newer structure
        # BKS_minimize_flag is flag for whether suffer the system to a BKS energy check, 1: yes, 0: no
        if self.BKS_minimize_flag:
            L_BKS_tranpose = self.BKS_initialize_box("new_SiO2_confi.data." + str(comm.Get_rank()), 0)
            L_BKS_tranpose.command("log log.lammps.BKS.transpose." + str(comm.Get_rank()))
            L_BKS_tranpose.command("fix 1 all nve")
            myflag = True
            #L_BKS_tranpose.command("run 0")  # this step may fail when O-O distance is too close
            ee = " "
            try:
                L_BKS_tranpose.command("run 0")  # this step may fail when O-O distance is too close
            except Exception as e:
                myflag = False
                f = open("Lammps_error.print"+str(comm.Get_rank()), "a+")
                f.write("{} {}\n".format(str(self.count), str(e)))
                f.close()
                pass
            if not myflag:
                L_BKS_tranpose.close()
                return
            else:
                f = open("Lammps_error.print"+str(comm.Get_rank()), "a+")
                f.write("{} {}\n".format(str(self.count), str(ee)))
                f.close()
            BKS_after_E = L_BKS_tranpose.get_thermo("pe")
            L_BKS_tranpose.close()
            self.attribute[6] = BKS_after_E

            temp_value = (self.attribute[2]-self.attribute[1])/8.617e-5/self.temp - \
                         (self.attribute[6] - self.attribute[5])/8.617e-5/self.tempBKS
            #temp_value = -(self.attribute[6] - self.attribute[5])/8.617e-5/self.tempBKS

            # if comm.Get_rank() == 1:
            #     print("My_GOD==========")
            #     print(self.attribute)

            if temp_value > 100:
                temp_value = 100
            if temp_value < -100:
                temp_value = -100

            ##poss_num = min(1.0, exp(temp_value))
            #poss_num *= min(1.0, exp(temp_value))
            poss_num *= exp(temp_value)

            if rand_num > poss_num:
                return
            if DEBUG_MODE:
                print("[Successful(BKS) on CPU]", comm.Get_rank(),
                      "\n[BKS energy]  Initial_global_PE(without_transposition): %3.3f" % self.attribute[5],
                      "Final_global_PE(after_transposition): %3.3f" % self.attribute[6])
        else:
            self.calculation_flag[comm.Get_rank()] = 1

        # change the topology in L for a successful local transposition and do Bond Transposition
        # do bond list update in main function
        #self.bond_list_Si[Si1] = [O2, O3, O4, O5]
        #self.bond_list_Si[Si2] = [O2, O1, O6, O7]
        #self.bond_list_O[O1] = [Si2, Si3]
        #self.bond_list_O[O3] = [Si1, Si4]

        # Do bond transposition on the big system L
        self.bond_transposition_operation(self.lmp)
        self.lmp.command("read_dump new_SiO2_confi.data." + str(comm.Get_rank())+" 0 x y z replace yes box no")
        return

    # define a sphere region in or outside a sphere with "name"
    # use lammps interface (not PyLammps)
    def define_sphere_group_across_boundary(self, L, x, y, z, r, name, in_out):
        Lx, Ly, Lz = self.lx, self.ly, self.lz
        flagx, flagy, flagz = 1, 1, 1
        xlo, ylo, zlo = L.extract_global('boxxlo', 1), L.extract_global('boxylo', 1), L.extract_global('boxzlo', 1)
        xhi, yhi, zhi = L.extract_global('boxxhi', 1), L.extract_global('boxyhi', 1), L.extract_global('boxzhi', 1)
        assert 2 * r <= xhi - xlo
        assert 2 * r <= yhi - ylo
        assert 2 * r <= zhi - zlo
        if abs(xlo - x) > abs(xhi - x):
            flagx = -1
        if abs(ylo - y) > abs(yhi - y):
            flagy = -1
        if abs(zlo - z) > abs(zhi - z):
            flagz = -1
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    xi, yi, zi = x + i * flagx * Lx, y + j * flagy * Ly, z + k * flagz * Lz
                    L.command("region r_temp_"+str(i)+str(j)+str(k) + " sphere "
                              +str(xi)+" "+str(yi)+" "+str(zi)+" "+str(r) + " side in")
        #cmd = "region sphere_temp1 union 8 r_temp_000 r_temp_001 r_temp_010 r_temp_011 r_temp_100 r_temp_101 r_temp_110 r_temp_111 side"
        L.command("region sphere_temp1 union 8 r_temp_000 r_temp_001 r_temp_010 r_temp_011 "
                  "r_temp_100 r_temp_101 r_temp_110 r_temp_111 side " + str(in_out))
        L.command("group " + name + " region sphere_temp1")
        L.command("region sphere_temp1 delete")
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    L.command("region r_temp_"+str(i)+str(j)+str(k) + " delete")


    # substitute the atom's position in L1 whose ID also appear in L2
    def substitute_with_new_local_atom_position(self, L1, L2):
        #assert L2.get_natoms() > 0
        L2.command("reset_timestep 0")
        L2.command("write_dump all custom relaxed_local_atoms.L2.data." + str(comm.Get_rank()) + " id type x y z")
        L1.command("read_dump relaxed_local_atoms.L2.data." + str(comm.Get_rank()) + " 0 x y z replace yes box no")
        # L1.write_dump("all custom relaxed_local_atoms.L1.data.before id x y z")
        # L1.write_dump("all custom relaxed_local_atoms.L1.data.after id x y z")
        return


    # minimize the system a few steps

    def minimize_system(self, L, cycle):
        L.command("fix 1 all nve")
        L.command("run 0")
        preE = L.get_thermo("pe")
        L.command("unfix 1")
        L.command("min_style cg")
        for x in range(cycle):
            # print("Current cycle >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ", x)
            L.command("min_style cg")
            L.command("minimize 1.0e-4 1.0e-4 100 1000")
            currentPE = L.get_thermo("pe")
            #print("CurrentPE", currentPE)
            if preE and abs(preE - currentPE)/abs(preE) <= 0.001:
                return currentPE
            preE = currentPE
        return preE



    # this function does the actual bond_transposition_operation in Lammps object
    # L must be a PyLammps Object which can be first a lammps() object and then wrapped with PyLammps
    # L is a lammps() object, not PyLammps()
    def bond_transposition_operation(self, L):
        #L = PyLammps(ptr=lmp)
        O2, Si1, Si2, O1, O4, O5, O3, O6, O7, Si3, Si4 = self.bond_trans_list
        L.command("print 'A new bond transposition starts:-------------------------'")
        # change radial bond
        #print(Si1, O1, Si2, O3)
        L.command("group Si_O_bond_1 id " + str(Si1)+" "+str(O1)+" " + str(Si2) + " "+str(O3))
        L.command("delete_bonds Si_O_bond_1 bond 1 remove")
        L.command("group Si_O_bond_1 delete")
        L.command("create_bonds single/bond 1 "+str(Si1)+" "+str(O3))
        L.command("create_bonds single/bond 1 "+str(Si2)+" "+str(O1))

        # change radial bond Si-O-Si
        L.command("group Si_O_Si_bond_1 id " + " ".join([str(Si1), str(O1), str(Si3), str(Si2), str(O3), str(Si4)]))
        L.command("delete_bonds Si_O_Si_bond_1 angle 2 remove")
        L.command("group Si_O_Si_bond_1 delete")
        L.command("create_bonds single/angle 2 "+str(Si4)+" "+str(O3)+" "+str(Si1))
        L.command("create_bonds single/angle 2 "+str(Si3)+" "+str(O1)+" "+str(Si2))

        # change radial bond O-Si-O
        L.command("group O_Si_O_bond id " + " ".join([str(O1), str(Si1), str(O4), str(O3), str(Si2), str(O7)]))
        L.command("delete_bonds O_Si_O_bond angle 1 remove")
        L.command("group O_Si_O_bond delete")
        L.command("create_bonds single/angle 1 " + " ".join([str(O3), str(Si1), str(O4)]))
        L.command("create_bonds single/angle 1 " + " ".join([str(O1), str(Si2), str(O7)]))

        L.command("group O_Si_O_bond id " + " ".join([str(O1), str(Si1), str(O5), str(O3), str(Si2), str(O6)]))
        L.command("delete_bonds O_Si_O_bond angle 1 remove")
        L.command("group O_Si_O_bond delete")
        L.command("create_bonds single/angle 1 " + " ".join([str(O3), str(Si1), str(O5)]))
        L.command("create_bonds single/angle 1 " + " ".join([str(O1), str(Si2), str(O6)]))

        L.command("group O_Si_O_bond id " + " ".join([str(O1), str(Si1), str(O2), str(O3), str(Si2), str(O2)]))
        L.command("delete_bonds O_Si_O_bond angle 1 remove")
        L.command("group O_Si_O_bond delete")
        L.command("create_bonds single/angle 1 " + " ".join([str(O3), str(Si1), str(O2)]))
        L.command("create_bonds single/angle 1 " + " ".join([str(O1), str(Si2), str(O2)]))
        return

    # initialize a BKS simulation box
    # using Appendix A from (PRB 97, 054106 2018)
    # version of BKS potential in order to avoid collapse of atoms at high temperature using Vollmayr Kob's bks
    # this need the table potential file: BKS.wolf.table.cut.1.2   (smallest interatomic distance: 1.2)
    # this need the table potential file: BKS.wolf.table  cut=0.01   (smallest interatomic distance: 1.2)
    """
    def BKS_initialize_box(self, dump_file_name, step):
        L = lammps(cmdargs=['-screen', 'lammps.bks.screen.' + str(comm.Get_rank())], comm=split)
        #print("***********", rank)
        L.command('log log.initial.bks')
        L.command('units metal')
        L.command('boundary p p p')
        L.command('atom_style charge')
        L.command('region box block 0 1 0 1 0 1')
        L.command('create_box 2 box')
        # 'keep' value for 'add' so that ID of added atom does not change
        L.command('read_dump ' + dump_file_name + "  " + str(step) + " x y z add keep box yes")

        L.command('mass 1 15.9994')  # O atom
        L.command('mass 2 28.0855')  # Si atom
        L.command('set type 1 charge -1.2')
        L.command('set type 2 charge  2.4')
        L.command('pair_style table linear 9001')
        L.command('pair_coeff 1 1 BKS.wolf.table.cut.1.2 O-O')
        L.command('pair_coeff 1 2 BKS.wolf.table.cut.1.2 O-Si')
        L.command('pair_coeff 2 2 BKS.wolf.table.cut.1.2 Si-Si')
        L.command('neighbor 0.5 bin')
        L.command('neigh_modify every 1 delay 0 check yes')
        L.command('thermo_style custom step temp ke pe etotal vol press pxx pyy pzz pxz lx ly lz enthalpy')
        L.command('thermo_modify norm yes')
        #print(L.system.natoms)
        return L
    """
    # use Fenglin's BKS version from Vollmayr
    def BKS_initialize_box(self, dump_file_name, step):
        L = lammps(cmdargs=['-screen', 'lammps.bks.screen.' + str(comm.Get_rank())], comm=split)
        #print("***********", rank)
        L.command('log log.initial.bks')
        L.command('units lj')
        L.command('boundary p p p')
        L.command('atom_style charge')
        L.command('region box block 0 1 0 1 0 1')
        L.command('create_box 2 box')
        # 'keep' value for 'add' so that ID of added atom does not change
        L.command('read_dump ' + dump_file_name + "  " + str(step) + " x y z add keep box yes")
        L.command("variable  r equal 1388.7730*1.0")
        L.command('mass 1 1.0')  # O atom
        L.command('mass 2 1.755')  # Si atom
        L.command('set type 1 charge -4.55364')
        L.command('set type 2 charge  9.10728')
        L.command('pair_style  bks 5.5 10')
        L.command("pair_coeff 1 1 $r 0.3623188 175.0")
        L.command("pair_coeff 1 2 18003.7572 0.2052048 133.5318")
        L.command("pair_coeff 2 2 0 1 1 0.5")
        L.command('neighbor 0.3 bin')
        L.command('neigh_modify every 1 delay 0 check yes')
        L.command('thermo_style custom step temp ke pe etotal vol press pxx pyy pzz pxz lx ly lz enthalpy')
        L.command('thermo_modify norm no')
        L.command("kspace_style pppm 1.0e-4")

        #print(L.system.natoms)
        return L

    def get_BKS_PE(self, l):
        # get the initial PE of the configuration (relaxed using Keating potential) based on BKS potential
        l.command('reset_timestep 0')
        l.command('write_dump all custom current_SiO2_confi.get_BKS_PE.data.' + str(comm.Get_rank()) + ' id type q x y z')  # used for BKS energy calculation
        L_BKS = self.BKS_initialize_box("current_SiO2_confi.get_BKS_PE.data." + str(comm.Get_rank()), 0)
        L_BKS.command('fix 1 all nve')
        L_BKS.command('run 0')
        if DEBUG_MODE:
            print("RANK=", comm.Get_rank())
            print("PE=", L_BKS.get_thermo("pe"))
        pe = L_BKS.get_thermo("pe")
        pressure = L_BKS.get_thermo("press") * 160 * 10000  # bar
        L_BKS.close()
        return pe, pressure


    def relax_volume_with_BKS(self):
        # get the initial PE of the configuration (relaxed using Keating potential) based on BKS potential

        L_BKS = self.BKS_initialize_box("new_SiO2_confi.data." + str(comm.Get_rank()), 0)
        L_BKS.command('fix 1 all nve')
        L_BKS.command('run 0')
        try:
            start_P = L_BKS.get_thermo("press") * 160 * 10000  #  bar
        except:
            start_P = None
            pass
        L_BKS.command("unfix 1")
        best_ratio = 1
        #exit()
        #res = optimize.minimize(lambda volume_ratio: abs(self.BKS_pressure_square(abs(volume_ratio[0]), L_BKS)),
                                #1.0, method='TNC', bounds=((0.8, 1.2),), options={'stepmx':0.05, 'accuracy':0.01})
        if DEBUG_MODE:
            start_time = time.time()
        # optimize volume so that press = 0
        try:
            res = optimize.minimize(lambda volume_ratio: abs(self.BKS_pressure_square(abs(volume_ratio[0]), L_BKS)),
                                    best_ratio, method='TNC', bounds=((0.8, 1.2),), options={'stepmx':0.2, 'xtol': 0.001})
            best_ratio = abs(res.x[0])
        except:
            print("Wrong during pressure minimization of the BKS system volume")
        L_BKS.command('change_box' + ' all x scale ' + str(best_ratio) + " y scale " +
                        str(best_ratio) + " z scale " + str(best_ratio) + " remap")
        L_BKS.command('fix 10 all nve')
        L_BKS.command('run 0')
        self.attribute[7] = L_BKS.get_thermo("press") * 160 * 10000 # bar
        self.attribute[8] = L_BKS.get_thermo("vol")
        L_BKS.command('unfix 10')
        if DEBUG_MODE:
            print("RANK=", comm.Get_rank())
            print("start_P = ", start_P)
            print("final_P = ", self.attribute[7])
            print("Time used for Volume minimization:", time.time() - start_time)
        L_BKS.command('reset_timestep 0')
        L_BKS.command('write_dump all custom current_SiO2_confi.BKS.relaxed.data.' + str(comm.Get_rank()) + ' id type q x y z')  # used for BKS energy calculation
        L_BKS.close()
        self.lmp.command("read_dump current_SiO2_confi.BKS.relaxed.data." + str(comm.Get_rank())+ " 0 x y z replace yes box yes" ) # remember to substitute Box size
        self.lmp.command("fix 3 all nve")
        self.lmp.command("run 0")

        #update keating potential energy and its pressure
        self.attribute[2] = self.lmp.get_thermo("pe")
        self.attribute[0] = self.lmp.get_thermo("press")
        self.lmp.command("unfix 3")

    # for calculation of pressure squared in a BKS system so the pressure can be minimized using scipy optimize.minimize package
    def BKS_pressure_square(self, scale_ratio, lmp_BKS):
        #cmd = 'change_box' + ' all x scale ' + str(scale_ratio) + " y scale " + str(scale_ratio) + " z scale " + str(scale_ratio) + " remap"
        #print(cmd)
        lmp_BKS.command('change_box' + ' all x scale ' + str(scale_ratio) + " y scale " +
                        str(scale_ratio) + " z scale " + str(scale_ratio) + " remap")
        lmp_BKS.command('fix 10 all nve')
        lmp_BKS.command('run 0')
        lmp_BKS.command('unfix 10')
        press = lmp_BKS.get_thermo("press") * 160 * 10000  # Bar
        #print("press = ", press)
        lmp_BKS.command('change_box' + ' all x scale ' + str(1.0 / scale_ratio) +  " y scale " +
                        str(1.0 / scale_ratio) + " z scale " + str(1.0 / scale_ratio) + " remap")
        lmp_BKS.command('fix 10 all nve')
        lmp_BKS.command('run 0')
        lmp_BKS.command('unfix 10')
        return press ** 2

    #  get thermo parameters from lammps
    #  L is a lammps() object
    def get_thermo(self, L, some_property):
        # get the initial PE of the configuration (relaxed using Keating potential) based on BKS potential
        try:
            L.command('fix nve_run all nve')
            L.command('run 0')
            L.command('unfix nve_run')
            value = L.get_thermo(some_property)
        except:
            value = None
        assert value is not None, "get_thermo wrong"
        return value

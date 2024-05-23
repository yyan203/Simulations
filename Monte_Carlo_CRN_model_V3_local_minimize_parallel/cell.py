

MaxCellAtoms = 5000
MaxCellNbr = 100

# this is about the cell formed by dividing the box

class Cell:
    def __init__(self):
        self.member = [0] * MaxCellAtoms
        self.nmember = 0
        self.nbr = [0] * MaxCellNbr
        self.nnbr = 0
        self.origin = [0.0, 0.0, 0.0]
        self.size = [0.0, 0.0, 0.0]
    """
    def read_xyz(self, filename):
        with open(filename) as f:
            for index, line in enumerate(f):
                if index == 0:
                    self.atomnum = int(line)
                if index > 1:
                    t = line.split(' ')
                    # print(str(t[1]))
                    self.atoms[int(t[0])] = Atom(int(t[0]), str(t[1]), float(t[2]), float(t[3]), float(t[4]))
                    # print(self.atoms[int(t[0])].xyz_)
                    self.maxID = max(self.maxID, int(t[0]))
        # print(self.atoms[10].xyz_)
        # print(self.atoms[10].type_)
        print("XYZ file import:", self.atomnum, " atoms!")
    """

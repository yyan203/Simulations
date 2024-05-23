
class Atom:
    def __init__(self, id, type, x, y, z):
        assert isinstance(x, float)
        self.id = id
        self.type = type
        self.x = [x, y, z]


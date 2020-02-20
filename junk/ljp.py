class LJP:
    def __init__(self, e, s):
        self.e = e
        self.s = s

    def __call__(self, r):
        return 4 * self.e * ((self.s / r)**12 - (self.s / r)**6)

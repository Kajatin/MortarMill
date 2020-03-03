class Brick:
    """Represents a brick in the supplied image"""

    def __init__(self, id):
        self.id = id
        self.contour = None
        self.center = None

    def __repr__(self):
        return super().__repr__() + f'\tid: {self.id} center: {self.center}'


if __name__ == '__main__':
    brick = Brick(2)

    print(brick)
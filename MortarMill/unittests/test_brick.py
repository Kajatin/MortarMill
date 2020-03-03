import unittest
from brick import Brick

class TestBrick(unittest.TestCase):

    def setUp(self):
        self.id = 1
        self.brick = Brick(self.id)

    def test_init(self):
        self.assertEqual(self.brick.id, self.id)
        self.assertIsNone(self.brick.center)
        self.assertIsNone(self.brick.contour)


if __name__ == '__main__':
    unittest.main()

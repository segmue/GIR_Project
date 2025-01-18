from numpy import sqrt
import math

from numpy import sqrt
import math
class Point():
    # initialise
    def __init__(self, x=None, y=None):
        if len(x) > 1:
            raise ValueError(f"x should be a single value, is {x}")
        if len(y) > 1:
            raise ValueError(f"x should be a single value, is {y}")
        self.x = x
        self.y = y

    # representation
    def __repr__(self):
        return f'Point(x={self.x}, y={self.y})'

    # calculate Euclidean distance between two points
    def distEuclidean(self, other):
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    # calculate Manhattan distance between two points
    def distManhattan(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y)

    def distHaversine(self, other):
        my_x_rad, my_y_rad = self.deg2rad(self.x), self.deg2rad(self.y)
        other_x_rad, other_y_rad = self.deg2rad(other.x), self.deg2rad(other.y)
        r = 6371000  # Mittlerer Radius volumengleicher Kugel nach GRS80

        hav_phi = self.haversine_function(other_x_rad - my_x_rad) + math.cos(my_x_rad) * math.cos(other_x_rad) * self.haversine_function(
            other_y_rad - my_y_rad)
        d = 2 * r * math.asin(sqrt(hav_phi))
        return d

    def deg2rad(self, degree):
        return degree * (math.pi / 180)

    def haversine_function(self, radians):
        return (1 - math.cos(radians)) / 2

    # Test for equality between Points
    def __eq__(self, other):
        if not isinstance(other, Point):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.x == other.x and self.y == other.y

    # We need this method so that the class will behave sensibly in sets and dictionaries
    def __hash__(self):
        return hash((self.x, self.y))

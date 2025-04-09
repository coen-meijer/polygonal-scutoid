import numpy as np
import math
import random

from collections import namedtuple, OrderedDict  # just to be explicit
from enum import Enum
from dataclasses import dataclass, astuple

import gauss_solver

POLYGON_START = np.array([0.0, 100.0])
# POLYGON_START_ANGLE = 0.0
POLYGON_SCALE = 2.0

# equidistance
# find a point in plane that had equal distance to three different points
# iteratively ? no. 
#
@dataclass
class Point:
    x: float
    y: float
    z: float

    def __sub__(self, p2):
        return Point(self.x - p2.x, self.y - p2.y, self.z - p2.z)

    def __add__(self, p2):
        return Point(self.x + p2.x, self.y + p2.y, self.z + p2.z)

    def get_array(self):
        return np.array([self.x, self.y, self.z])


@dataclass
class Plane:
    x:float
    y:float
    z:float
    c:float


LetteredPolyhedron = namedtuple("LetteredPolyhedron", "points faces")


def distance_equation_vector(point):
    # s = x^2 + y^2 + z^2 - 2*x*p_x - 2*y*p_y - 2*z*p_z + p_x^2 + p_y^2 + p_z^2
    # we ignore the quadratic terms, they are the same across the
    # different distances. They cancel out when in a later stage we subsract
    # the distance to one point to the distance to another

    constant = point.x * point.x + point.y * point.y + point.z * point.z
    return np.array([point.x * - 2.0, point.y * -2.0, point.z * -2.0, constant])


def restraint_solver(*restraints):
    first_point = None
    matrix_lines = []
    for restraint in restraints:
        if isinstance(restraint, Point):
            if first_point is None:
                first_point = distance_equation_vector(restraint)
            else:
                distance = distance_equation_vector(restraint)
                line = distance - first_point
                line[-1] = - line[-1]  # the difference sould be zero
                matrix_lines.append(line)
        elif isinstance(restraint, Plane):
            matrix_lines.append(astuple(restraint))
    print("matrix lines:", matrix_lines)
    matrix = np.array(matrix_lines)
    if(gauss_solver.gauss_solve(matrix)):
        return Point(*matrix[:, -1])
    else:
        return None
    

def fold_flat_matrix(start_point, vec_1, vec_2):
    vec_1 = vec_1 - start_point
    vec_2 = vec_2 - start_point
    vec_1 /= math.sqrt(np.inner(vec_1, vec_1))
    vec_2 -= vec_1 * np.inner(vec_1, vec_2)
    vec_2 /= math.sqrt(np.inner(vec_2, vec_2))
    return np.array([vec_1, vec_2])


def fold_flat_polygon(*points):
    fold_matrix = fold_flat_matrix(points[0], points[1], points[-1])
    points = points - points[0]
    return list([np.matmul(fold_matrix, point)for point in points])


def gen_voronoi_face(*restraints):
    first = restraints[0]
    second = restraints[1]
    rest = restraints[2:]
    result = []
    for rest1, rest2 in zip(rest, rest[1:] + rest[:1]):
        result.append(restraint_solver(first, second, rest1, rest2))
    return result


def dist(point_1, point_2):
    x_diff = point_1.x - point_2.x
    y_diff = point_1.y - point_2.y
    z_diff = point_2.z - point_2.z
    return math.sqrt(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff)


def find_angle(point_2d):
    if point_2d[0] > 0:
        angle = math.atan(point_2d[1] / point_2d[0])
    elif point_2d[0] == 0:
        if point_2d[1] > 0:
            angle = math.pi / 2
        else:
            angle = 3 * math.pi / 2
    else:
        angle = math.atan(point_2d[1]/ point_2d[0]) + math.pi
    return angle


def find_rotation_matrix_with_goneometric_functions(point_2d, destination_2d):
    rotation = find_angle(point_2d) - find_angle(destination_2d)
    sine = math.sin(rotation)
    cosine = math.cos(rotation)
    return np.array([[cosine, sine], [-sine, cosine]])
    # find the angle of the line of from the origin to the first point


def find_rotation_matrix_with_gauss_solver(point_2d, destination_2d):
    matrix = np.array([[point_2d[0],  point_2d[1], destination_2d[0]],
                                   [point_2d[1], -point_2d[0], destination_2d[1]]])
    result = gauss_solver.gauss_solve(matrix)
    print(matrix)
    sine = matrix[1, 2]
    cosine = matrix[0, 2]
    return np.array([[cosine, sine], [-sine, cosine]])


def find_rotation_and_scale_matrix(point_2d, destination_2d):
    return find_rotation_matrix_with_gauss_solver(point_2d, destination_2d)


def attach_polygon(anchor_0, anchor_1, corners):
    flat_polygon = fold_flat_polygon(corners)
    rotation_matrix = find_rotation_and_scale_matrix(anchor_1 - anchor_0, corners[1] - corners[0])
    rotated_polygon = [np.matmul(rotation_matrix, corner) + anchor_0 for corner in corners]
    # scale?


def string_face(corners, adjacent_face, scutoid_points):
    result = OrderedDict()
    for corner_letter in corners:
        result[corner_letter] = scutoid_points[corner_letter]
    # fold flat
    print(f'corners 3d: {result}')
    fold_matrix = fold_flat_matrix(result[corners[0]], result[corners[1]], result[corners[-1]])
    for corner in corners :
        result[corner] = np.matmul(fold_matrix, result[corner])
    print(f'corners folded flat: {result}')
    anchor = result[corners[0]]
    print(anchor)
    # attach (to adjacent_face)
    for corner in corners:
        result[corner] -= anchor
    print(f'corners transposed to origin: {result}')
    common_corners = list(set(corners).intersection(adjacent_face))
    assert len(common_corners) == 2, f"Two faces {corners} and {adjacent_face} seem not to be adjacent."
    print(f"common corners: {common_corners}")
    # find the corner the face needs to rotate make to it fit
    rotation_matrix = find_rotation_and_scale_matrix(result[common_corners[1]] - result[common_corners[0]],
                                                    adjacent_face[common_corners[0]] - adjacent_face[common_corners[1]])
    # find the discance the face needs to move to make it fit
    print(f"rotation_matrix: {rotation_matrix}")
    translation = adjacent_face[common_corners[0]] - result[common_corners[0]]
    print(f"translation: {translation}")
    for corner in corners:
        result[corner] = np.matmul(rotation_matrix, result[corner]) + translation
    return result


def scutoid1_alt():
    center = Point(3, 0, 1)
    opposite = Point(-3, 0, -1)
    neighbour_1 = Point(0, 7, 1)
    neighbour_2 = Point(10, 3, 1)
    neighbour_3 = Point(10, -3 , -1)
    triangle_neighbour = Point(7, -10, 1)
    neighbour_4 = Point(0, -7,-1)

    upper_plane = Plane(0, 0, 1, 10)
    lower_plane = Plane(0, 0, 1, -10)

    points = {}
    points['a'] = restraint_solver(center, upper_plane, opposite, neighbour_1).get_array()
    points['b'] = restraint_solver(center, upper_plane, neighbour_1, neighbour_2).get_array()
    points['c'] = restraint_solver(center, upper_plane, neighbour_2, neighbour_3).get_array()
    points['d'] = restraint_solver(center, upper_plane, neighbour_3, triangle_neighbour).get_array()
    points['e'] = restraint_solver(center, upper_plane, triangle_neighbour, neighbour_4).get_array()
    points['f'] = restraint_solver(center, upper_plane, neighbour_4, opposite).get_array()
    points['g'] = restraint_solver(center, triangle_neighbour, neighbour_3, neighbour_4).get_array()
    points['h'] = restraint_solver(center, lower_plane, opposite, neighbour_1).get_array()
    points['i'] = restraint_solver(center, lower_plane, neighbour_1, neighbour_2).get_array()
    points['j'] = restraint_solver(center, lower_plane, neighbour_2, neighbour_3).get_array()
    points['k'] = restraint_solver(center, lower_plane, neighbour_3, neighbour_4).get_array()
    points['l'] = restraint_solver(center, lower_plane, neighbour_4, opposite).get_array()

    faces = ["lkgef", "hlfa", "ihab", "jibc", "kjcdg", "gde", "fedcba", "klhij"]

    net_init = OrderedDict()
    net_init['l'] = np.array([0.0, 0.0])
    net_init['f'] = np.array([10.0, 0.0])

    first_face = string_face("lkgef", net_init, points)
    print (first_face)


# stappen
# reken de vlakken uit met de hoekpunten
# teken het eerste vlak
#    reken de punten om naar papier coordinaten
#    onthoud de rand voor nieuwe papier coordinaten voor nieuw vlak  

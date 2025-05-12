import numpy as np
import math
import random

from collections import namedtuple, OrderedDict  # just to be explicit
from enum import Enum
from dataclasses import dataclass, astuple

import gauss_solver

POLYGON_START = np.array([0.0, 100.0])

SETTINGS = {
        'shift' : 0.9,
        'grid'  : 10,
        'part'  : 3.4,
        'rize'  : 5.5
    }
# equidistance
# find a point in plane that had equal distance to three different points
# iteratively ? no. 

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


class UnfoldStage(Enum):
    FLATTENED = 1
    ROTATED = 2
    NET = 3


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
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~ restriant solver ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
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
        print(f"restraint_solver - restraint: {restraint}, lines: {matrix_lines}")
    matrix = np.array(matrix_lines)
    if(gauss_solver.gauss_solve(matrix)):
        return Point(*matrix[:, -1])
    else:
        print("couldn't solve matrix.")
        print(matrix)
        for line in matrix_lines:
            print(line)
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


def plane_from_points(point_1, point_2, point_3):
    print(f"plane from points - points: {point_1}, {point_2}, {point_3}")
    plane_direction = np.cross(point_2 - point_1, point_3 - point_1)
    print(f"plane from points - pre norm - plane_direction: {plane_direction}")
    plane_direction /= np.linalg.norm(plane_direction)
    print(f"plane from points - post_norm - plane_direction: {plane_direction}")
    constant = np.inner(plane_direction, point_1)
    return Plane(*plane_direction, constant)  # np.append(plane_direction, constant)


def find_rotation_and_scale_matrix(point_2d, destination_2d):
    return find_rotation_matrix_with_gauss_solver(point_2d, destination_2d)


def clockwise_edge(face, letters):
    face_string = ''.join(face)
    face_plus = face_string + face_string[0]
    sequence = ''.join(letters)
    if sequence in face_plus:
        return letters
    else:
        return letters[::-1]


def face_flat_adjacent(corners, adjacent_face, scutoid_points, stage=UnfoldStage.NET):
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ face_adjacent:", corners, "adjacent", ''.join(adjacent_face.keys()),"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    result = OrderedDict()
    for corner_letter in corners:
        result[corner_letter] = scutoid_points[corner_letter]
    fold_matrix = fold_flat_matrix(result[corners[0]], result[corners[1]], result[corners[-1]])
    for corner in corners :
        result[corner] = np.matmul(fold_matrix, result[corner])
    print(f'corners folded flat: {result}')
    if stage == UnfoldStage.FLATTENED:
        return result    # for debugging
    common_corners = list(set(corners).intersection(adjacent_face))
    assert len(common_corners) == 2, f"Two faces {corners} and {adjacent_face} seem not to be adjacent."
    print(f"common corners: {common_corners}")
    # find the corner the face needs to rotate make to it fit
    result_order = clockwise_edge(result, common_corners)
    adjacent_order = clockwise_edge(adjacent_face, common_corners)
    rotation_matrix = find_rotation_and_scale_matrix(
        result[result_order[1]] - result[result_order[0]],
        adjacent_face[adjacent_order[0]] - adjacent_face[adjacent_order[1]])
    # find the discance the face needs to move to make it fit
    print(f"rotation_matrix: {rotation_matrix}")
    for corner in corners:
        result[corner] = np.matmul(rotation_matrix, result[corner])
    print(f"rotated faces: {result}")
    if stage == UnfoldStage.ROTATED:
        return result    # for debugging
    translation = adjacent_face[common_corners[0]] - result[common_corners[0]]
    print(f" point:{common_corners[0]}, move from {result[common_corners[0]]} to{adjacent_face[common_corners[0]]}")
    print(f"translation: {translation}")
    for corner in corners:
        result[corner] = result[corner] + translation
    print(f"net: {result}")
    return result


def scale(points, factor):
    for point in points.keys():
        points[point] *=2


def mirror(points, faces):
    mirror_points = {}
    for letter in points.keys():
        mirror_point = points[letter]
        mirror_point[0] = - mirror_point[0]
        mirror_points[letter] = mirror_point
    mirror_faces = []  # keep clockwise ordering
    for face in faces:
        mirror_faces.append(face[::-1])
    return mirror_points, mirror_faces


def z_flip(points, faces):
    mirror_points = {}
    for letter in points.keys():
        mirror_point = points[letter]
        mirror_point[2] = - mirror_point[2]
        mirror_points[letter] = mirror_point
    mirror_faces = []  # keep clockwise ordering
    for face in faces:
        mirror_faces.append(face[::-1])
    return mirror_points, mirror_faces


def scutoid1(shift=0.9, grid=10, part=3.4, rize=5.5):
    center = Point(part, 0, shift)
    opposite = Point(-part, 0, -shift)
    neighbour_1 = Point(0, grid - part, shift)
    neighbour_2 = Point(grid, part, shift)
    neighbour_3 = Point(grid, -part , -shift)
    triangle_neighbour = Point(grid - part, -grid, shift)
    neighbour_4 = Point(0, part - grid, -shift)

    upper_plane = Plane(0, 0, 1, rize)
    lower_plane = Plane(0, 0, 1, -rize)

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

    faces = ["lkgef", "hlfa", "ihab", "ibcj", "kjcdg", "gde", "fedcba", "klhij"]

    return points, faces


def puzzle_piece(shift=0.9, grid=10, part=3.4, rize=5.5):
    # similar to the scutoid function but with less faces and a cut off plane
    # the cut off plane is from a differen scutoid, so we have to
    # calculate some of its points as well

    center = Point(part, 0, shift)
    opposite = Point(-part, 0, -shift)
    far = Point(-grid, -part, -shift)
    upper_plane = Plane(0, 0, 1, rize)
    lower_plane = Plane(0, 0, 1, -rize)
    neighbour_4 = Point(0, part - grid, -shift)

    # Calculate the plane that cuts of the part
    low_far_plane_point = restraint_solver(lower_plane, far, opposite, neighbour_4).get_array()
    low_close_plane_point = restraint_solver(lower_plane, center, opposite, neighbour_4).get_array()
    high_close_plane_point = restraint_solver(upper_plane, center, opposite, neighbour_4).get_array()

    border_plane = plane_from_points(low_far_plane_point, low_close_plane_point, high_close_plane_point)
    print(f"border_plane: {border_plane}")

    neighbour_1 = Point(0, grid - part, shift)
    neighbour_2 = Point(grid, part, shift)

    triangle_neighbour = Point(grid - part, -grid, shift)

    points = {}
    points['a'] = restraint_solver(center, upper_plane, opposite, neighbour_1).get_array()
    points['b'] = restraint_solver(center, upper_plane, neighbour_1, neighbour_2).get_array()
    points['c'] = restraint_solver(center, upper_plane, neighbour_2, border_plane).get_array()
    points['d'] = restraint_solver(center, upper_plane, border_plane, opposite).get_array()

    points['e'] = restraint_solver(center, lower_plane, opposite, neighbour_1).get_array()
    points['f'] = restraint_solver(center, lower_plane, neighbour_1, neighbour_2).get_array()
    points['g'] = restraint_solver(center, lower_plane, neighbour_2, border_plane).get_array()
    points['h'] = restraint_solver(center, lower_plane, border_plane, opposite).get_array()

    faces = ["adhe", "baef", "cbfg", "dcgh", "abcd", "hgfe"]

    return points, faces


def calculate_corner(shift=0.9, grid=10, part=3.4, rize=5.5):
    # a lot of code borrowed from puzzle_piece

    center = Point(part, 0, shift)
    opposite = Point(-part, 0, -shift)
    far = Point(-grid, -part, -shift)
    upper_plane = Plane(0, 0, 1, rize)
    lower_plane = Plane(0, 0, 1, -rize)
    neighbour_4 = Point(0, part - grid, -shift)

    # Calculate the plane that cuts of the part
    low_far_plane_point = restraint_solver(lower_plane, far, opposite, neighbour_4).get_array()
    low_close_plane_point = restraint_solver(lower_plane, center, opposite, neighbour_4).get_array()
    high_close_plane_point = restraint_solver(upper_plane, center, opposite, neighbour_4).get_array()

    border_plane = plane_from_points(low_far_plane_point, low_close_plane_point, high_close_plane_point)
    print(f"border_plane: {border_plane}")

    neighbour_2 = Point(grid, part, shift)

    corner = restraint_solver(center, upper_plane, neighbour_2, border_plane).get_array()

    return corner[0], corner[1]


def puzzle_border(shift=0.9, grid=10, part=3.4, rize=5.5, border_width = 3.0, margin = 1.2):
    corner_x, corner_y = calculate_corner(shift=shift, grid=grid, part=part, rize=rize)
    print(f"corner_x: {corner_x}, corner_y: {corner_y}")
    other_corner_x = -grid + corner_y
    other_corner_y = -corner_x
    side = math.sqrt((corner_x - other_corner_x) ** 2 + (corner_y - other_corner_y) ** 2) + margin
    print(f"side: {side}")
    points = {}
    points['a'] = np.array([0., 0., 0.])
    points['b'] = np.array([0., side, 0.])
    points['c'] = np.array([side, side, 0.])
    points['d'] = np.array([side, 0., 0.])

    points['e'] = np.array([-border_width, -border_width, 0.0])
    points['f'] = np.array([-border_width, side + border_width, 0.0])
    points['g'] = np.array([side + border_width, side + border_width, 0.0])
    points['h'] = np.array([side + border_width, -border_width, 0.0])

    for low, high in zip('abcdefgh', 'ijklmnop'):
        points[high] = np.array([*points[low][:-1], 2 * rize])

    faces = ['feab', 'bfgc', 'cghd', 'daeh',
             'aijb', 'bjkc', 'ckld', 'dlia',
             'imnj', 'jnok', 'kopl', 'lpmi',
             'nmef', 'onfg', 'pogh', 'mphe'
             ]

    return points, faces


def find_adjacent_face(face, potential_neighbours):
    face_set = set(face)
    for neighbour in potential_neighbours:
        if len(face_set.intersection(neighbour)) == 2:
            return neighbour
    return None

import numpy as np
import math
import random

from collections import namedtuple

Point = namedtuple("Point", ['x', 'y', 'z'])
Plane = namedtuple("Plane", ['x', 'y', 'z', 'c'])


# equidistance
# find a point in plane that had equal distance to three different points
# iteratively ? no. 
#

def pivot(matrix, column):
    index_largest = column
    maximum = abs(matrix[column, column])
    for index in range(column + 1, len(matrix)):
        if abs(matrix[index][column]) > maximum:
            maximum = abs(matrix[index][column])
            index_largest = index
    if column != index_largest:
        matrix[[column, index_largest]] = matrix [[index_largest, column]]  #swap

def gauss_solve_lower(matrix):
    for diagonal_index in range(matrix.shape[0]):
        pivot(matrix, diagonal_index)
        diagonal_val = matrix[diagonal_index, diagonal_index]
        matrix[diagonal_index] /= diagonal_val
        for row_index in range(diagonal_index + 1, matrix.shape[0]):
            elimination_val = matrix[row_index, diagonal_index]
            matrix[row_index] -= elimination_val * matrix[diagonal_index]


def gauss_solve_upper(matrix):
    for diagonal_index in range(matrix.shape[0] - 1, -1, -1):
        for column_index in range(matrix.shape[0] - 1, diagonal_index, -1):
            elimination_val = matrix[diagonal_index, column_index]
            matrix[diagonal_index] -= elimination_val * matrix[column_index]
        

def gauss_solve(matrix):
    gauss_solve_lower(matrix)
    gauss_solve_upper(matrix)
#    if (gauss_solve_lower(matrix)):
#        return gauss_solve_upper(matrix)
#    else:
#        return False
    # return solve_lower(matrix) and solve_upper(matrix)?


def distance_equation_vector(point):
    # s = x^2 + y^2 + z^2 - 2*x*p_x - 2*y*p_y - 2*z*p_z + p_x^2 + p_y^2 + p_z^2
    # we ignore the quadratic terms, they are the same across the
    # different distances. They cancel out when in a later stage we subsract
    # the distance to one point to the distance to another

    linear_coefficient_x = -2.0 * point[0]
    linear_coefficient_y = -2.0 * point[1]
    linear_coefficient_z = -2.0 * point[2]
    constant             = point[0] * point[0] + point[1] * point[1] + point[2] * point[2]
    return np.array([linear_coefficient_x, linear_coefficient_y, linear_coefficient_z, constant])


def equal_distance(*points, z_plane = None):
    if (len(points) != 4 and z_plane is None) or (len(points) != 3 and z_plane is not None):
        print("z_plane is", z_plane)
        raise ValueError("The number of points should be 4, or 3, the latter only when z_plane is given.")
    equations = [distance_equation_vector(p) for p in points]
    last_index = len(points) - 1
    print("distance equations", equations)
    for i in range(last_index):
        equations[i] -= equations[last_index]
        equations[i][-1] = -equations[i][-1]  # the difference should equal zero
    equations = equations[:-1]
    if z_plane is not None:
        equations.append(np.array([0.0, 0.0, 1.0, z_plane]))  # volgende keer testen        
    matrix = np.array(equations)
#    print(matrix)
    solution = gauss_solve(matrix)
#    print('________')
#    print(matrix)
    return matrix[:, -1]


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
            matrix_lines.append(restraint)
    print("matrix lines:", matrix_lines)
    matrix = np.array(matrix_lines)
    gauss_solve(matrix)
    return Point(*matrix[:, -1])        
                

def test_gauss_solver():
    matrix = np.array([2.0, 3.0, 6.0, 4.0, 9.0, 15.0]).reshape(2,3)
    gauss_solve(matrix)
    print(matrix)
    print("________")
    matrix = np.array([
        [ 3.0,  2.0, -1.0,  1.0],
        [ 2.0, -2.0,  4.0, -2.0],
        [-1.0,  0.5, -1.0,  0.0],
        ])
    print(matrix)
    print("________")
    gauss_solve(matrix)
    print(matrix)
    print("________")


def gen_test_equal_distance():
    z_plane = random.uniform(-10, 10)
    center = [random.uniform(-10, 10), random.uniform(-10, 10), z_plane]
    print('center:', center)
    radius = random.uniform(1, 10)
    print("radius:", radius)
    points = []
    for i in range(3):
        y = random.uniform(-radius, radius)
        intersection_radius = math.sqrt(radius * radius - y * y)
        y += center[1]
        angle = random.uniform(0, 2*math.pi)
        x = center[0] + math.sin(angle) * intersection_radius
        z = center[2] + math.cos(angle) * intersection_radius
        print("point", x, y, z)
#        print("dist:", math.sqrt( (x - center[0]) ** 2 +
#                                  (y - center[1]) ** 2 +
#                                  (z - center[2]) ** 2))
        points.append([x, y, z])
    result_center = equal_distance(*points, z_plane=z_plane)
    difference = center - result_center
    error = math.sqrt(difference.dot(difference))
    print(error)


def test_restriant_solver():
    restraints = [Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 0.0),
                  Point(0.0, 1.0, 0.0), Plane(0.0, 0.0, 1.0, 1.0)]
    print(restraint_solver(*restraints))
    

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


def external_face(seed, neighbour_seeds, z_plane):
    first_point = equal_distance(
        seed, neighbour_seeds[0], neighbour_seeds[-1], z_plane=z_plane)
    result = [first_point]
    for neighbour, next_neighbour in zip(neighbours, beighbours[1:]):
        result.append(equal_distance(
            seed, neighbour, next_neighbour, z_plane=z_plane))
    return result


def four_side_side_face(seed, neighbours, top_plane, bottom_plane):
    result = []
    result.append(equal_distance(
        seed, neighbours[0], neighbours[1], z_plane=bottom_plane))
    result.append(equal_distance(
        seed, neighbours[0], neighbours[1], z_plane=top_plane))
    result.append(equal_distance(
        seed, neighbours[1], neighbours[2], z_plane=top_plane))
    result.append(equal_distance(
        seed, neighbours[1], neighbours[2], z_plane=bottom_plane))
    return result


def triangle_side_face_top(seed, neighbours, plane):
    result = []
    result.append(equal_distance(
        seed, neighbours[0], neighbours[1], neighbours[2]))
    result.append(equal_distance(
        seed, neighbours[0], neighbours[1], z_plane=plane))
    result.append(equal_distance(
        seed, neighbours[1], neighbours[2], z_plane=plane))
    # permutatie voor andere orrientatie?
    return result


def gen_voronoi_face(*restraints):
    first = restaints[0]
    second = restaints[1]
    rest = restraints[2:]
    restult = []
    for rest1, rest2 in zip(rest, rest[1:] + rest[:1])
        restult.append(restraint_solver(first, second, rest1, rest2))
    return result
    

def five_sided_side_face_left_bottom(seed, left_of_face,
                                     oppisite_face,
                                     right_of_face_top,
                                     right_of_face_bottom,
                                     top_plane, bottom_plane):
    # permutatie voor andere orientatie?
    result = []
    result.append(equal_distance(
        seed, oppisite_face, left_of_face, z_plane=bottom_plane))
    result.append(equal_distance(
        seed, oppisite_face, left_of_face, z_plane=top_plane))
    result.append(equal_distance(
        seed, oppisite_face, right_of_face_top, z_plane=top_plane))
    result.append(equal_distance(
        seed, opposite_face, right_of_face_top, right_of_face_bottom))
    retuls.append(equal_distance(
        seed, opposite_face, right_of_face_bottom, z_plane=bottom_plane))
    # hier verder. ... generale versie met named tuples? Point en Plane
    

def test_equal_distance():
    equal_distance((-0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0))
    print("=========================")
    equal_distance((10.0, 10.0, 0.0), (10.0, 20.0, 0.0), (20.0, 20.0, 10.0), z_plane = 100.0)
    print('+++++++++++++++++++++++++')
    gen_test_equal_distance()


def main():
    test_gauss_solver()
    print("test_equal_distance")
    test_equal_distance()
    vec_1 = np.array([1.0, 0.0, 0.0])
    vec_2 = np.array([1.0, 1.0, 2.0])
    start_point = np.array([0.0, 0.0, 0.0])
    print(fold_flat_matrix(start_point, vec_1, vec_2))
    point_list = fold_flat_polygon(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 4.0, 3.0]),
        np.array([0.0, 0.0, 6.0])
        )
    print(point_list)
    print("********************************************************************************")
    test_restriant_solver()


if __name__ == "__main__":
    main()                      


# stappen
# reken de vlakken uit met de hoekpunten
# teken het eerste vlak
#    reken de punten om naar papier coordinaten
#    onthoud de rand voor nieuwe papier coordinaten voor nieuw vlak  






import numpy as np
import scutoid
from scutoid import Point, Plane
import gauss_solver

# test_scutoid


def test_gauss_solver():
    matrix = np.array([2.0, 3.0, 6.0, 4.0, 9.0, 15.0]).reshape(2,3)
    gauss_solver.gauss_solve(matrix)
    print(matrix)
    print("________")
    matrix = np.array([
        [ 3.0,  2.0, -1.0,  1.0],
        [ 2.0, -2.0,  4.0, -2.0],
        [-1.0,  0.5, -1.0,  0.0],
        ])
    print(matrix)
    print("________")
    gauss_solver.gauss_solve(matrix)
    print(matrix)
    print("________")


def test_restriant_solver():
    restraints = [Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 0.0),
                  Point(0.0, 1.0, 0.0), Plane(0.0, 0.0, 1.0, 1.0)]
    print(scutoid.restraint_solver(*restraints))


def test_find_rotation_matrix():
    p_2d = np.array([0.0, 5.0])
    dest_2d = np.array([3.0, 4.0])
    print("find_rotation_matrix_with_goneometric_functions:", scutoid.find_rotation_matrix_with_goneometric_functions(p_2d, dest_2d))
    print("find_rotation_matrix_with_gauss_solver:", scutoid.find_rotation_matrix_with_gauss_solver(p_2d, dest_2d))


def test_plane_from_points():
    # make a test vector
    test_sets = [
        [np.array([1., 0., 0.]), np.array([1., 1., 1.]), np.array([1., 0., 1.])],
        [np.array([3., 4., 0.]), np.array([3., 4., 5.]), np.array([7., 1., 0.])],
        [np.array([1., 2., 3.]), np.array([3., 4., 0.]), np.array([3., 4., 12.])],
        [np.array([3., 4., 7.]), np.array([7., 1., 7.]), np.array([-4., 3., 14])]
        ]
    for test_vectors in test_sets:
        plane = scutoid.plane_from_points(*test_vectors)
        print(plane)
        for point in test_vectors:
            assert abs(plane.c - np.inner(point, np.array([plane.x, plane.y, plane.z]))) < 0.000_001
            print(f"{point[0]} * {plane.x} + {point[1]} * {plane.y} + {point[2]} * {plane.z} = {plane.c}")


def main():
    test_gauss_solver()
    print("test_equal_distance")
    vec_1 = np.array([1.0, 0.0, 0.0])
    vec_2 = np.array([1.0, 1.0, 2.0])
    start_point = np.array([0.0, 0.0, 0.0])
    print(scutoid.fold_flat_matrix(start_point, vec_1, vec_2))
    point_list = scutoid.fold_flat_polygon(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 4.0, 3.0]),
        np.array([0.0, 0.0, 6.0])
        )
    print("fold flat:", point_list)
    print("********************************************************************************")
    test_restriant_solver()
#    print('**** test gen_voronoi_face() *****')
#    print(scutoid.gen_voronoi_face(Point(0.0, 0.0, 0.0), Plane(0., 0., 1., 1.,),
#                           Point(2., 0., 0.), Point(-1., 1., 0.), Point(-1., -1., 0.))
#         )
    print("###################################################################################")
    test_find_rotation_matrix()
    print(r"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(scutoid.scutoid1())
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    test_plane_from_points()
    print("_______________________________________________________________________________")
    points, faces = scutoid.puzzle_piece()
    print(f"point c:{points['c']}")

if __name__ == "__main__":
    main()

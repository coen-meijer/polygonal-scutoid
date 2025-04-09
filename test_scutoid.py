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
    print('**** test gen_voronoi_face() *****')
    print(scutoid.gen_voronoi_face(Point(0.0, 0.0, 0.), Plane(0., 0., 1., 1.,),
                           Point(2., 0., 0.), Point(-1., 1., 0.), Point(-1., -1., 0.))
         )
    print("###################################################################################")
    test_find_rotation_matrix()
    print(r"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    scutoid.scutoid1_alt()

if __name__ == "__main__":
    main()

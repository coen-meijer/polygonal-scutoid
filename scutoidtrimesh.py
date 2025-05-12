import os

import scutoid
# from scutoid import SETTINGS
import trimesh
import pyglet
import numpy as np

PATH = "3dmodels"

NAMES = {
    scutoid.scutoid1: 'scutoid',
    scutoid.puzzle_piece: "puzzle_piece",
    scutoid.puzzle_border: "puzzle_border"
    }

SETTINGS = {
        'shift' : 0.9,
        'grid'  : 10,
        'part'  : 3.4,
        'rize'  : 7
    }

def gen_mesh(points, faces, filename):
#     points, faces = scutoid.mirror(*scutoid.scutoid1())
    mesh_points = []
    mesh_triangles = []
    letters = list(points.keys())
    mesh_points = list(points.values())

    for face in faces:
        print(face)
        first = face[0]
        for second, third in zip(face[1:], face [2:]):
            mesh_triangles.append([letters.index(first),
                                   letters.index(second),
                                   letters.index(third)])

    mesh = trimesh.Trimesh(vertices=mesh_points, faces=mesh_triangles)

    mesh.export(filename)

def main():
#    points, faces = scutoid.z_flip(*scutoid.scutoid1(**SETTINGS))
#    points, faces = scutoid.z_flip(*scutoid.puzzle_piece(**SETTINGS))
    points, faces = scutoid.puzzle_border(**SETTINGS)
#    mirror_points, mirror_faces = mirror_puzzle_peice = scutoid.mirror(points, faces)
    scutoid.scale(points, 2.0)
    gen_mesh(points, faces, os.path.join(PATH, f'scutoid-{SETTINGS["shift"]}-{SETTINGS["grid"]}-{SETTINGS["part"]}-{SETTINGS["rize"]}.stl'))


if __name__ == "__main__":
    main()

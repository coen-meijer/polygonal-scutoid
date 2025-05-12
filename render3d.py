# naar voorbeeld van Gerlinde.

from math import cos, pi, sin

from random import random

# import numpy as np
import panda3d
from direct.showbase.ShowBase import ShowBase
from panda3d.core import LineSegs, NodePath

# imports cube_example
from direct.task import Task
from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    CullFaceAttrib,
)

format = GeomVertexFormat.getV3c4()
format = GeomVertexFormat.registerFormat(format)

import scutoid as scut  # misschien gewoon importen als scutoid
# from scutoid import LetteredPolyhedron
from scutoid import UnfoldStage

DISPLAY_STAGES = [UnfoldStage.FLATTENED, UnfoldStage.ROTATED, UnfoldStage.NET]
# DISPLAY_STAGES = [UnfoldStage.ROTATED, UnfoldStage.NET]


DRAW_LINES = False
DRAW_SCUTOID = True
DRAW_FACES_FLAT = True

def lines_example():
    lines = LineSegs()
    lines.moveTo(1, 1, 1)
    lines.drawTo(2, 2, 2)
    lines.setThickness(4)
    return lines

def caltulate_normal(vertex1, vertex2, vertex3):
    diff1 = vertex2 - vertex1
    diff2 = vertex3 - vertex1
    return diff1.cross(diff2).normalize()

def draw_lettered_poyhedron(points, faces, colors):
    # Set up the data writers.
    vertex_format = GeomVertexFormat.get_v3c4()  # ????
    vertex_data = GeomVertexData("name", vertex_format, Geom.UHStatic)
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")
    triangles = GeomTriangles(Geom.UHStatic)
    vertex_name_list = []  # a list of string represntations of the vertexes added to the list

    def draw_face(face, face_color):
        # in order to give the whole face a solid color
        # we add the vertex seperately for each face
        # add the first vertex to the face
        for point in face:
            vertex_writer.add_data3f(*points[point])
            color_writer.add_data4f(*face_color)
            vertex_name_list.append(point + "@" + face)

        first = face[0]
        for second, third in zip(face[1:], face[2:]):
            triangles.addVertices(vertex_name_list.index(first + "@" + face),
                                  vertex_name_list.index(second + "@" + face),
                                  vertex_name_list.index(third + "@" + face))

    # code for the block draw_lettered_poyhedron
    for face, color in zip(faces, colors):
        draw_face(face, color)

    geom = Geom(vertex_data)
    geom.addPrimitive(triangles)
    node = GeomNode("scutoid")
    node.addGeom(geom)

    return node


def form_2d_to_3d(vec, height):
    print(vec)
    return [vec[0], vec[1], height]


def render_polygon_flat_face(scutoid_points, faces, colors, height, start, stage=UnfoldStage.NET, height_increment=0):
    vertex_format = GeomVertexFormat.get_v3c4()  # ????
    vertex_data = GeomVertexData("name", vertex_format, Geom.UHStatic)
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")
    triangles = GeomTriangles(Geom.UHStatic)
    vertex_name_list = []  # a list of string represntations of the vertexes added to the list

    def draw_face_flat(face_points, height, face_color):
        # in order to give the whole face a solid color
        # we add the vertex seperately for each face
        # add the first vertex to the face
        face = ''.join(face_points.keys())
        for point in face_points.keys():
            point_3d = form_2d_to_3d(face_points[point], height)  # face point?
            vertex_writer.add_data3f(*point_3d)
            color_writer.add_data4f(*face_color)
            vertex_name_list.append(point + "@" + face)

        first = face[0]
        for second, third in zip(face[1:], face[2:]):
            triangles.addVertices(
                vertex_name_list.index(first + "@" + face),
                vertex_name_list.index(second + "@" + face),
                vertex_name_list.index(third + "@" + face))

    added_faces = {}
    for face, color in zip(faces, colors):
        # find a face that is find_adjacent_face
        adjacent_face_name = scut.find_adjacent_face(face, added_faces.keys())
        if adjacent_face_name is None:  # initialize
            adjacent = start
        else:
            adjacent = added_faces[adjacent_face_name]
        points = scut.face_flat_adjacent(face, adjacent, scutoid_points, stage)
        added_faces[face] = points
        draw_face_flat(points, height, color)
        height += height_increment

    geom = Geom(vertex_data)
    geom.addPrimitive(triangles)
    node = GeomNode("scutoid")
    node.addGeom(geom)

    return node


def random_color(alpha=1.0):
    return random(), random(), random(), alpha


class main(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        lines = lines_example()
        if DRAW_LINES:
            lines_node = lines.create()
            node_path = NodePath(lines_node)
            node_path.reparentTo(self.render)
        if DRAW_SCUTOID:
            # get the scutoid
            scutoid_points, scutoid_faces = scut.scutoid1()
            face_colors = []
            for face in scutoid_faces:
                face_colors.append(random_color())
            scutoid_node = draw_lettered_poyhedron(scutoid_points, scutoid_faces, face_colors)
            scutoid_node_path = NodePath(scutoid_node)
            scutoid_node_path.reparentTo(self.render)
            if DRAW_FACES_FLAT:
                import numpy as np
                start = {'l': np.array([0.0, 0.0]), 'f': np.array([10.0, 0.0])}
                for z_index, stage in enumerate(DISPLAY_STAGES):

                    print("###################################### STAGE:", stage, "############################################")
                    if stage == UnfoldStage.NET:
                        height_increment = 0
                    else:
                        height_increment = -3
                    flat_scutoid_node = render_polygon_flat_face(scutoid_points, scutoid_faces, face_colors, -20 + -40 * z_index, start, stage, height_increment)
                    flat_scutoid_node_path = NodePath(flat_scutoid_node)
                    flat_scutoid_node_path.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))
                    flat_scutoid_node_path.reparentTo(self.render)


if __name__ == "__main__":
    #load_prc_file('settings.prc')
    base = main()
    base.run()

# controls!!
# zoom: hold righ mouse button and move op or down
# rotate: press mouse weel and move the mouse.

# this can be disabled with a disable mouse command.

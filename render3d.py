# naar voorbeeld van Gerlinde.

from math import cos, pi, sin

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
)

format = GeomVertexFormat.getV3c4()
format = GeomVertexFormat.registerFormat(format)

import scutoid as scut
from scutoid import LetteredPolyhedron


def polyhedron_face(points_dict, face):
    pass

def polyhedron(scutoid_data):
    pass
#    lines.moveTo(1, 1, 1)
#    lines.drawTo(2, 2, 2)

def lines_example():
    lines = LineSegs()
    lines.moveTo(1, 1, 1)
    lines.drawTo(2, 2, 2)
    lines.setThickness(4)
    return lines

def cube_example():
    # Instantiate a vertex buffer
    # https://docs.panda3d.org/1.10/python/programming/internal-structures/procedural-generation/creating-vertex-data
    vdata = GeomVertexData("name", format, Geom.UHStatic)
    vertex = GeomVertexWriter(vdata, "vertex")
    color = GeomVertexWriter(vdata, "color")

    # Add vertices and colors
    vertex.addData3f(-1, -1, -1)
    color.addData4f(0, 0, 0, 1)

    vertex.addData3f(-1, -1, 1)
    color.addData4f(0, 0, 1, 1)

    vertex.addData3f(-1, 1, -1)
    color.addData4f(0, 1, 0, 1)

    vertex.addData3f(-1, 1, 1)
    color.addData4f(0, 1, 1, 1)

    vertex.addData3f(1, -1, -1)
    color.addData4f(1, 0, 0, 1)

    vertex.addData3f(1, -1, 1)
    color.addData4f(1, 0, 1, 1)

    vertex.addData3f(1, 1, -1)
    color.addData4f(1, 1, 0, 1)

    vertex.addData3f(1, 1, 1)
    color.addData4f(1, 1, 1, 1)

    # Create the triangles (2 per face)
    # https://docs.panda3d.org/1.10/python/programming/internal-structures/procedural-generation/creating-primitives
    prim = GeomTriangles(Geom.UHStatic)
    prim.addVertices(0, 1, 2)
    prim.addVertices(2, 1, 3)
    prim.addVertices(2, 3, 6)
    prim.addVertices(6, 3, 7)
    prim.addVertices(6, 7, 4)
    prim.addVertices(4, 7, 5)
    prim.addVertices(4, 5, 0)
    prim.addVertices(0, 5, 1)
    prim.addVertices(1, 5, 3)
    prim.addVertices(3, 5, 7)
    prim.addVertices(6, 4, 2)
    prim.addVertices(2, 4, 0)

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("node")
    node.addGeom(geom)

    return node


class main(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        lines = lines_example()

#        scutoid_data = scut.scutoid1_alt()
#        scutoid_polyhedron = polyhedron(scutoid_data)

#        lines.setThickness(4)
        lines_node = lines.create()
        np = NodePath(lines_node)
        np.reparentTo(self.render)
        cube_node = cube_example()
        cube_node_path = NodePath(cube_node)
        cube_node_path.reparentTo(self.render)



if __name__ == "__main__":
    #load_prc_file('settings.prc')
    base = main()
    base.run()

# controls!!
# zoom: hold righ mouse button and move op or down
# rotate: press mouse weel and move the mouse.

# this can be disabled with a disable mouse command.

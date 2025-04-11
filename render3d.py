# naar voorbeeld van Gerlinde.

import panda3d
from direct.showbase.ShowBase import ShowBase
from panda3d.core import LineSegs, NodePath

import scutoid as scut
from scutoid import LetteredPolyhedron


def polyhedron_face(points_dict, face):
    pass

def polyhedron(scutoid_data):
    pass
#    lines.moveTo(1, 1, 1)
#    lines.drawTo(2, 2, 2)


class main(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        lines = LineSegs()

        scutoid_data = scut.scutoid1_alt()
        scutoid_polyhedron = polyhedron(scutoid_data)

        lines.setThickness(4)
        node = self.lines.create()
        np = NodePath(node)
        np.reparentTo(self.render)


if __name__ == "__main__":
    #load_prc_file('settings.prc')
    base = main()
    base.run()

# controls!!
# zoom: hold righ mouse button and move op or down
# rotate: press mouse weel and move the mouse.

# this can be disabled with a disable mouse command.

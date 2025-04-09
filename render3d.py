# naar voorbeeld van Gerlinde.

import panda3d
from direct.showbase.ShowBase import ShowBase
from panda3d.core import LineSegs, NodePath

import scutoid
from scutoid import LetteredPolygon


class main(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        lines = LineSegs()

        sutoid_data = scutoid.scutoid_1
        scutoid = polynome(scutoid_data)

        lines.setThickness(4)
        node = self.lines.create()
        np = NodePath(node)
        np.reparentTo(self.render)

    def polynome_face(self, points_dict, face):
        pass

    def polyome(self, points_dict, faces):
        pass
#        lines.moveTo(1, 1, 1)
#        lines.drawTo(2, 2, 2)



if __name__ == "__main__":
    #load_prc_file('settings.prc')
    base = main()
    base.run()

# controls!!
# zoom: hold righ mouse button and move op or down
# rotate: press mouse weel and move the mouse.

# this can be disabled with a disable mouse command.

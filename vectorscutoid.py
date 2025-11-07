# vectorscutoid.py
import cairo
import scutoid
import numpy as np


POINT_SIZE = .375
A4_LENGTH = 840
A4_WIDTH = 595

A4_SIZE = np.array([A4_WIDTH, A4_LENGTH])


def vectorscutoid(scutoid_points, scutoid_faces, start, context):
#    with cairo.PDFSurface("scutoid_net.pdf", 210.0 / POINT_SIZE, 297.0 / POINT_SIZE) as surface:
#    context = cairo.Context(surface)

    context.stroke()
    context.set_source_rgba(0.0, 0.0, 0.0, 1.0)
    context.set_line_width(1.0)


    def draw_face_flat(face_points):
        # in order to give the whole face a solid color
        # we add the vertex seperately for each face
        # add the first vertex to the face
        context.move_to(*face_points[list(face_points.keys())[0]])
        for point in list(face_points.keys())[1:] :
            context.line_to(*face_points[point])
        context.line_to(*face_points[list(face_points.keys())[0]])

    added_faces = {}
    for face in scutoid_faces:
        # find a face that is find_adjacent_face
        adjacent_face_name = scutoid.find_adjacent_face(face, added_faces.keys())
        if adjacent_face_name is None:  # initialize
            adjacent = start
        else:
            adjacent = added_faces[adjacent_face_name]
        print(list(added_faces.keys()))
        points = scutoid.face_flat_adjacent(face, adjacent, scutoid_points, scutoid.UnfoldStage.NET)
        added_faces[face] = points
        draw_face_flat(points)

#        context.stroke()


def main():
    with cairo.PDFSurface("scutoid_net.pdf", 210.0 / POINT_SIZE, 297.0 / POINT_SIZE) as surface:
        context = cairo.Context(surface)

        offset_1 = np.array([200.0, 150.0])
        side_1 = np.array([100.0, 125.0])

        scutoid_points, scutoid_faces = scutoid.scutoid1()
        start = {'l': offset_1, 'f': offset_1 + side_1}
        vectorscutoid(scutoid_points, scutoid_faces, start, context)

        # now the mirror

#        scutoid_points, scutoid_faces = scutoid.scutoid1()

        offset_2 = np.array([200.0, 650.0])
        side_2 = np.array([-100.0, 125.0])

        scutoid_points, scutoid_faces = scutoid.mirror(scutoid_points, scutoid_faces)
        start = {'l':offset_2, 'f':offset_2 + side_2}
        vectorscutoid(scutoid_points, scutoid_faces, start, context)

        context.stroke()
    print("file saved! - probably")


if __name__ == "__main__":
    main()

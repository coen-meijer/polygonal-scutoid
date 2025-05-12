# vectorscutoid.py
import cairo
import scutoid
import numpy as np


POINT_SIZE = .375
A4_LENGTH = 840
A4_WIDTH = 595


def vectorscutoid(scutoid_points, scutoid_faces, start):
    with cairo.PDFSurface("scutoid_net.pdf", 210.0 / POINT_SIZE, 297.0 / POINT_SIZE) as surface:
        context = cairo.Context(surface)

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

        context.stroke()

    print("file saved?")


def main():
    scutoid_points, scutoid_faces = scutoid.scutoid1()
    start = {'l': np.array([200.0, 200.0]), 'f': np.array([300.0, 300.0])}
    vectorscutoid(scutoid_points, scutoid_faces, start)


if __name__ == "__main__":
    main()

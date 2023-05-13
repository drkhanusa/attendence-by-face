import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
import insightface

color = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255)]
def draw_on(img, faces, index_min):
    dimg = img.copy()
    face = faces[0]
    box = face.bbox.astype(int)
    thickness = 3
    x = int((int(box[0])+int(box[2]))/2)
    y = int((int(box[1])+int(box[3]))/2)
    if index_min is not None:
        color[index_min] = (0, 255, 0)
    cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color[0], 2)

    cv2.arrowedLine(dimg, (x, box[1] - 5), (x, box[1] - 40), color[1], thickness)
    cv2.arrowedLine(dimg, (x, box[3] + 5), (x, box[3] + 40), color[2], thickness)

    cv2.arrowedLine(dimg, (box[0] - 5, y), (box[0] - 40, y), color[3], thickness)
    cv2.arrowedLine(dimg, (box[2] + 5, y), (box[2] + 40, y), color[4], thickness)

    return dimg


if __name__ == "__main__":

    app = FaceAnalysis(allowed_modules=['detection'])# enable detection model only
    app.prepare(ctx_id=0, det_size=(640, 640))

    handler = insightface.model_zoo.get_model("C:/Users/hungt/.insightface/models/buffalo_l/w600k_r50.onnx")
    handler.prepare(ctx_id=0)

    kps_opposite = [[281.54736, 176.55824], [363.72873, 174.48114], [326.08832, 223.83246], [296.56854, 267.02502], [355.87146, 264.92523]]
    kps_up = [[287.7631, 127.96423], [352.5035, 125.50511], [322.28873, 140.3377], [296.64462, 186.80501], [352.32675, 183.93025]]
    kps_down = [[290.49832, 243.33574], [373.0931, 240.86241], [334.33276, 301.59393], [303.1017, 319.3982], [360.49393, 318.09207]]
    kps_right = [[368.60593, 158.41211], [399.65457, 157.67996], [421.69885, 196.44415], [388.15115, 244.27081], [409.69754, 243.29143]]
    kps_left = [[219.35115, 170.01491], [268.74713, 164.97511], [211.61333, 204.1819 ], [219.03375, 253.70105], [256.42157, 249.71962]]
    embedding_vectors = np.zeros((512, 5))

    vid = cv2.VideoCapture(0)
    while (True):
        ret, frame = vid.read()
        # Display the resulting frame
        faces = app.get(frame)
        distance_opposite, distance_up, distance_down, distance_right, distance_left = 0, 0, 0, 0, 0
        if faces == []:
            cv2.imshow('frame', frame)
        else:
            if len(faces) > 1:
                cv2.putText(frame, 'Only one face in screen', (140, 240), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 1)
                cv2.imshow('frame', frame)

            else:
                keypoint = faces[0]['kps']
                for i in range(5):
                    distance_opposite += np.linalg.norm(kps_opposite[i] - keypoint[i])
                    distance_up += np.linalg.norm(kps_up[i] - keypoint[i])
                    distance_down += np.linalg.norm(kps_down[i] - keypoint[i])
                    distance_left += np.linalg.norm(kps_left[i] - keypoint[i])
                    distance_right += np.linalg.norm(kps_right[i] - keypoint[i])
                distance = [distance_opposite, distance_up, distance_down, distance_left, distance_right]
                index_min = np.argmin(distance)
                print(distance[index_min])

                if distance[index_min] < 60:
                    embedding = handler.get(frame, faces[0])
                    embedding_vectors[:, index_min] = embedding
                    rimg = draw_on(frame, faces, index_min)
                else:
                    rimg = draw_on(frame, faces, None)
                cv2.imshow('frame', rimg)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            name = 'khanhnt'
            save = pd.DataFrame(embedding_vectors)
            save.to_csv(f"E:/PYTHON/pythonProject/insightface/python-package/insightface/data/Faces_database/{name}.csv")
            break
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

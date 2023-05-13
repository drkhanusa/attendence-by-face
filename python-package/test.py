import cv2
import numpy as np
import pandas as pd

import insightface
from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image

app = FaceAnalysis(allowed_modules=['detection'])
app.prepare(ctx_id=0, det_size=(640, 640))

handler = insightface.model_zoo.get_model('C:/Users/hungt/.insightface/models/buffalo_l/w600k_r50.onnx')
handler.prepare(ctx_id=0)

path = "E:/PYTHON/pythonProject/insightface/python-package/insightface/data/Faces_database/khanhnt.csv"
embedding_vectors = pd.read_csv(path).values
name = " "

# define a video capture object
vid = cv2.VideoCapture(0)
while (True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    # Display the resulting frame
    faces = app.get(frame)
    if faces == []:
        cv2.imshow('frame', frame)
    else:
        if len(faces) > 1:
            cv2.putText(frame, 'Only one face in screen', (140, 240), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 1)
            cv2.imshow('frame', frame)

        else:
            rimg = app.draw_on(frame, faces, name)
            cv2.imshow('frame', rimg)

            embedding = handler.get(frame, faces[0])
            # embedding = app.get(frame)[-1]['embedding']
            for i in range(1, 6):
                distance = np.linalg.norm(embedding - embedding_vectors[:, i])
                if distance < 20:
                    name = (path.split("/")[-1]).split(".")[0]
                else:
                    name = " "
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import insightface
# from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image
#
# app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# app.prepare(ctx_id=0, det_size=(640, 640))
# img_opp = cv2.imread("E:/khanhnt/opposite.jpg")
# img_up = cv2.imread("E:/khanhnt/up.jpg")
# img_down = cv2.imread("E:/khanhnt/down.jpg")
# img_right = cv2.imread("E:/khanhnt/right.jpg")
# img_left = cv2.imread("E:/khanhnt/left.jpg")
#
# faces_opp = app.get(img_opp)
# faces_up = app.get(img_up)
# faces_down = app.get(img_down)
# faces_right = app.get(img_right)
# faces_left = app.get(img_left)
#
# print("opposite: ", faces_opp[0]['kps'])
# print("up: ", faces_up[0]['kps'])
# print("down: ", faces_down[0]['kps'])
# print("right: ", faces_right[0]['kps'])
# print("left: ", faces_left[0]['kps'])
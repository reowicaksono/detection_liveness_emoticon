import sys
import argparse
import copy
import datetime

import numpy as np
import cv2 as cv

from facial_fer_model import FacialExpressionRecog

sys.path.append('../face_detection_yunet')
from yunet import YuNet

# Check OpenCV version
assert cv.__version__ >= "4.8.0", \
       "Please install latest opencv-python to try this demo: python3 -m pip install --upgrade opencv-python"

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(description='Facial Expression Recognition')
parser.add_argument('--input', '-i', type=str,
                    help='Path to the input image. Omit for using default camera.')
parser.add_argument('--model', '-m', type=str, default='facial_expression_recognition_mobilefacenet_2022july.onnx',
                    help='Path to the facial expression recognition model.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--save', '-s', action='store_true',
                    help='Specify to save results. This flag is invalid when using camera.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Specify to open a window for result visualization. This flag is invalid when using camera.')
args = parser.parse_args()

# def remove_background(image_path, output_path):
#     # Baca gambar PNG dengan channel alpha (RGBA)
#     image = cv.imread(image_path, cv.IMREAD_UNCHANGED)

#     # Ambil alpha channel (kanal transparansi)
#     alpha_channel = image[:, :, 3]

#     # Ambil nilai ambang untuk latar belakang yang ingin dihilangkan
#     threshold = 100

#     # Terapkan ambang
#     _, thresh = cv.threshold(alpha_channel, threshold, 255, cv.THRESH_BINARY)

#     # Temukan kontur dari objek yang ingin dipertahankan
#     contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#     # Buat mask untuk objek yang ingin dipertahankan
#     mask = np.zeros_like(alpha_channel)
#     cv.drawContours(mask, contours, -1, (255), thickness=cv.FILLED)

#     # Gabungkan mask dengan gambar asli
#     result = cv.bitwise_and(image, image, mask=mask)

#     # Simpan hasil ke file output
#     cv.imwrite(output_path, result)

def visualize(image, det_res, fer_res, box_color=(0, 255, 0), text_color=(0, 0, 255)):
    # print('%s %3d faces detected.' % (datetime.datetime.now(), len(det_res)))

    output = image.copy()
    landmark_color = [
        (255, 0, 0),  # right eye
        (0, 0, 255),  # left eye
        (0, 255, 0),  # nose tip
        (255, 0, 255),  # right mouth corner
        (0, 255, 255)  # left mouth corner
    ]

    for ind, (det, fer_type) in enumerate(zip(det_res, fer_res)):
        bbox = det[0:4].astype(np.int32)
        fer_type = FacialExpressionRecog.getDesc(fer_type)
        # print("Face %2d: %d %d %d %d %s." % (ind, bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], fer_type))

        # Calculate the center of the face bounding box
        face_center = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2

        try:
            # Load the emoticon image
            emoticon_path = "../facial_expression_recognition/emoticon/{}.png".format(fer_type)
            emoticon_image = cv.imread(emoticon_path)

            # Resize the emoticon image to a smaller size
            new_size = (bbox[2] // 2, bbox[3] // 2)
            emoticon_image = cv.resize(emoticon_image, new_size)

            # Remove the background of the emoticon image (set background to black)
            mask = cv.threshold(emoticon_image[:, :, 2], 100, 255, cv.THRESH_BINARY)[1]
            emoticon_image[mask == 0] = [0, 0, 0]  # Set background to black

            # Place the emoticon image at a specified offset from the center of the face bounding box
            offset = 10
            emoticon_x = face_center[0] - emoticon_image.shape[1] // 2 + offset
            emoticon_y = face_center[1] - emoticon_image.shape[0] // 2

            # Place the emoticon image in the output image
            output[emoticon_y:emoticon_y + emoticon_image.shape[0], emoticon_x:emoticon_x + emoticon_image.shape[1]] = emoticon_image

            # Draw a border around the emoticon image
            cv.rectangle(output, (emoticon_x, emoticon_y), (emoticon_x + emoticon_image.shape[1], emoticon_y + emoticon_image.shape[0]), (0, 0, 255), 2)

        except Exception as e:
            print(f"Error loading emoticon for {fer_type}: {e}")

        # Draw a rectangle around the face
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

        # Draw the expression label on the face
        cv.putText(output, fer_type, (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        # Draw landmarks on the face
        landmarks = det[4:14].astype(np.int32).reshape((5, 2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(output, landmark, 2, landmark_color[idx], 2)

    return output







def process(detect_model, fer_model, frame):
    h, w, _ = frame.shape
    detect_model.setInputSize([w, h])
    dets = detect_model.infer(frame)

    if dets is None:
        return False, None, None

    fer_res = np.zeros(0, dtype=np.int8)
    for face_points in dets:
        fer_res = np.concatenate((fer_res, fer_model.infer(frame, face_points[:-1])), axis=0)
    return True, dets, fer_res


if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    detect_model = YuNet(modelPath='../face_detection_yunet/face_detection_yunet_2023mar.onnx')

    fer_model = FacialExpressionRecog(modelPath=args.model,
                                      backendId=backend_id,
                                      targetId=target_id)

    # If input is an image
    if args.input is not None:
        image = cv.imread(args.input)

        # Get detection and fer results
        status, dets, fer_res = process(detect_model, fer_model, image)

        if status:
            # Draw results on the input image
            image = visualize(image, dets, fer_res)

        # Save results
        if args.save:
            cv.imwrite('result.jpg', image)
            print('Results saved to result.jpg\n')

        # Visualize results in a new window
        if args.vis:
            cv.namedWindow(args.input, cv.WINDOW_AUTOSIZE)
            cv.imshow(args.input, image)
            cv.waitKey(0)
    else:  # Omit input to call default camera
        deviceId = 0
        cap = cv.VideoCapture(deviceId)

        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            # Get detection and fer results
            status, dets, fer_res = process(detect_model, fer_model, frame)

            if status:
                # Draw results on the input image
                frame = visualize(frame, dets, fer_res)

            # Visualize results in a new window
            cv.imshow('FER Demo', frame)

"""
    This code will take care of surveillance.
"""
# pylint: disable=E1101
# pylint: disable=E0211
# pylint: disable=W0603
import datetime
import threading

import cv2
from mtcnn import MTCNN
from flask import Response, Flask, render_template

OUTPUT_IMAGE, LOCK, VIDEO_STREAM = None, threading.Lock(), cv2.VideoCapture(0)
DETECTOR = MTCNN()

APP = Flask(__name__, template_folder="frontend")

class Surveillance:
    """
        This class contains all the methods of surveillance.
    """
    @APP.route("/")
    def index():
        """
            Index page which will be hit when you open base URL.
        """
        return render_template("index.html")

    @classmethod
    def detect_motion(cls):
        """
            This will detect motion of video.
        """
        global OUTPUT_IMAGE, LOCK, VIDEO_STREAM
        while True:
            _, frame = VIDEO_STREAM.read()
            frame = cv2.resize(frame, (frame.shape[1], frame.shape[0]))
            frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = DETECTOR.detect_faces(frame1)

            if result != []:
                for person in result:
                    bounding_box = person['box']
                    keypoints = person['keypoints']

                    cv2.rectangle(frame, (bounding_box[0], bounding_box[1]),\
                                 (bounding_box[0]+bounding_box[2],\
                                    bounding_box[1] + bounding_box[3]), (0, 155, 255), 2)
                    cv2.circle(frame, (keypoints['left_eye']), 2, (0, 155, 255), 2)
                    cv2.circle(frame, (keypoints['right_eye']), 2, (0, 155, 255), 2)
                    cv2.circle(frame, (keypoints['nose']), 2, (0, 155, 255), 2)
                    cv2.circle(frame, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
                    cv2.circle(frame, (keypoints['mouth_right']), 2, (0, 155, 255), 2)

            timestamp = datetime.datetime.now()
            cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"),\
                                    (10, frame.shape[0] - 10),\
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            with LOCK:
                OUTPUT_IMAGE = frame.copy()

    @classmethod
    def generate(cls):
        """
            This will generate output_image.
        """
        global OUTPUT_IMAGE, LOCK
        while True:
            with LOCK:
                if OUTPUT_IMAGE is None:
                    continue
                flag, encoded_image = cv2.imencode(".jpg", OUTPUT_IMAGE)
                if not flag:
                    continue
            yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n'\
                                                + bytearray(encoded_image) + b'\r\n'

    @APP.route("/video_feed")
    def video_feed():
        """
            This will return frame on calling UI.
        """
        return Response(Surveillance().generate(),\
                        mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    mythread = threading.Thread(target=Surveillance().detect_motion)
    mythread.daemon = True
    mythread.start()

    APP.config["CACHE_TYPE"] = "null"
    APP.jinja_env.cache = {}
    APP.run(host="127.0.0.1", port=65000, debug=True, threaded=True, use_reloader=False)

VIDEO_STREAM.release()

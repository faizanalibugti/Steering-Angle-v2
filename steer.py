import numpy as np
import cv2
from grabscreen import grab_screen
import time
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

def keras_predict(model, image):
    processed = keras_process_image(image)
    steering_angle = float(model.predict(processed, batch_size=1))
    steering_angle = steering_angle * 60
    return steering_angle


def keras_process_image(img):
    image_x = 100
    image_y = 100
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    #config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras
    
    steer = cv2.imread('steering_wheel_image.jpg', 0)
    rows, cols = steer.shape
    smoothed_angle = 0

    # Load Keras model
    model = load_model('steer.h5')

    while (True):
        last_time = time.time()
        screen = grab_screen(region=(0, 40, 1000, 600))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        #gray = cv2.resize((cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))[:, :, 1], (500, 500))
        steering_angle = keras_predict(model, gray)
        print("Predicted steering angle: {}".format(steering_angle))
        # cv2.imshow('frame', cv2.resize(frame, (600, 400), interpolation=cv2.INTER_AREA))
        cv2.imshow('Screen Capture', screen)
        cv2.imshow('Neural Network Input', gray)
        smoothed_angle += 0.2 * pow(abs((steering_angle - smoothed_angle)), 2.0 / 3.0) * (
            steering_angle - smoothed_angle) / abs(
            steering_angle - smoothed_angle)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
        dst = cv2.warpAffine(steer, M, (cols, rows))
        cv2.imshow("steering wheel", dst)

        #cv2.imshow('window', screen)r
        print("fps: {}".format(1 / (time.time() - last_time)))

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break


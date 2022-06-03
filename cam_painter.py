import numpy as np
import cv2
import tensorflow as tf
import hand_tracking_module as hand_tracking
import time

def process_image(img, shape):
    img = cv2.resize(img, shape)
    img = img / 255.
    return img

model = tf.keras.models.load_model("draw_model.h5")

labels = ['banana', 'rainbow', 'church', 'pants', 'sun', 'pizza', 'circle', 'cloud']
labels_copy = labels.copy()
actual_label = ""
nn_prediction = ""
draw_finished = False
header_img = cv2.imread("Header/header.jpg")

# Get capture and set config
cap = cv2.VideoCapture(0)
cap.set(3, 1288)
cap.set(4, 728)

hand_detector = hand_tracking.handDetector()
draw_color = (0, 0, 255)

selection_mode = [0, 1]
idx_selection = 0
xp, yp = 0, 0
img_canvas = np.zeros((720, 1280, 3), np.uint8)
first_frame = True
start_time = time.time()
end_time = 0
count = 0
while True:

    _, frame = cap.read()

    if _:
        if draw_finished or first_frame:
            count += 1
            idx_label = np.random.randint(0, len(labels))
            print(idx_label)
            actual_label = labels[idx_label]

            if first_frame:
                first_frame = False
            if draw_finished:
                draw_finished = False

        frame = cv2.flip(frame, 1)
        frame = hand_detector.find_hands(frame)
        lm_list = hand_detector.find_position(frame)

        if len(lm_list) != 0:
            xp, yp = 0, 0
            x1, y1 = lm_list[8][1:]
            x2, y2 = lm_list[12][1:]

            fingers = hand_detector.fingers_up()

            if fingers[1] and fingers[2]:
                # print("Pen, rubber and send result selection mode")
                if y1 < 125:
                    if 1000 < x1 < 1100:
                        draw_color = (0, 0, 255)
                    elif 1100 < x1 < 1280:
                        draw_color = (0, 0, 0)
                    elif 590 < x1 < 690 and (end_time - start_time) > 5:
                        draw_finished = True


                cv2.rectangle(frame, (x1, y1 - 25), (x2, y2 + 25), draw_color, cv2.FILLED)

            if fingers[1] and not fingers[2]:
                # print("entrei")
                cv2.circle(frame, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
                # print("Drawing mode")
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                if draw_color == (0, 0, 0):
                    cv2.line(frame, (xp, yp), (x1, y1), draw_color, 225)
                    cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, 225)
                else:
                    cv2.line(frame, (xp, yp), (x1, y1), draw_color, 45)
                    cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, 45)

                xp, yp = x1, y1

        img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, img_inv)
        frame = cv2.bitwise_or(frame, img_canvas)

        frame[0:125, 0:1280] = header_img
        cv2.putText(frame, f"{count}/{len(labels_copy)}", (620, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, actual_label, (310, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # print(nn_prediction)
        cv2.putText(frame, nn_prediction, (800, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # frame = cv2.addWeighted(frame, 8.5, img_canvas, 0.5, 0)
        cv2.imshow("Frame", frame)
        img_teste = cv2.resize(img_inv, (28, 28))
        cv2.imshow("img_inv", img_teste)
        cv2.imshow("img_invvv", img_inv)
        cv2.waitKey(1)

        if draw_finished and (end_time - start_time) > 5:
            processed_img = process_image(img_inv, (28, 28))
            preds = model.predict(tf.expand_dims(processed_img, axis=0))
            print(preds)
            num = max(preds[0])
            idx = list(preds[0]).index(num)
            nn_prediction = labels_copy[idx]
            img_canvas = np.zeros((720, 1280, 3), np.uint8)
            print(end_time - start_time)
            start_time = time.time()
            print(nn_prediction)
            labels.remove(actual_label)

        if count == 9:
            break

    cap.release()
    end_time = time.time()
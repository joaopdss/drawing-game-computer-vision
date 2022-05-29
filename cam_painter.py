import numpy as np
import cv2
import os
import hand_tracking_module as hand_tracking

header_img = cv2.imread("Header/header.jpg")

cap = cv2.VideoCapture(0)
cap.set(3, 1288)
cap.set(4, 728)
hand_detector = hand_tracking.handDetector()
draw_color = (0, 0, 255)

selection_mode = [0, 1]
idx_selection = 0
xp, yp = 0, 0
img_canvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    _, frame = cap.read()

    if _:
        frame = cv2.flip(frame, 1)
        frame = hand_detector.find_hands(frame)
        lm_list = hand_detector.find_position(frame)

        if len(lm_list) != 0:
            x1, y1 = lm_list[8][1:]
            x2, y2 = lm_list[12][1:]

            fingers = hand_detector.fingers_up()

            if fingers[1] and fingers[2]:
                xp, yp = 0, 0

                print("Pen & rubber selection mode")
                if y1 < 125:
                    if x1 > 1100 and x1 < 1280:
                        draw_color = (0, 0, 0)
                    if x1 > 950 and x1 < 1100:
                        draw_color = (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1 - 25), (x2, y2 + 25), draw_color, cv2.FILLED)

            print(x1)
            if fingers[1] and not fingers[2]:
                print("entrei")
                cv2.circle(frame, (x1, y1), 12, (0, 0, 255), cv2.FILLED)
                print("Drawing mode")
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                if draw_color == (0, 0, 0):
                    cv2.line(frame, (xp, yp), (x1, y1), draw_color, 125)
                    cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, 125)
                else:
                    cv2.line(frame, (xp, yp), (x1, y1), draw_color, 25)
                    cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, 25)

                xp, yp = x1, y1

        img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, img_inv)
        frame = cv2.bitwise_or(frame, img_canvas)

        frame[0:125, 0:1280] = header_img
        # frame = cv2.addWeighted(frame, 8.5, img_canvas, 0.5, 0)
        cv2.imshow("Frame", frame)
        cv2.imshow("Canvas", img_canvas)
        cv2.waitKey(1)
import cv2
import mediapipe as mp

class handDetector():
    def __init__(self, mode=False, max_num_hands=1, min_detection_conf=0.5, min_tracking_conf=0.5):
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf = min_tracking_conf

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(self.mode, self.max_num_hands)

        self.tips_ids = [4, 8, 12, 16, 20]

    def find_hands(self, frame):
        self.results = self.hands.process(frame)

        if self.results.multi_hand_landmarks:
            self.mp_draw.draw_landmarks(frame, self.results.multi_hand_landmarks[0], self.mp_hands.HAND_CONNECTIONS,
                                        self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
        return frame

    def find_position(self, frame, hand_num=0):
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            hand_point = self.results.multi_hand_landmarks[hand_num]

            for id, lm in enumerate(hand_point.landmark):
                height, width, channels = frame.shape
                cx, cy = int(lm.x*width), int(lm.y*height)
                self.lm_list.append([id, cx, cy])
                # if id == 8:
                #     cv2.circle(frame, (cx, cy), 12, (0, 0, 255), cv2.FILLED)
        return self.lm_list

    def fingers_up(self):
        fingers = []

        if self.lm_list[self.tips_ids[0]][1] < self.lm_list[self.tips_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.lm_list[self.tips_ids[id]][2] < self.lm_list[self.tips_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
# def run():
#     cap = cv2.VideoCapture(0)
#     detector = handDetector()
#     while True:
#         _, frame = cap.read()
#
#         if _:
#             frame = detector.find_hands(frame)
#             lm_list = detector.find_position(frame)
#             if len(lm_list) != 0:
#                 print(lm_list[8])
#             cv2.imshow("Frame", frame)
#             cv2.waitKey(1)
#
# run()
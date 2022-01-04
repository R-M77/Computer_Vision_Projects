import cv2 as cv
import numpy as np
import mediapipe as mp
import Key_Inputter
import math

mp_draw = mp.solutions.drawing_utils
mp_draw_style = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
font = cv.FONT_HERSHEY_SIMPLEX

# Select camera | 0 for webcam
cap = cv.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # 'continue' for live feed video
            # 'break' for uploaded video
            continue   

    
    # To improve performance, optionally mark the img as not writeable to
        img.flags.writeable = False
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = hands.process(img)
        imgHeight, imgWidth, _ = img.shape

        # Draw the hand annotations on the img.
        img.flags.writeable = True
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        co=[]
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw_style.get_default_hand_landmarks_style(),
                    mp_draw_style.get_default_hand_connections_style())
                for point in mp_hands.HandLandmark:
                    if str(point) == "HandLandmark.WRIST":
                        normalizedLandmark = hand_landmarks.landmark[point]
                        pixelCoordinatesLandmark = mp_draw._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                            normalizedLandmark.y,
                                                                                        imgWidth, imgHeight)

                        try:
                            co.append(list(pixelCoordinatesLandmark))
                        except:
                            continue

        if len(co) == 2:
            xm, ym = (co[0][0] + co[1][0]) / 2, (co[0][1] + co[1][1]) / 2
            radius = 150
            try:
                m=(co[1][1]-co[0][1])/(co[1][0]-co[0][0])
            except:
                continue
            a = 1 + m ** 2
            b = -2 * xm - 2 * co[0][0] * (m ** 2) + 2 * m * co[0][1] - 2 * m * ym
            c = xm ** 2 + (m ** 2) * (co[0][0] ** 2) + co[0][1] ** 2 + ym ** 2 - 2 * co[0][1] * ym - 2 * co[0][1] * co[0][
                0] * m + 2 * m * ym * co[0][0] - 22500
            xa = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
            xb = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
            ya = m * (xa - co[0][0]) + co[0][1]
            yb = m * (xb - co[0][0]) + co[0][1]
            if m!=0:
                ap = 1 + ((-1/m) ** 2)
                bp = -2 * xm - 2 * xm * ((-1/m) ** 2) + 2 * (-1/m) * ym - 2 * (-1/m) * ym
                cp = xm ** 2 + ((-1/m) ** 2) * (xm ** 2) + ym ** 2 + ym ** 2 - 2 * ym * ym - 2 * ym * xm * (-1/m) + 2 * (-1/m) * ym * xm - 22500
                try:
                    xap = (-bp + (bp ** 2 - 4 * ap * cp) ** 0.5) / (2 * ap)
                    xbp = (-bp - (bp ** 2 - 4 * ap * cp) ** 0.5) / (2 * ap)
                    yap = (-1 / m) * (xap - xm) + ym
                    ybp = (-1 / m) * (xbp - xm) + ym

                except:
                    continue

            cv.circle(img=img, center=(int(xm), int(ym)), radius=radius, color=(195, 255, 62), thickness=15)

            l = (int(math.sqrt((co[0][0] - co[1][0]) ** 2 * (co[0][1] - co[1][1]) ** 2)) - 150) // 2
            cv.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (195, 255, 62), 20)

            if co[0][0] < co[1][0] and co[0][1] - co[1][1] > 65:
                # When the slope is negative, we turn left.
                print("Turn right.")
                Key_Inputter.release_key('s')
                Key_Inputter.release_key('a')
                Key_Inputter.press_key('d')
                cv.putText(img, "Turn right", (50, 50), font, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                cv.line(img, (int(xap), int(yap)), (int(xm), int(ym)), (195, 255, 62), 20)

            elif co[1][0] > co[0][0] and co[1][1] - co[0][1] > 65:
                print("Turn left.")
                Key_Inputter.release_key('s')
                Key_Inputter.release_key('d')
                Key_Inputter.press_key('a')
                cv.putText(img, "Turn left", (50, 50), font, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                cv.line(img, (int(xbp), int(ybp)), (int(xm), int(ym)), (195, 255, 62), 20)

            else:
                print("keeping straight")
                Key_Inputter.release_key('s')
                Key_Inputter.release_key('a')
                Key_Inputter.release_key('d')
                Key_Inputter.press_key('w')
                cv.putText(img, "keep straight", (50, 50), font, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                if ybp>yap:
                    cv.line(img, (int(xbp), int(ybp)), (int(xm), int(ym)), (195, 255, 62), 20)
                else:
                    cv.line(img, (int(xap), int(yap)), (int(xm), int(ym)), (195, 255, 62), 20)

        if len(co)==1:
            print("keeping back")
            Key_Inputter.release_key('a')
            Key_Inputter.release_key('d')
            Key_Inputter.release_key('w')
            Key_Inputter.press_key('s')
            cv.putText(img, "keeping back", (50, 50), font, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        # show img frames, quit if 'q' pressed
        cv.imshow('MediaPipe Hands', cv.flip(img, 1))
        if cv.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release()
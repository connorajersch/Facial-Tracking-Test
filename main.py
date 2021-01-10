import os
import platform
import cv2
import dlib
from scipy.spatial import distance
import time

disk_dir = ""


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    EAR = (A + B) / (2 * C)
    return EAR


def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[1], mouth[7])
    B = distance.euclidean(mouth[3], mouth[5])
    C = distance.euclidean(mouth[0], mouth[4])
    MAR = (A + B) / (2 * C)
    return MAR


def main():
    global disk_dir
    plat = platform.system()
    if plat == "Windows":
        disk_dir = os.path.join(os.getenv("APPDATA"), "HSL")

    elif plat == "Linux":
        disk_dir = os.path.join(os.path.expanduser("~"), ".HSL")

    else:
        print("Unsupported operating system: %s" % plat)
        print("This software only supports Windows and Linux")
        exit(1)

    stream = cv2.VideoCapture(0)
    HOG_face_detector = dlib.get_frontal_face_detector()
    dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    isRunningMouth = False
    isRunningEye = False
    eyeOpen = True
    mouthT1 = 0
    mouthT2 = 0
    eyeT1 = 0
    eyeT2 = 0
    blinkCounter = 0

    while True:
        _, frame = stream.read()
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = HOG_face_detector(grey_frame)

        for face in faces:
            face_landmarks = dlib_facelandmark(grey_frame, face)
            left_eye = []
            right_eye = []
            inner_mouth = []

            # display left eye
            for n in range(36, 42):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                left_eye.append((x, y))
                next_point = n + 1
                if n == 41:
                    next_point = 36

                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

            # display right eye
            for n in range(42, 48):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                right_eye.append((x, y))
                next_point = n + 1
                if n == 47:
                    next_point = 42
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

            # display chin outline
            for n in range(0, 17):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                if n < 16:
                    next_point = n + 1
                    x2 = face_landmarks.part(next_point).x
                    y2 = face_landmarks.part(next_point).y
                    cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

            # display left eyebrow outline
            for n in range(17, 22):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                if n < 21:
                    next_point = n + 1
                    x2 = face_landmarks.part(next_point).x
                    y2 = face_landmarks.part(next_point).y
                    cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

            # display right eyebrow outline
            for n in range(22, 27):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                if n < 26:
                    next_point = n + 1
                    x2 = face_landmarks.part(next_point).x
                    y2 = face_landmarks.part(next_point).y
                    cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

            # display nose ridge outline
            for n in range(27, 31):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                if n < 30:
                    next_point = n + 1
                    x2 = face_landmarks.part(next_point).x
                    y2 = face_landmarks.part(next_point).y
                    cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

            # display lower nose outline
            for n in range(31, 36):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                if n < 35:
                    next_point = n + 1
                    x2 = face_landmarks.part(next_point).x
                    y2 = face_landmarks.part(next_point).y
                    cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

            # Outer mouth outline
            for n in range(48, 60):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                next_point = n + 1
                if n == 59:
                    next_point = 48
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

            # Inner mouth outline
            for n in range(60, 68):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                inner_mouth.append((x, y))
                next_point = n + 1
                if n == 67:
                    next_point = 60
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

            left_EAR = eye_aspect_ratio(left_eye)
            right_EAR = eye_aspect_ratio(right_eye)

            EAR = (left_EAR + right_EAR) / 2
            EAR = round(EAR, 2)
            #print(EAR)

            MAR = mouth_aspect_ratio(inner_mouth)
            MAR = round(MAR, 2)
            #print(MAR)
            if MAR >= 0.50 and not isRunningMouth:
                # start timer and stop it from being started again
                mouthT1 = time.perf_counter()
                print(mouthT1)
                isRunningMouth = True

            elif MAR < 0.5 and isRunningMouth:
                mouthT2 = time.perf_counter()
                print(mouthT2)
                if (mouthT2 - mouthT1) > 5:
                    print("Did you just yawn??")
                isRunningMouth = False

            '''
            Calculate Blink Values
            '''
            blinkThreshold = 0.19
            blinkTime = 0.05
            if EAR <= blinkThreshold and not isRunningEye:
                # start timer and stop it from being started again
                eyeT1 = time.perf_counter()
                print("Blink started")
                isRunningEye = True
                eyeOpen = False

            elif EAR > blinkThreshold and isRunningEye:
                eyeT2 = time.perf_counter()
                if (eyeT2 - eyeT1) > blinkTime:
                    blinkCounter += 1
                    print(eyeT2 - eyeT1)
                isRunningEye = False
                eyeOpen = True

        key = cv2.waitKey(1)
        if key == 27:
            break

        '''
        Text overlay
        '''
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.7
        fontColour = (0, 0, 255)
        lineType = 2

        # (x,y) from top left
        position1 = (10, 30)
        position2 = (10, 60)
        position3 = (10, 90)
        position4 = (10, 120)
        position5 = (10, 150)

        cv2.putText(frame, "Blink Counter: " + str(blinkCounter), position1, font, fontScale, fontColour, lineType)
        cv2.putText(frame, "Freqency: " + " blinks/s", position2, font, fontScale, fontColour, lineType)
        cv2.putText(frame, "Avg Blink Duration: " + " s", position3, font, fontScale, fontColour, lineType)
        cv2.putText(frame, "Last Blink Duration: " + " s", position4, font, fontScale, fontColour, lineType)
        cv2.putText(frame, "Eye Status: " + str(eyeOpen), position5, font, fontScale, fontColour, lineType)

        cv2.imshow("Drowsy", frame)

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

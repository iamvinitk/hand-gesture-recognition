import os
import time

import cv2
import numpy as np
import cnn
import chatgpt

# mask = 0
background = 0
counter = 0

number_of_samples = 1000
mod = 0

take_background_sub_sample = False

gesture_name = ""
save_image_flag = False
speak_flag = False


def background_sub_mask(frame, roi_x, roi_y, roi_w, roi_h, frame_count):
    global take_background_sub_sample, background
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 1)
    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if take_background_sub_sample:
        background = roi
        take_background_sub_sample = False
        print("Background captured")

    # Take the absolute difference of the background and current frame
    diff = cv2.absdiff(roi, background)

    # Threshold the diff image so that we get the foreground
    _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    mask = cv2.GaussianBlur(diff, (3, 3), 5)
    mask = cv2.erode(diff, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    mask = cv2.dilate(diff, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    res = cv2.bitwise_and(roi, roi, mask=mask)

    return res


def binary_mask(frame, roi_x, roi_y, roi_w, roi_h, frame_count):
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 1)

    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 2)

    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    _, res = cv2.threshold(threshold, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return res


def skin_mask(frame, roi_x, roi_y, roi_w, roi_h, frame_count):
    low_range = np.array([0, 50, 80], dtype=np.uint8)
    upper_range = np.array([30, 200, 255], dtype=np.uint8)

    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 1)

    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, low_range, upper_range)

    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    mask = cv2.GaussianBlur(mask, (15, 15), 1)

    res = cv2.bitwise_and(roi, roi, mask=mask)

    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    return res


def save_image(image, path):
    global counter, number_of_samples, gesture_name, save_image_flag
    if counter > number_of_samples:
        print("Finished capturing images")
        counter = 0
        gesture_name = ""
        save_image_flag = False
        print("Enter a new gesture group name, by pressing 'n'!")
        return
    name = path + str(counter + 1001) + ".png"
    print(f"Saving image {name}")
    cv2.imwrite(name, image)
    counter += 1
    time.sleep(0.05)


def main():
    global take_background_sub_sample, gesture_name, save_image_flag, speak_flag

    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 1
    fx = 10
    fy = 350
    fh = 30

    roi_x = 1000
    roi_y = 400
    roi_w = 512
    roi_h = 512

    binary_mode_flag = True
    background_sub_mode_flag = False
    quiet_mode_flag = False
    prediction_mode_flag = False

    path = ""

    # read from webcam
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Original Video', cv2.WINDOW_NORMAL)

    frame_count = 0
    fps = ""

    plot = np.zeros((512, 512, 3), np.uint8)

    start_time = time.time()

    model = cnn.get_model(train=False)

    frames = 0

    output_predictions = []

    while True:
        ret, frame = cap.read()
        roi = None
        frame = cv2.flip(frame, 1)

        if ret:
            if background_sub_mode_flag:
                roi = background_sub_mask(frame, roi_x, roi_y, roi_w, roi_h, frame_count)
            elif binary_mode_flag:
                roi = binary_mask(frame, roi_x, roi_y, roi_w, roi_h, frame_count)
            else:
                roi = skin_mask(frame, roi_x, roi_y, roi_w, roi_h, frame_count)

            frame_count += 1
            frames += 1

            end_time = time.time()
            time_diff = end_time - start_time

            if time_diff >= 1:
                fps = f"FPS: {frame_count}"
                start_time = time.time()
                frame_count = 0

        cv2.putText(frame, fps, (30, 30), font, size, (0, 255, 0), 2, 1)
        cv2.putText(frame, "Options", (fx, fy), font, size, (0, 255, 0), 2, 1)
        cv2.putText(frame, "b: Toggle Binary/Skin Mask", (fx, fy + fh), font, size, (0, 255, 0), 1, 1)
        cv2.putText(frame, "x: Toggle Background Sub Mask", (fx, fy + 2 * fh), font, size, (0, 255, 0), 1, 1)
        cv2.putText(frame, "s: Save Image", (fx, fy + 3 * fh), font, size, (0, 255, 0), 1, 1)
        cv2.putText(frame, "d: Quiet Mode", (fx, fy + 4 * fh), font, size, (0, 255, 0), 1, 1)
        cv2.putText(frame, "n: New Gesture", (fx, fy + 5 * fh), font, size, (0, 255, 0), 1, 1)
        cv2.putText(frame, "p: Prediction", (fx, fy + 6 * fh), font, size, (0, 255, 0), 1, 1)
        cv2.putText(frame, "t: Talk", (fx, fy + 7 * fh), font, size, (0, 255, 0), 1, 1)
        cv2.putText(frame, "q: Quit", (fx, fy + 8 * fh), font, size, (0, 255, 0), 1, 1)

        # Keyboard inputs
        key = cv2.waitKey(5) & 0xff

        # Quit with 'q' press
        if key == ord('q'):
            break

        # Toggle Binary/Skin Mask with 'b' press
        elif key == ord('b'):
            binary_mode_flag = not binary_mode_flag
            background_sub_mode_flag = False
            if binary_mode_flag:
                print("Binary Threshold filter active")
            else:
                print("Skin Mask filter active")

        # Toggle Background Sub Mask with 'x' press
        elif key == ord('x'):
            take_background_sub_sample = True
            background_sub_mode_flag = True
            print("Background Sub Mask filter active")

        elif key == ord('i'):
            roi_y -= 5

        elif key == ord('k'):
            roi_y += 5

        elif key == ord('j'):
            roi_x -= 5

        elif key == ord('l'):
            roi_x += 5

        elif key == ord('d'):
            quiet_mode_flag = not quiet_mode_flag
            print(f"Quiet mode is {quiet_mode_flag}")

        elif key == ord('s'):
            print("Saving image...")
            if gesture_name != "":
                save_image_flag = True
            else:
                save_image_flag = False
                print("Enter a gesture group name first, by pressing 'n'!")

        elif key == ord('n'):
            gesture_name = input("Enter gesture name: ")
            try:
                os.makedirs("./dataset/" + gesture_name)
            except OSError:
                print("Creation of the directory %s failed" % gesture_name)

            path = "./dataset/" + gesture_name + "/"

        elif key == ord('p'):
            prediction_mode_flag = not prediction_mode_flag
            if prediction_mode_flag:
                print("Prediction mode on")
            else:
                print("Prediction mode off")

        if key == ord('t'):
            speak_flag = True

        if save_image_flag:
            save_image(roi, path)

        if prediction_mode_flag:
            output_prediction = cnn.predict(model, roi)
            output_predictions.append(output_prediction)
            if frames % 10:
                output_prediction = max(set(output_predictions), key=output_predictions.count)
                output_predictions = []
                # Right top
                cv2.putText(frame, f"Gesture:{output_prediction}", (500, 30), font, size, (0, 0, 255), 2, 1)
            if speak_flag:
                # print("-------------------")
                # time.sleep(1)
                chatgpt.process_message(output_prediction)
                speak_flag = False

        if not quiet_mode_flag and roi is not None:
            cv2.imshow('Original Video', frame)
            cv2.imshow('ROI', roi)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# Hi, Ok(up), not ok (Down), Closed FIst (everything will be good), C (Clarity)

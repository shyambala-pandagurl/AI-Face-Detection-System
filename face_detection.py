
import cv2
import os

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Error: Face Cascade file not loaded!")
    exit()


# Face Detection Function
def detect_faces(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30)
    )

    if len(faces) > 0:
        status = "Face Detected"
        color = (0, 255, 0)
    else:
        status = "No Face Detected"
        color = (0, 0, 255)

    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    return frame


# Main Program
while True:

    print("\n========== MAIN MENU ==========")
    print("1. Webcam")
    print("2. Image")
    print("3. Exit")
    print("==============================")

    choice = input("Enter your choice (1/2/3): ").strip()

    # WEBCAM OPTION
    if choice == "1":

        print("\nStarting Webcam...")
        print("Press Q in the Webcam window to close Webcam.")

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            print("Error: Webcam not available!")
            camera.release()
            continue

        try:
            while True:

                ret, frame = camera.read()

                if not ret:
                    print("Unable to read Webcam.")
                    break

                frame = detect_faces(frame)

                cv2.imshow("AI Face Detection - Webcam", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == ord("Q"):
                    print("\nWebcam Closed!")
                    break

        finally:
            camera.release()
            cv2.destroyAllWindows()

        print("\nReturning to Main Menu...")
        continue

    # IMAGE OPTION
    elif choice == "2":

        print("\nImage Detection Started!")

        while True:

            image_path = input(
                "\nEnter Image Path (or type Q to close Image mode): "
            ).strip().strip('"')

            if image_path.lower() == "q":
                print("\nImage Mode Closed!")
                break

            image = cv2.imread(image_path)

            if image is None:
                print("Error: Image not found! Please enter a valid path.")
                continue

            result = detect_faces(image)

            cv2.imshow("AI Face Detection - Image", result)
            cv2.waitKey(1)

            print("\nN = Next Image")
            print("Q = Close Image Mode")

            command = input("Enter your choice (N/Q): ").strip().lower()

            cv2.destroyAllWindows()

            if command == "n":
                continue

            elif command == "q":
                print("\nImage Mode Closed!")
                break

            else:
                print("Invalid choice! Please enter N or Q.")

        cv2.destroyAllWindows()
        print("\nReturning to Main Menu...")
        continue

    # EXIT OPTION
    elif choice == "3":

        print("\nThank you for using AI Face Detection System!")
        print("Program Exited Successfully.")
        cv2.destroyAllWindows()
        break

    else:
        print("\nInvalid Choice! Please select 1, 2, or 3.")

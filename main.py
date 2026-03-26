import cv2
import os

def load_model():
    """
    Load Haar Cascade model for face detection
    """
    model_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("Haar Cascade file not found!")

    face_cascade = cv2.CascadeClassifier(model_path)
    return face_cascade


def start_webcam_detection(face_cascade):
    """
    Start real-time face detection using webcam
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return

    print("Press 'q' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show face count
        cv2.putText(frame, f'Faces Detected: {len(faces)}',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)

        cv2.imshow("Face Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_from_image(face_cascade, image_path):
    """
    Detect faces from a single image
    """
    if not os.path.exists(image_path):
        print("Image file not found!")
        return

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)

    print(f"Faces detected: {len(faces)}")

    cv2.imshow("Image Face Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    print("==== Face Detection System ====")
    print("1. Webcam Detection")
    print("2. Image Detection")

    choice = input("Enter your choice (1/2): ")

    face_cascade = load_model()

    if choice == '1':
        start_webcam_detection(face_cascade)

    elif choice == '2':
        image_path = input("Enter image path: ")
        detect_from_image(face_cascade, image_path)

    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()

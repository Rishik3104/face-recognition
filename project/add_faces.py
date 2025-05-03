from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import time

# Ensure the necessary directories 
if not os.path.exists('data/'):
    os.makedirs('data/')

# Load existing data if available
if 'names.pkl' in os.listdir('data/') and 'faces_data.pkl' in os.listdir('data/'):
    with open('data/names.pkl', 'rb') as f:
        LABELS = pickle.load(f)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
else:
    LABELS = []
    FACES = np.empty((0, 2500))  # 50x50 flattened images

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
if len(LABELS) > 0:
    knn.fit(FACES, LABELS)

# Ask if user wants to delete an existing entry
delete_choice = input("Do you want to delete an existing face entry? (yes/no): ").strip().lower()
if delete_choice == 'yes':
    username_to_delete = input("Enter the username to delete: ").strip()
    if username_to_delete in LABELS:
        # Filter out entries of the specified username
        indices_to_keep = [i for i, label in enumerate(LABELS) if label != username_to_delete]
        LABELS = [LABELS[i] for i in indices_to_keep]
        FACES = FACES[indices_to_keep]

        # Save updated data
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(LABELS, f)
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(FACES, f)

        print(f"Face data for '{username_to_delete}' has been deleted.")
    else:
        print(f"No data found for username '{username_to_delete}'.")

# Proceed with face registration if not deleting or if deletion is complete
add_choice = input("Do you want to add a new face entry? (yes/no): ").strip().lower()
if add_choice == 'yes':
    # Start video capture for face registration
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    username = input("Enter the username for the new face: ").strip()
    
    # Check if username already exists
    if username in LABELS:
        print("Username already exists. Skipping capture.")
    else:
        count = 0
        collected_faces = []

        while count < 100:
            ret, frame = video.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                crop_image = frame[y:y+h, x:x+w]
                resized_image = cv2.resize(crop_image, (50, 50)).flatten()
                collected_faces.append(resized_image)
                count += 1
                cv2.putText(frame, f"Capturing {count}/100", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow("Capturing Faces", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

        # Save new face data if 100 images were captured
        if count == 100:
            LABELS.extend([username] * 100)
            FACES = np.append(FACES, np.array(collected_faces), axis=0)

            # Save updated data
            with open('data/names.pkl', 'wb') as f:
                pickle.dump(LABELS, f)
            with open('data/faces_data.pkl', 'wb') as f:
                pickle.dump(FACES, f)

            print(f"Face data for '{username}' has been added.")
        else:
            print("Face capture was interrupted.")

else:
    print("No new face data was added.")




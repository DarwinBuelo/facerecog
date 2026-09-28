import os
import face_recognition
import numpy as np

encoding_file = "biden_encoding.npy"

# 1. Check if the encoding is already saved
if os.path.exists(encoding_file):
    print("Loading encoding from file...")
    biden_encoding = np.load(encoding_file) # load encoding from .npy file
else:
    print("Calculating encoding from image...")
    known_image = face_recognition.load_image_file("img/test.jpg")
    biden_encoding = face_recognition.face_encodings(known_image)[0]
    
    # Save it for future use
    np.save(encoding_file, biden_encoding)

# Example using the loaded/calculated encoding
unknown_image = face_recognition.load_image_file("img/test.jpg")
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
print(biden_encoding)
print("==================================")
print(unknown_encoding)
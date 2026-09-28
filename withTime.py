import os
import time
import face_recognition
import numpy as np

encoding_file = "biden_encoding.npy"
image_path = "img/test.jpg"

# Option A: Benchmark loading from file vs calculating from scratch
if os.path.exists(encoding_file):
    # Benchmark loading from file
    start_time = time.perf_counter()
    biden_encoding = np.load(encoding_file)
    load_time = time.perf_counter() - start_time
    print(f"[{load_time:.6f}s] Loaded encoding from disk (.npy)")
    
    # Optional: Benchmark calculation just to show the difference statistics
    start_time = time.perf_counter()
    known_image = face_recognition.load_image_file(image_path)
    temp_encoding = face_recognition.face_encodings(known_image)[0]
    calc_time = time.perf_counter() - start_time
    print(f"[{calc_time:.6f}s] Calculated encoding from image")
    
    # Calculate time saved
    time_saved = calc_time - load_time
    speedup = calc_time / load_time if load_time > 0 else 0
    print(f"---")
    print(f"⚡ Time saved: {time_saved:.6f} seconds ({speedup:.1f}x faster!)")

else:
    print("Calculating encoding from image for the first time...")
    start_time = time.perf_counter()
    known_image = face_recognition.load_image_file(image_path)
    biden_encoding = face_recognition.face_encodings(known_image)[0]
    calc_time = time.perf_counter() - start_time
    print(f"[{calc_time:.6f}s] Calculated encoding from image")
    
    # Save it for future use
    np.save(encoding_file, biden_encoding)
    print(f"Saved encoding to {encoding_file}")

print("==================================")

# Process unknown image
unknown_image = face_recognition.load_image_file(image_path)
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

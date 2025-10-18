import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import easyocr

# --- Load models ---
model_placa = YOLO("models/placa.pt")       # License plate detector
model_texto = YOLO("models/textoplaca.pt")  # Text region detector

# --- Initialize EasyOCR ---
reader = easyocr.Reader(["en"]) 

# --- Load image ---
image_path = "images/carro polo (2).jpeg"
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Detect license plate ---
results_placa = model_placa(img_rgb)
annotated_img = results_placa[0].plot()

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Detected License Plate(s)")
plt.show()

boxes = results_placa[0].boxes
if len(boxes) == 0:
    print("No license plate detected.")
else:
    # Take the most confident plate
    best_box = boxes[torch.argmax(boxes.conf)]
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    cropped_plate = img_rgb[y1:y2, x1:x2]

    plt.figure(figsize=(6, 4))
    plt.imshow(cropped_plate)
    plt.axis("off")
    plt.title("Cropped License Plate")
    plt.show()

    # --- Detect text region in the plate ---
    results_texto = model_texto(cropped_plate)
    text_boxes = results_texto[0].boxes

    if len(text_boxes) == 0:
        print("No text region detected on the plate.")
    else:
        # Take the most confident text region
        best_text_box = text_boxes[torch.argmax(text_boxes.conf)]
        x1t, y1t, x2t, y2t = map(int, best_text_box.xyxy[0])
        cropped_text = cropped_plate[y1t:y2t, x1t:x2t]

        plt.figure(figsize=(6, 2))
        plt.imshow(cropped_text)
        plt.axis("off")
        plt.title("Cropped Text Region")
        plt.show()

        # --- Run OCR on cropped text ---
        ocr_result = reader.readtext(cropped_text, detail=0)
        plate_text = "".join(ocr_result).replace(" ", "")

        print(f"\n🧾 Detected Plate Text: {plate_text}\n")

        # --- Optional: draw the recognized text on the plate image ---
        annotated_plate = cropped_plate.copy()
        cv2.putText(annotated_plate, plate_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        plt.figure(figsize=(6, 4))
        plt.imshow(annotated_plate)
        plt.axis("off")
        plt.title("Plate with OCR Text")
        plt.show()

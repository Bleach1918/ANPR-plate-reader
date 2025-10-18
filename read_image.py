import os, re, cv2, torch, easyocr, numpy as np, matplotlib.pyplot as plt
from ultralytics import YOLO

class App:
    def __init__(self):
        self.model_placa = YOLO("models/placa.pt")      
        self.model_texto = YOLO("models/textoplaca.pt")
        self.reader = easyocr.Reader(["en", "pt"]) 

    def read_plate(self, image):
        img = cv2.imread(image)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results_placa = self.model_placa(img_rgb)
        annotated_img = results_placa[0].plot()
        
        boxes = results_placa[0].boxes
        
        if len(boxes) == 0:
            print("No license plate detected")
        else:
            best_box = boxes[torch.argmax(boxes.conf)]
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            cropped_plate = img_rgb[y1:y2, x1:x2]
            
            plt.figure(figsize=(6, 4))
            plt.imshow(cropped_plate)
            plt.axis("off")
            plt.title("Cropped License Plate")
            plt.show()

            results_texto = self.model_texto(cropped_plate)
            text_boxes = results_texto[0].boxes
            
            if len(text_boxes) == 0:
                print("No text region detected on the plate.")
            else:
                best_text_box = text_boxes[torch.argmax(text_boxes.conf)]
                x1t, y1t, x2t, y2t = map(int, best_text_box.xyxy[0])
                cropped_text = cropped_plate[y1t:y2t, x1t:x2t]

                ocr_result = self.reader.readtext(cropped_text, detail=0)
                plate_text = "".join(ocr_result).replace(" ", "")

            if plate_text:
                return plate_text
            else:
                print("plate_text is None")
                return ""

    def fix_plate(self, plate):
        letter_to_digit = {"O":"0","Q":"0","I":"1","L":"1","Z":"2","S":"5","B":"8","G":"6"}
        digit_to_letter = {"0":"O","1":"I","2":"Z","5":"S","8":"B"} 
        corrected = ""
        
        if plate[4].isalpha():
            # Mercosul: AAA1A23
            for i, c in enumerate(plate):
                if i <= 2 or i == 4:  # letter positions
                    corrected += c if c.isalpha() else digit_to_letter.get(c, "X")
                else:  # number positions
                    corrected += c if c.isdigit() else letter_to_digit.get(c, "0")
        elif plate[4].isdigit():
            # Old: AAA1234
            for i, c in enumerate(plate):
                if i <= 2:  # letter positions
                    corrected += c if c.isalpha() else digit_to_letter.get(c, "X")
                else:  # number positions
                    corrected += c if c.isdigit() else letter_to_digit.get(c, "0")
        else:
            corrected = plate

        corrected = corrected[:7].ljust(7, "X")
        return corrected
        
if __name__ == "__main__":
    app = App()
    plates = []
    for filename in os.listdir("images"):
        print(filename)
        try:
            plate = app.read_plate("images/" + filename)
            plates.append(app.fix_plate(plate))
        except:
            pass
            
    print(plates)
        
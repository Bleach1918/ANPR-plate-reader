import os, cv2, torch, easyocr, threading
import tkinter as tk, numpy as np, matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image, ImageTk
from tkinter import Label, Frame, Button, Entry

class WebcamApp:
    def __init__(self, root):
        
        self.model_placa = YOLO("models/placa.pt")
        self.model_texto = YOLO("models/textoplaca.pt")
        self.reader = easyocr.Reader(["en", "pt"])
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 768)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 432)
        
        self.root = root
        self.root.title("License Plate Detector")
        self.root.geometry("1152x648")
        self.root.resizable(False, False)
        
        self.video_label = Label(self.root)
        self.video_label.place(x=10, y=10)
        
        self.read_button = Button(self.root, text="Ler Placa", command=self.read_plate, width=15, height=3)
        self.read_button.place(x=788, y=10)
        
        self.fix_button = Button(self.root, text="Arrumar Placa", command=self.fix_plate, width=15, height=3)
        self.fix_button.place(x=928, y=10)
        
        self.placa_label = Label(self.root, text="Placa")
        self.placa_label.place(x=788, y=80)
        
        self.placa_entry = Entry(self.root, width=30)
        self.placa_entry.place(x=788, y=100)
        
        self.frame = None
        self.running = True
        self.plate_text = ""
        self.cropped_plate_img = None
        self.last_plate_box = None
        
        threading.Thread(target=self.capture_frames, daemon=True).start()
        self.update_frame()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def capture_frames(self):
        while self.running:
            ret, frame= self.cap.read()
            if not ret:
                continue
            
            small_frame = cv2.resize(frame, (320, 240))
            results = self.model_placa(small_frame, verbose=False)
            
            scale_x = frame.shape[1] / 320
            scale_y = frame.shape[0] / 240
            
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)

                    # Draw bounding box on main frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            self.frame = frame
            self.results = results
    
    def update_frame(self):
        if self.frame is not None:
            img_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
        if self.cropped_plate_img is not None:
            plate_rgb = cv2.cvtColor(self.cropped_plate_img, cv2.COLOR_BGR2RGB)
            img_plate = Image.fromarray(plate_rgb)
            img_plate = img_plate.resize((300, 100)) 
            imgtk_plate = ImageTk.PhotoImage(image=img_plate)
            self.plate_label.imgtk = imgtk_plate
            self.plate_label.configure(image=imgtk_plate)
        
        self.root.after(10, self.update_frame)
        
    def read_plate(self):
        results = self.results
        if self.frame is None:
            print("no frame available")
            return
        
        if self.results is None:
            print("no plate available")
            return
        
        img = self.frame
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        scale_x = img_rgb.shape[1] / 320
        scale_y = img_rgb.shape[0] / 240
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

        cropped_plate = img_rgb[y1:y2, x1:x2]
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
            self.placa_entry.delete(0, tk.END)
            self.placa_entry.insert(0, plate_text)
            print(plate_text)
        else:
            self.placa_entry.delete(0, tk.END)
            self.placa_entry.insert(0, "Erro ao ler placa")

    def fix_plate(self):
        plate = self.placa_entry.get()
        
        if plate is None:
            return
        
        
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
        print(corrected)
        
        self.placa_entry.delete(0, tk.END)
        self.placa_entry.insert(0, corrected)
        
    def on_closing(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root)
    root.mainloop()
import tkinter as tk
from tkinter import Label
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import time

class WebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam License Plate Detector")
        self.root.geometry("800x600")
        self.root.resizable(False, False)

        # Open webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # YOLO model for plates (use GPU if available)
        self.model_placa = YOLO("models/placa.pt")  # add .cuda() if you have GPU

        # Label to display video frames
        self.label = Label(root)
        self.label.pack()

        # Threading variables
        self.frame = None
        self.results = None
        self.running = True

        # Start frame capture thread
        threading.Thread(target=self.capture_frames, daemon=True).start()

        # Start GUI update loop
        self.update_frame()

        # Close app properly on exit
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def capture_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Resize for faster YOLO inference
                small_frame = cv2.resize(frame, (320, 240))
                # Run YOLO every frame (or add skip logic if needed)
                results = self.model_placa(small_frame, verbose=False)
                
                # Scale boxes to original frame size
                scale_x = frame.shape[1] / 320
                scale_y = frame.shape[0] / 240
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Scale to original frame size
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)
                        # Draw rectangle on frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "Plate", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                self.frame = frame  # Save frame for GUI display
            else:
                time.sleep(0.01)  # small delay if frame not captured

    def update_frame(self):
        if self.frame is not None:
            # Convert BGR → RGB for Tkinter
            img_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        # Update every 10ms
        self.root.after(10, self.update_frame)

    def on_closing(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root)
    root.mainloop()

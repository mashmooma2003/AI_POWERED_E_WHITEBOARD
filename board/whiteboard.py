import cv2
import mediapipe as mp
import numpy as np
import os
import pytesseract

# --- 1. INITIALIZATION ---
# Point to the actual engine in Program Files
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
global full_session_text
full_session_text = ""
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75, 
    min_tracking_confidence=0.7) 

mp_drawing = mp.solutions.drawing_utils
tipIds = [4, 8, 12, 16, 20]
# --- Autocorrect Dictionary ---
COMMON_WORDS = {
    "1S": "IS", "I5": "IS", "LS": "IS",
    "VAAS": "WAS", "W4S": "WAS",
    "4T": "AT", "A7": "AT",
    "7O": "TO", "T0": "TO",
    "0F": "OF", "OF": "OF",
    "1N": "IN", "IN": "IN"
}
# --- Thickness & Color Definitions ---
mode = "DRAW"  
text_buffer = "" 
THICKNESS_OPTIONS = {"SMALL": 10, "MEDIUM": 25, "LARGE": 50}
brushThickness = THICKNESS_OPTIONS["MEDIUM"] 
headerColor = (200, 200, 200)
drawColor = (255, 0, 255) 

color_palette = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 0, 0) 
]

current_stroke_points = [] 

# --- 2. PROCESSING FUNCTIONS ---
def fingersUp(hand_landmarks):
    fingers = []
    if not hand_landmarks: return fingers
    # Thumb
    if hand_landmarks.landmark[tipIds[0]].x > hand_landmarks.landmark[tipIds[0] - 1].x: 
        fingers.append(1)
    else: fingers.append(0)
    # 4 Fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tipIds[id]].y < hand_landmarks.landmark[tipIds[id] - 2].y:
            fingers.append(1)
        else: fingers.append(0)
    return fingers
from fpdf import FPDF
import time

def export_to_pdf(text_content):
    # Create the folder if it doesn't exist
    folder_name = "Saved_Notes"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"📁 Created directory: {folder_name}")

    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Class Notes", ln=True, align='C')
    pdf.ln(10)
    
    # Date/Time
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Date: {time.ctime()}", ln=True, align='L')
    pdf.ln(5)
    
    # Content
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=text_content)
    
    # Save inside the folder
    filename = os.path.join(folder_name, f"Lesson_{int(time.time())}.pdf")
    pdf.output(filename)
    print(f"✅ PDF Saved to: {filename}")
def get_corrected_shape(points):
    if len(points) < 15: return None
    cnt = np.array(points).reshape((-1, 1, 2)).astype(np.int32)
    peri = cv2.arcLength(cnt, True)
    
    # --- 1. Circle Detection ---
    area = cv2.contourArea(cnt)
    if peri > 0:
        circularity = 4 * np.pi * (area / (peri * peri))
        if circularity > 0.75:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            return "CIRCLE", (int(x), int(y), int(radius))

    # --- 2. Polygon Detection (Triangle/Square) ---
    # epsilon is 3% of the perimeter. If it's too high, it ignores corners.
    # If it's too low, it sees every small shake as a new corner.
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
    
    num_corners = len(approx)
    
    if num_corners == 3:
        return "TRIANGLE", approx
    elif num_corners == 4:
        return "SQUARE", approx
    
    return "POLYGON", approx

# --- 3. UI SETUP ---
def setup_ui(w, h):
    imgHeader = np.zeros((100, w, 3), np.uint8) 
    cv2.rectangle(imgHeader, (0, 0), (w, 100), headerColor, cv2.FILLED) 
    num_items = len(color_palette) + len(THICKNESS_OPTIONS) + 2 
    spacing = w // num_items 
    current_x = 0 

    for i, color in enumerate(color_palette):
        cv2.rectangle(imgHeader, (current_x + 5, 5), (current_x + spacing - 5, 95), color, cv2.FILLED)
        if color == (0, 0, 0): 
             cv2.putText(imgHeader, "ERASE", (current_x + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        current_x += spacing

    for size_label in THICKNESS_OPTIONS.keys():
        cv2.rectangle(imgHeader, (current_x + 5, 5), (current_x + spacing - 5, 95), (150, 150, 150), cv2.FILLED) 
        cv2.putText(imgHeader, size_label, (current_x + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.circle(imgHeader, (current_x + spacing // 2, 65), THICKNESS_OPTIONS[size_label] // 2, (255, 255, 255), cv2.FILLED) 
        current_x += spacing

    cv2.rectangle(imgHeader, (current_x + 5, 5), (current_x + spacing - 5, 95), (255, 255, 0), cv2.FILLED)
    cv2.putText(imgHeader, "TEXT", (current_x + 20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    current_x += spacing

    cv2.rectangle(imgHeader, (current_x + 5, 5), (w - 5, 95), (50, 50, 50), cv2.FILLED)
    cv2.putText(imgHeader, "CLEAR", (current_x + 25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return imgHeader, spacing




def draw_header_ui(img, active_idx=-1):
    w = img.shape[1]
    spacing = w // 10 # Adjusted spacing to fit all buttons
    cv2.rectangle(img, (0, 0), (w, 100), (200, 200, 200), cv2.FILLED)
    
    curr_x = 0
    # Colors (0-4)
    for i, color in enumerate(color_palette):
        cv2.rectangle(img, (curr_x + 5, 5), (curr_x + spacing - 5, 95), color, cv2.FILLED)
        if active_idx == i:
            cv2.rectangle(img, (curr_x + 5, 5), (curr_x + spacing - 5, 95), (255, 255, 255), 5)
        curr_x += spacing
        
    # Thickness (5-7)
    for i, label in enumerate(THICKNESS_OPTIONS):
        idx = i + 5
        cv2.rectangle(img, (curr_x + 5, 5), (curr_x + spacing - 5, 95), (150, 150, 150), cv2.FILLED)
        if active_idx == idx:
            cv2.rectangle(img, (curr_x + 5, 5), (curr_x + spacing - 5, 95), (255, 255, 255), 5)
        cv2.putText(img, label[:1], (curr_x + 10, 55), 1, 1, (0,0,0), 2)
        curr_x += spacing
        
    # TEXT Button (8)
    cv2.rectangle(img, (curr_x + 5, 5), (curr_x + spacing - 5, 95), (255, 255, 0), cv2.FILLED)
    if active_idx == 8:
        cv2.rectangle(img, (curr_x + 5, 5), (curr_x + spacing - 5, 95), (255, 255, 255), 5)
    cv2.putText(img, "TXT", (curr_x + 10, 60), 1, 1, (0, 0, 0), 2)
    curr_x += spacing

    # CLEAR Button (9) - This puts it back in the header!
    cv2.rectangle(img, (curr_x + 5, 5), (w - 5, 95), (50, 50, 50), cv2.FILLED)
    if active_idx >= 9:
        cv2.rectangle(img, (curr_x + 5, 5), (w - 5, 95), (255, 255, 255), 5)
    cv2.putText(img, "CLR", (curr_x + 10, 60), 1, 1, (255, 255, 255), 2)
    
    
# --- 4. MAIN LOOP ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

xp, yp = 0, 0 
imgCanvas = None 
imgHeader = None
header_height = 100
active_btn = -1
while cap.isOpened():
    success, image = cap.read()
    if not success: break
    image = cv2.flip(image, 1) 
    
    if imgCanvas is None:
        h, w, c = image.shape
        imgCanvas = np.zeros((h, w, 3), np.uint8)
        imgHeader, palette_spacing = setup_ui(w, header_height)
    
    image[0:header_height, 0:w] = imgHeader
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            x_tip = int(hand_landmarks.landmark[8].x * w)
            y_tip = int(hand_landmarks.landmark[8].y * h)
            myFingers = fingersUp(hand_landmarks)
            if y_tip < 100:
             active_btn = x_tip // (w // 9) # This determines which button glows
            else:
             active_btn = -1 # No highlight if hand is down on canvas
            
            # --- GESTURE 1: SELECTION & PROCESSING (Two Fingers Up) ---
            if myFingers[1] == 1 and myFingers[2] == 1:
                xp, yp = 0, 0 
                
                if y_tip < header_height:
                    button_index = x_tip // palette_spacing
                    if button_index < len(color_palette):
                        drawColor = color_palette[button_index]
                        mode = "DRAW"
                    elif 5 <= button_index <= 7:
                        thickness_keys = list(THICKNESS_OPTIONS.keys())
                        brushThickness = THICKNESS_OPTIONS[thickness_keys[button_index - 5]]
                    elif button_index == 8:
                        mode = "TEXT"
                        drawColor = (255, 255, 0)
                    elif button_index >= 9: # CLEAR Button
                        # 1. Save the current text to the 'Session History'
                        if text_buffer.strip():
                            # We use a global variable to keep track of the whole class notes
                            if 'full_session_text' not in globals():
                               
                                full_session_text = ""
                            full_session_text += text_buffer + " "
                        
                        # 2. Clear ONLY the visual board
                        imgCanvas = np.zeros((h, w, 3), np.uint8)
                        text_buffer = ""
                        print("🧹 Board Cleared. Text saved to session memory.")
                elif len(current_stroke_points) > 15:
                    if mode == "TEXT":
                        pts = np.array(current_stroke_points)
                        tx, ty, tw, th = cv2.boundingRect(pts)
                        stroke_area = tw * th

                        # 1. NOISE FILTER (Ignores small dots/stray lines)
                       # Lowered area threshold to 300 so single digits/letters are caught
                        if stroke_area > 300: 
                            imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
                            _, imgThresh = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
                            
                            kernel = np.ones((5, 5), np.uint8)
                            imgThresh = cv2.dilate(imgThresh, kernel, iterations=1)
                            
                            try:
                                # PSM 6 is the best "all-rounder" for single chars AND short words
                                config = r'--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
                                
                                raw_word = pytesseract.image_to_string(imgThresh, config=config).strip().upper()
                                
                                if raw_word:
                                    # --- Comprehensive Autocorrect ---
                                    corrections = {
                                        "1S": "IS", "I5": "IS", "LS": "IS", "VAAS": "WAS", "W4S": "WAS", "4T": "AT",
                                        "S": "5", "I": "1", "L": "1", "O": "0", "Z": "2", "G": "6", "B": "8"
                                    }
                                    word = corrections.get(raw_word, raw_word)
                                    
                                    text_buffer = word
                                    
                                    # --- NOTE: Removed imgCanvas = np.zeros(...) ---
                                    # This allows the teacher to keep writing until they hit CLEAR.
                                    
                                    # Display the AI's guess in the corner or near the stroke
                                    cv2.putText(image, f"AI Saw: {word}", (tx, ty - 10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    print(f"✅ Recognized: {word}")
                                    
                            except Exception as e:
                                print(f"OCR Error: {e}")
                        else:
                            print("⚠️ Noise ignored (Too small)")
                    elif mode == "DRAW":
                        shape_data = get_corrected_shape(current_stroke_points)
                        if shape_data:
                            shape_type, pts = shape_data
                            print(f"🔹  {shape_type}") # PRINTS SHAPE NAME
                            imgCanvas = np.zeros((h, w, 3), np.uint8)
                            if shape_type == "CIRCLE":
                                cv2.circle(imgCanvas, (pts[0], pts[1]), pts[2], drawColor, brushThickness)
                            else:
                                cv2.drawContours(imgCanvas, [pts], -1, drawColor, brushThickness)
                    
                    current_stroke_points = [] 

            # --- GESTURE 2: DRAWING MODE (Index Only) ---
            # --- GESTURE 2: DRAWING MODE (Index Only) ---
            elif myFingers[1] == 1 and myFingers[2] == 0:
                cv2.circle(image, (x_tip, y_tip), 15, drawColor, cv2.FILLED)
                if xp == 0 and yp == 0: xp, yp = x_tip, y_tip
                
                # Check if we are using the Black color (Eraser)
                if drawColor == (0, 0, 0):
                    eraserThickness = 80  # Much thicker to "rub out" words easily
                    cv2.line(imgCanvas, (xp, yp), (x_tip, y_tip), (0, 0, 0), eraserThickness)
                else:
                    cv2.line(imgCanvas, (xp, yp), (x_tip, y_tip), drawColor, brushThickness)
                
                current_stroke_points.append((x_tip, y_tip))
                xp, yp = x_tip, y_tip

    # --- 5. MERGE CANVAS AND DISPLAY ---
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY_INV) 
    image = cv2.bitwise_and(image, cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR))
    image = cv2.bitwise_or(image, imgCanvas)
    
    # Bottom HUD
    cv2.rectangle(image, (0, h-50), (w, h), (30, 30, 30), -1)
    cv2.putText(image, f"MODE: {mode} | RECOGNIZED: {text_buffer}", (20, h-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # Press 'S' on keyboard to finish the lesson and export everything to ONE PDF
    draw_header_ui(image[0:100, 0:w], active_btn)

    cv2.namedWindow('AI-Smart Whiteboard', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
    'AI-Smart Whiteboard',
    cv2.WND_PROP_TOPMOST,
    1
    )
    cv2.imshow('AI-Smart Whiteboard', image)
    key = cv2.waitKey(1) & 0xFF

# Save PDF
    if key == ord('s'):
        final_notes = full_session_text + text_buffer
        export_to_pdf(final_notes)
        full_session_text = ""
        print("💾 Final PDF Exported!")

# Quit
    elif key == ord('q'):
     break

cap.release()
cv2.destroyAllWindows()
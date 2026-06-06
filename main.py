from tkinter import *
from PIL import Image, ImageDraw

from preprocess import preprocess_pil
from predict import predict

# ── Constants ──────────────────────────────────────────────────────────────
CANVAS_SIZE  = 380
BRUSH_SIZE   = 18
BG           = "#0e0e0e"
SURFACE      = "#1a1a1a"
ACCENT       = "#e8e8e8"
MUTED        = "#555555"
BTN_PREDICT  = "#f0f0f0"
BTN_CLEAR    = "#2a2a2a"
FONT_MONO    = ("Courier", 11)

# ── Window ──────────────────────────────────────────────────────────────────
win = Tk()
win.title("NeuroCipher")
win.geometry("440x620")
win.configure(bg=BG)
win.resizable(False, False)

# ── Header ───────────────────────────────────────────────────────────────────
Label(win, text="NEUROCIPHER", font=("Courier", 13, "bold"),
      bg=BG, fg=ACCENT).pack(pady=(22, 2))
Label(win, text="handwritten digit recognition", font=("Courier", 9),
      bg=BG, fg=MUTED).pack(pady=(0, 16))

# ── Drawing Canvas ────────────────────────────────────────────────────────────
canvas = Canvas(win, width=CANVAS_SIZE, height=CANVAS_SIZE,
                bg="white", highlightthickness=1, highlightbackground="#333")
canvas.pack()

pil_image  = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
pil_draw   = ImageDraw.Draw(pil_image)
last_x = last_y = None

def on_draw(event):
    global last_x, last_y
    x, y = event.x, event.y
    if last_x is not None:
        canvas.create_line(last_x, last_y, x, y,
                           fill="black", width=BRUSH_SIZE,
                           capstyle=ROUND, smooth=True)
        pil_draw.line((last_x, last_y, x, y), fill=0, width=BRUSH_SIZE)
    last_x, last_y = x, y

def on_release(event):
    global last_x, last_y
    last_x = last_y = None

canvas.bind("<B1-Motion>", on_draw)
canvas.bind("<ButtonRelease-1>", on_release)

# ── Buttons ────────────────────────────────────────────────────────────────────
def run_prediction():
    sample = preprocess_pil(pil_image)
    digit, confidence = predict(sample)
    result_var.set(f"{digit}")
    conf_var.set(f"{confidence * 100:.1f}%")

def clear_canvas():
    global pil_image, pil_draw
    canvas.delete("all")
    pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
    pil_draw  = ImageDraw.Draw(pil_image)
    result_var.set("—")
    conf_var.set("—")

btn_frame = Frame(win, bg=BG)
btn_frame.pack(pady=14)

Button(btn_frame, text="PREDICT", command=run_prediction,
       font=("Courier", 10, "bold"), bg=BTN_PREDICT, fg="#111",
       width=14, relief=FLAT, padx=8, pady=7).grid(row=0, column=0, padx=8)

Button(btn_frame, text="CLEAR", command=clear_canvas,
       font=("Courier", 10, "bold"), bg=BTN_CLEAR, fg=ACCENT,
       width=14, relief=FLAT, padx=8, pady=7).grid(row=0, column=1, padx=8)

# ── Result ─────────────────────────────────────────────────────────────────────
result_var = StringVar(value="—")
conf_var   = StringVar(value="—")

result_frame = Frame(win, bg=SURFACE, padx=24, pady=16)
result_frame.pack(fill=X, padx=30, pady=(4, 0))

Label(result_frame, text="DIGIT", font=("Courier", 8), bg=SURFACE, fg=MUTED).grid(row=0, column=0, padx=20)
Label(result_frame, text="CONFIDENCE", font=("Courier", 8), bg=SURFACE, fg=MUTED).grid(row=0, column=1, padx=20)

Label(result_frame, textvariable=result_var, font=("Courier", 32, "bold"),
      bg=SURFACE, fg=ACCENT).grid(row=1, column=0, padx=20)
Label(result_frame, textvariable=conf_var, font=("Courier", 22, "bold"),
      bg=SURFACE, fg=ACCENT).grid(row=1, column=1, padx=20)

# ── Run ─────────────────────────────────────────────────────────────────────────
win.mainloop()

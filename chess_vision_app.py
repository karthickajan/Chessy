"""
Chess Vision App — Python-only UI (replaces Angular)
Gradio-based: image upload + webcam capture → board detection → Lichess URL

Drop-in replacement for your Angular frontend.
Works locally and on Hugging Face Spaces (free hosting).

Install deps:
    pip install gradio ultralytics opencv-python-headless numpy

Run locally:
    python chess_vision_app.py

Deploy to HF Spaces:
    1. Create a new Space (Gradio SDK)
    2. Push this file + your model + requirements.txt
    3. Add model to Git LFS if >100MB
"""

import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import urllib.parse
import torch

# ─────────────────────────────────────────────
# TORCH SAFE GLOBALS (needed for custom model weights)
# ─────────────────────────────────────────────
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential, ModuleList
from ultralytics.nn.modules.conv import Conv, Concat
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU
from ultralytics.nn.modules.block import C2f, Bottleneck, SPPF, DFL
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.upsampling import Upsample
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from torch.nn.modules.loss import BCEWithLogitsLoss
from ultralytics.utils.tal import TaskAlignedAssigner

torch.serialization.add_safe_globals([
    DetectionModel, Sequential, ModuleList, Conv, Concat, Conv2d,
    BatchNorm2d, SiLU, C2f, Bottleneck, SPPF, MaxPool2d, Upsample,
    Detect, DFL, IterableSimpleNamespace, v8DetectionLoss,
    BCEWithLogitsLoss, TaskAlignedAssigner, BboxLoss,
])

# ─────────────────────────────────────────────
# CONFIG — update paths to match your project
# ─────────────────────────────────────────────
BOARD_MODEL_PATH = "chess2.pt"  # Board detection model
PIECE_MODEL_PATH = "best.pt"   # Piece classification model
CONF_THRESHOLD = 0.5           # Detection confidence threshold

# Map YOLO class indices → FEN piece letters (matches your best.pt)
CLASS_TO_FEN = {
    0: 'b', 1: 'b', 2: 'k', 3: 'k', 4: 'n', 5: 'n',
    6: 'board',  # ignored
    7: 'p', 8: 'p', 9: 'q', 10: 'q', 11: 'r', 12: 'r',
    13: 'eb', 14: 'ew',  # empty squares
    15: 'B', 16: 'B', 17: 'K', 18: 'K', 19: 'N', 20: 'N',
    21: 'P', 22: 'P', 23: 'Q', 24: 'Q', 25: 'R', 26: 'R',
}

# ─────────────────────────────────────────────
# LOAD MODELS (once at startup)
# ─────────────────────────────────────────────
print("Loading YOLO models...")
board_model = YOLO(BOARD_MODEL_PATH)
piece_model = YOLO(PIECE_MODEL_PATH)
print("Models loaded.")


# ─────────────────────────────────────────────
# YOUR BOARD DETECTION LOGIC
# Replace the body of this function with your actual implementation
# ─────────────────────────────────────────────
def detect_board(image_bgr: np.ndarray):
    """
    Detect the chessboard using chess2.pt YOLO model.
    Returns the cropped board resized to 512×512, or None if not found.
    """
    from PIL import Image

    # Run board detection model
    results = board_model.predict(image_bgr, verbose=False)
    if not results[0].boxes or len(results[0].boxes) == 0:
        return None

    box = results[0].boxes[0]
    coord = box.xyxy[0].tolist()
    x1, y1, x2, y2 = int(coord[0]), int(coord[1]), int(coord[2]), int(coord[3])

    board = image_bgr[y1:y2, x1:x2]
    board = cv2.resize(board, (512, 512))
    return board, coord


# ─────────────────────────────────────────────
# SLICE BOARD INTO 64 TILES
# ─────────────────────────────────────────────
def slice_board(board_img: np.ndarray):
    """Split a 512×512 board image into 8×8 = 64 tiles."""
    size = board_img.shape[0]  # assume square
    cell = size // 8
    tiles = []
    for row in range(8):
        for col in range(8):
            tile = board_img[row*cell:(row+1)*cell, col*cell:(col+1)*cell]
            tiles.append(tile)
    return tiles  # list of 64 tiles, row-major (a8→h8, a7→h7, … a1→h1)


# ─────────────────────────────────────────────
# PREDICT EACH TILE
# ─────────────────────────────────────────────
def predict_tiles(tiles):
    """
    Run YOLO on all 64 tiles in one batch call. Returns a list of 64 FEN characters.
    Batch inference is 3–5x faster than looping one tile at a time.
    """
    results_list = piece_model(tiles, conf=CONF_THRESHOLD, verbose=False)

    fen_chars = []
    for result in results_list:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            fen_chars.append("1")  # empty square
        else:
            best_idx = int(boxes.conf.argmax())
            cls = int(boxes.cls[best_idx])
            piece = CLASS_TO_FEN.get(cls, "1")
            if piece in ("board", "eb", "ew"):
                fen_chars.append("1")
            else:
                fen_chars.append(piece)

    return fen_chars


# ─────────────────────────────────────────────
# BUILD FEN STRING
# ─────────────────────────────────────────────
def build_fen(fen_chars: list) -> str:
    """Convert list of 64 FEN chars into a valid FEN piece-placement string."""
    rows = []
    for rank in range(8):
        row_chars = fen_chars[rank*8:(rank+1)*8]
        row_fen = ""
        empty = 0
        for ch in row_chars:
            if ch == "1":
                empty += 1
            else:
                if empty:
                    row_fen += str(empty)
                    empty = 0
                row_fen += ch
        if empty:
            row_fen += str(empty)
        rows.append(row_fen)
    return "/".join(rows)


# ─────────────────────────────────────────────
# LICHESS URL BUILDER
# ─────────────────────────────────────────────
def build_lichess_url(fen: str, turn: str = "w") -> str:
    full_fen = f"{fen} {turn} - - 0 1"
    encoded = urllib.parse.quote(full_fen)
    return f"https://lichess.org/analysis/{encoded}"


# ─────────────────────────────────────────────
# ANNOTATE IMAGE FOR DISPLAY
# ─────────────────────────────────────────────
def draw_grid(board_img: np.ndarray, fen_chars: list) -> np.ndarray:
    """Draw the 8×8 grid and detected piece labels on the board image."""
    vis = board_img.copy()
    size = vis.shape[0]
    cell = size // 8
    font = cv2.FONT_HERSHEY_SIMPLEX

    for row in range(8):
        for col in range(8):
            x1, y1 = col * cell, row * cell
            x2, y2 = x1 + cell, y1 + cell
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

            ch = fen_chars[row * 8 + col]
            if ch != "1":
                color = (255, 255, 255) if ch.isupper() else (0, 0, 0)
                cv2.putText(vis, ch, (x1 + cell//4, y1 + 3*cell//4),
                            font, 0.6, color, 2)
    return vis


# ─────────────────────────────────────────────
# MAIN PIPELINE (called by Gradio)
# ─────────────────────────────────────────────
def analyze_board(image: np.ndarray, player_turn: str):
    """
    Full pipeline: image → board crop → 64 tiles → YOLO → FEN → Lichess URL.
    Returns: annotated board image, FEN string, Lichess URL, status message.
    """
    if image is None:
        return None, "", "", "⚠️ No image provided."

    # Gradio gives RGB; convert to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 1. Detect board
    result = detect_board(image_bgr)
    if result is None:
        return None, "", "", "❌ Could not detect a chessboard. Try a clearer image."
    board, coord = result

    # 2. Slice into 64 tiles
    tiles = slice_board(board)

    # 3. Predict each tile
    fen_chars = predict_tiles(tiles)

    # 4. Build FEN
    turn = "w" if player_turn == "White to move" else "b"
    fen = build_fen(fen_chars)
    lichess_url = build_lichess_url(fen, turn)

    # 5. Annotate board for display (convert back to RGB for Gradio)
    annotated = draw_grid(board, fen_chars)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    status = f"✅ Board detected! {fen_chars.count('1')} empty squares, {64 - fen_chars.count('1')} pieces found."
    return annotated_rgb, fen, lichess_url, status


# ─────────────────────────────────────────────
# GRADIO UI  (replaces Angular entirely)
# ─────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 900px; margin: auto; }
#lichess-url textarea { font-family: monospace; font-size: 13px; color: #4e9eff; }
"""

with gr.Blocks(
    title="♟ Chess Vision — Board Analyzer",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=CSS,
) as demo:

    gr.Markdown(
        """
        # ♟ Chess Vision
        ### Analyze any chess position from a photo or your webcam
        Upload an image or take a photo — the app detects the board,
        identifies every piece, and opens the position in **Lichess Analysis**.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📷 Upload image or use webcam",
                sources=["upload", "webcam", "clipboard"],
                type="numpy",
                height=320,
            )
            turn_radio = gr.Radio(
                choices=["White to move", "Black to move"],
                value="White to move",
                label="Whose turn is it?",
            )
            analyze_btn = gr.Button("🔍 Analyze Board", variant="primary", size="lg")

        with gr.Column(scale=1):
            board_output = gr.Image(label="Detected Board", height=320)
            status_box = gr.Textbox(label="Status", interactive=False, lines=1)

    with gr.Row():
        fen_box = gr.Textbox(
            label="FEN String",
            interactive=False,
            elem_id="lichess-url",
        )
        lichess_box = gr.Textbox(
            label="Lichess Analysis URL (copy & open in browser)",
            interactive=False,
            elem_id="lichess-url",
            lines=2,
        )

    analyze_btn.click(
        fn=analyze_board,
        inputs=[image_input, turn_radio],
        outputs=[board_output, fen_box, lichess_box, status_box],
    )

    gr.Markdown(
        """
        ---
        **Tips:**
        - Works best with a flat-on view of the board, good lighting
        - Physical boards work better than screen photos
        - If detection fails, crop tighter to the board before uploading
        """
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # Remove this line for local-only access
        share=False,              # Set True for a temporary public link (no hosting needed)
    )

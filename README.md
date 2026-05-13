# ♟ Chessy — Chess Board Analyzer

> Take a photo of any chess board → get the position in Lichess Analysis instantly.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What it does

1. You upload a photo (or use your webcam)
2. `chess2.pt` detects and crops the board
3. The board is sliced into 64 tiles
4. `best.pt` classifies each tile as a piece or empty
5. A FEN string is built and opened in **Lichess Analysis**

---

## Demo

| Upload photo | Detected position | Lichess Analysis |
|---|---|---|
| 📷 Phone or webcam | ✅ Piece labels overlaid | 🔗 Opens in-app |

---

## Speed Modes

| Mode | Time (CPU) | Accuracy | Use when |
|---|---|---|---|
| ⚡ Fast | ~20–24s | Good | Quick preview |
| ⚖️ Balanced | ~27–32s | Better | Daily use |
| 🎯 Accurate | ~40–55s | Best | Position matters |

All three modes use the same `best.pt` model — only inference resolution and confidence threshold differ. No GPU required.

---

## Run Locally

```bash
# 1. Clone
git clone https://github.com/karthickajan/Chessy.git
cd Chessy

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python chess_vision_app.py
```

Open `http://localhost:7860` in your browser.

---

## File Structure

```
Chessy/
├── chess_vision_app.py   # Main app (Gradio UI + pipeline)
├── chess2.pt             # Board detection model (YOLO)
├── best.pt               # Piece classification model (YOLO)
├── requirements.txt
└── README.md
```

---

## Models

| Model | Purpose | Classes |
|---|---|---|
| `chess2.pt` | Detects the board bounding box in a photo | 1 (board) |
| `best.pt` | Classifies each 64×64 tile | 27 (12 piece types × color + empty + board) |

`best.pt` was trained on per-tile crops, not full board images. It expects individual square inputs.

---

## Tips for best results

- Flat overhead shot (directly above the board)
- Even lighting — avoid strong shadows across squares
- Physical boards work better than screen photos
- Crop tightly to the board if detection fails
- Use **Accurate** mode when pieces are misidentified

---

## Tech Stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — object detection
- [Gradio](https://gradio.app) — Python web UI
- [OpenCV](https://opencv.org) — image processing
- [Lichess](https://lichess.org) — free chess analysis

---

## Roadmap

- [ ] Castling rights input
- [ ] Board orientation flip
- [ ] En passant square input
- [ ] Stockfish best move suggestion
- [ ] Mobile-optimised layout

---

## License

MIT — free to use, modify, and distribute.
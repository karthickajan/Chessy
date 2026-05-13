# Deploy Chessy to Hugging Face Spaces (Free Hosting)

This guide deploys your Chess Vision app to **Hugging Face Spaces** — a free Python hosting platform perfect for Gradio apps.

**Total time: ~25 minutes**

---

## Prerequisites

1. **Hugging Face Account** — Create one at https://huggingface.co (free)
2. **Git LFS installed** — Required for model files (chess2.pt, best.pt)

### Install Git LFS

**macOS:**
```bash
brew install git-lfs
git lfs install
```

**Windows (PowerShell as Admin):**
```bash
winget install Git.Git
git lfs install
```

**Ubuntu/Debian:**
```bash
sudo apt install git-lfs
git lfs install
```

Verify installation:
```bash
git lfs version
```

---

## Step 1: Create a Space on Hugging Face

1. Go to: https://huggingface.co/new-space
2. Fill in these fields:
   - **Owner:** Your username (e.g., `karthickajan`)
   - **Space name:** `chessy` (or any name you want)
   - **SDK:** Select **Gradio**
   - **Visibility:** Public (free tier)
3. Click **"Create Space"**

You'll be redirected to your new Space. Copy the HTTPS URL shown (e.g., `https://huggingface.co/spaces/karthickajan/chessy`).

---

## Step 2: Clone the Space Repository

```bash
git clone https://huggingface.co/spaces/karthickajan/chessy
cd chessy
```

---

## Step 3: Copy Project Files

Copy all necessary files from your Chessy project:

```bash
# Copy main app and config
cp /path/to/Chessy/chess_vision_app.py .
cp /path/to/Chessy/requirements.txt .
cp /path/to/Chessy/README.md .

# Copy model files (Git LFS will track these automatically)
cp /path/to/Chessy/chess2.pt .
cp /path/to/Chessy/best.pt .
```

**Replace `/path/to/Chessy/` with your actual project path.**

---

## Step 4: Update README.md with Hugging Face Header

Open `README.md` and **add these lines at the very top** (before any other content):

```yaml
---
title: Chessy Chess Board Analyzer
emoji: ♟
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: chess_vision_app.py
pinned: false
---
```

**Important:** The `---` must be the first characters of the file (no blank lines before).

Example:
```markdown
---
title: Chessy Chess Board Analyzer
emoji: ♟
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: chess_vision_app.py
pinned: false
---

# ♟ Chess Vision
### Photo → Board Analysis → Lichess

[rest of your README...]
```

---

## Step 5: Verify Git LFS is Tracking Models

```bash
# Check that models will be tracked by Git LFS
git lfs status

# You should see output like:
# On branch main
# Objects to be pushed to origin:
#   chess2.pt (128 MB)
#   best.pt (128 MB)
```

If models aren't shown, manually track them:
```bash
git lfs track "*.pt"
git add .gitattributes
```

---

## Step 6: Commit and Push

```bash
# Stage all files
git add .

# Commit
git commit -m "Initial deploy: Chessy chess board analyzer with Gradio UI"

# Push to Hugging Face
git push
```

**First push takes 2-5 minutes (uploading models).**

---

## Step 7: Wait for Build

Your Space will automatically build. Watch progress at:
- **Build Logs:** https://huggingface.co/spaces/karthickajan/chessy/logs
- **App will be live at:** https://huggingface.co/spaces/karthickajan/chessy

⏳ **Wait 3–5 minutes for the build to complete.**

Once green ✅, your app is live!

---

## Testing Your Deployment

1. Open https://huggingface.co/spaces/karthickajan/chessy
2. Upload a chess board image
3. Select **Balanced (~28s)** mode
4. Click **"🔍 Analyze Board"**
5. The Lichess board should load in the iframe

---

## Troubleshooting

### ❌ "Model files not uploading"
- Verify Git LFS: `git lfs version`
- Check: `git lfs status`
- Reinstall: `git lfs install --force`

### ❌ "Build fails with 'module not found'"
- Check `requirements.txt` has all packages
- Hugging Face runs: `pip install -r requirements.txt`
- Common missing: Try adding `opencv-python` (not headless version)

### ❌ "Space times out during inference"
- Free tier has 16GB RAM (your 2x128MB models fit fine)
- First inference takes longer (cold start)
- Space sleeps after 48h inactivity — wakes on next visit
- If consistently slow: Consider Hugging Face **Pro ($9/mo)** for persistent GPU

### ❌ "'No space left on device' error"
- Free tier: 50GB total storage
- Your models + app code: ~300MB (plenty of room)
- If needed, upgrade to Pro

### ❌ "Cannot find chess2.pt or best.pt at runtime"
- Models must be in same directory as `chess_vision_app.py`
- Check file structure in Space:
  ```
  chessy/
    ├── chess_vision_app.py
    ├── chess2.pt
    ├── best.pt
    ├── requirements.txt
    └── README.md
  ```

---

## Free Tier Specifications

| Resource | Limit |
|----------|-------|
| **CPU** | 2 vCPUs |
| **RAM** | 16 GB |
| **Storage** | 50 GB |
| **GPU** | ❌ (None — CPU only) |
| **Inference Time** | ~20-36s per board (normal) |
| **Sleep** | After 48h inactivity |
| **Cost** | **FREE** ✅ |

---

## Optional: Upgrade to Pro

For **persistent GPU + no sleep:**
- Cost: **$9/month**
- Includes: 2x CPU, 16GB RAM, 1x GPU (T4), no sleeping
- Inference time: ~2-5s per board (10x faster!)

---

## Next Steps

1. ✅ Test your Space
2. ✅ Share the URL with friends: `https://huggingface.co/spaces/karthickajan/chessy`
3. ✅ Pin to your Hugging Face profile
4. ✅ (Optional) Add to a portfolio website

---

## Support

- **Hugging Face Docs:** https://huggingface.co/docs/hub/spaces
- **Gradio Docs:** https://www.gradio.app/
- **Git LFS Help:** https://git-lfs.com/

Good luck! 🚀

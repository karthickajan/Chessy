---<div align="center">

title: Chess Vision Board Analyzer  <p>

emoji: ♟    <a href="https://yolovision.ultralytics.com/" target="_blank">

colorFrom: blue      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-yolo-vision-2023.png" alt="YOLO Vision banner"></a>

colorTo: indigo  </p>

sdk: gradio

sdk_version: "4.44.0"[中文](https://docs.ultralytics.com/zh/) | [한국어](https://docs.ultralytics.com/ko/) | [日本語](https://docs.ultralytics.com/ja/) | [Русский](https://docs.ultralytics.com/ru/) | [Deutsch](https://docs.ultralytics.com/de/) | [Français](https://docs.ultralytics.com/fr/) | [Español](https://docs.ultralytics.com/es/) | [Português](https://docs.ultralytics.com/pt/) | [हिन्दी](https://docs.ultralytics.com/hi/) | [العربية](https://docs.ultralytics.com/ar/) <br>

app_file: chess_vision_app.py

pinned: false<div>

---    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml/badge.svg" alt="Ultralytics CI"></a>

    <a href="https://codecov.io/github/ultralytics/ultralytics"><img src="https://codecov.io/github/ultralytics/ultralytics/branch/main/graph/badge.svg?token=HHW7IIVFVY" alt="Ultralytics Code Coverage"></a>

# ♟ Chess Vision — Board Analyzer    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv8 Citation"></a>

    <a href="https://hub.docker.com/r/ultralytics/ultralytics"><img src="https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker" alt="Docker Pulls"></a>

Upload a chess board photo → YOLO detects every piece → get a **Lichess Analysis** link.    <a href="https://ultralytics.com/discord"><img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a>

    <br>

Pure Python. No Angular, no backend separation. Works locally and on [Hugging Face Spaces](https://huggingface.co/spaces) (free).    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a>

    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

## How It Works    <a href="https://www.kaggle.com/ultralytics/yolov8"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>

</div>

1. **Board Detection** — `chess2.pt` (YOLOv8) locates the chessboard in the image<br>

2. **Grid Slicing** — The board is divided into 8×8 = 64 tiles

3. **Piece Classification** — `best.pt` (YOLOv8) classifies each tile (27 classes: 12 piece types × black/white + board + empty)[Ultralytics](https://ultralytics.com) [YOLOv8](https://github.com/ultralytics/ultralytics) is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. YOLOv8 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection and tracking, instance segmentation, image classification and pose estimation tasks.

4. **FEN Generation** — Detections are converted to a FEN string

5. **Lichess URL** — Opens the position in Lichess Analysis BoardWe hope that the resources here will help you get the most out of YOLOv8. Please browse the YOLOv8 <a href="https://docs.ultralytics.com/">Docs</a> for details, raise an issue on <a href="https://github.com/ultralytics/ultralytics/issues/new/choose">GitHub</a> for support, and join our <a href="https://ultralytics.com/discord">Discord</a> community for questions and discussions!



## Project StructureTo request an Enterprise License please complete the form at [Ultralytics Licensing](https://ultralytics.com/license).



```<img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png" alt="YOLOv8 performance plots"></a>

├── chess_vision_app.py   # Main app (Gradio UI + detection pipeline)

├── best.pt               # Piece classifier model (50 MB)<div align="center">

├── chess2.pt              # Board detector model (50 MB)  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="2%" alt="Ultralytics GitHub"></a>

├── requirements.txt       # Python dependencies  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">

└── README.md              # This file  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="2%" alt="Ultralytics LinkedIn"></a>

```  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">

  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="2%" alt="Ultralytics Twitter"></a>

## Run Locally  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">

  <a href="https://youtube.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="2%" alt="Ultralytics YouTube"></a>

```bash  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">

# 1. Install dependencies  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="2%" alt="Ultralytics TikTok"></a>

pip install -r requirements.txt  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">

  <a href="https://www.instagram.com/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="2%" alt="Ultralytics Instagram"></a>

# 2. Launch  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">

python chess_vision_app.py  <a href="https://ultralytics.com/discord"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="2%" alt="Ultralytics Discord"></a>

</div>

# 3. Open http://localhost:7860</div>

```

## <div align="center">Documentation</div>

## Deploy to Hugging Face Spaces (Free)

See below for a quickstart installation and usage example, and see the [YOLOv8 Docs](https://docs.ultralytics.com) for full documentation on training, validation, prediction and deployment.

```bash

# 1. Install Git LFS (models are >50 MB)<details open>

git lfs install<summary>Install</summary>



# 2. Create a Space at https://huggingface.co/new-space (SDK: Gradio)Pip install the ultralytics package including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).



# 3. Clone, copy files, push[![PyPI version](https://badge.fury.io/py/ultralytics.svg)](https://badge.fury.io/py/ultralytics) [![Downloads](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics)

git clone https://huggingface.co/spaces/YOUR_USERNAME/chess-vision

cd chess-vision```bash

cp /path/to/{chess_vision_app.py,best.pt,chess2.pt,requirements.txt,README.md} .pip install ultralytics

git add .```

git commit -m "Initial deploy"

git pushFor alternative installation methods including [Conda](https://anaconda.org/conda-forge/ultralytics), [Docker](https://hub.docker.com/r/ultralytics/ultralytics), and Git, please refer to the [Quickstart Guide](https://docs.ultralytics.com/quickstart).

```

</details>

Wait ~3 min for the Space to build. Open: `https://huggingface.co/spaces/YOUR_USERNAME/chess-vision`

<details open>

### Free Tier Limits (HF Spaces)<summary>Usage</summary>



| Resource | Free Tier |#### CLI

|----------|-----------|

| CPU      | 2 vCPUs   |YOLOv8 may be used directly in the Command Line Interface (CLI) with a `yolo` command:

| RAM      | 16 GB     |

| Storage  | 50 GB     |```bash

| GPU      | ❌ (paid only) |yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'

| Sleep    | After 48h inactivity, wakes on visit |```



**Is GPU needed?** No. YOLO on 64 tiles via CPU takes 2–8 seconds on HF's free tier. Acceptable for an analysis tool.`yolo` can be used for a variety of tasks and modes and accepts additional arguments, i.e. `imgsz=640`. See the YOLOv8 [CLI Docs](https://docs.ultralytics.com/usage/cli) for examples.



## Tips#### Python



- Works best with a flat, overhead view of the boardYOLOv8 may also be used directly in a Python environment, and accepts the same [arguments](https://docs.ultralytics.com/usage/cfg/) as in the CLI example above:

- Good lighting improves accuracy

- Physical boards work better than screenshots```python

- Crop tightly to the board if detection failsfrom ultralytics import YOLO



## License# Load a model

model = YOLO("yolov8n.yaml")  # build a new model from scratch

MITmodel = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)


# Use the model
model.train(data="coco128.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
```

See YOLOv8 [Python Docs](https://docs.ultralytics.com/usage/python) for more examples.

</details>

## <div align="center">Models</div>

YOLOv8 [Detect](https://docs.ultralytics.com/tasks/detect), [Segment](https://docs.ultralytics.com/tasks/segment) and [Pose](https://docs.ultralytics.com/tasks/pose) models pretrained on the [COCO](https://docs.ultralytics.com/datasets/detect/coco) dataset are available here, as well as YOLOv8 [Classify](https://docs.ultralytics.com/tasks/classify) models pretrained on the [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet) dataset. [Track](https://docs.ultralytics.com/modes/track) mode is available for all Detect, Segment and Pose models.

<img width="1024" src="https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png" alt="Ultralytics YOLO supported tasks">

All [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

<details open><summary>Detection (COCO)</summary>

See [Detection Docs](https://docs.ultralytics.com/tasks/detect/) for usage examples with these models trained on [COCO](https://docs.ultralytics.com/datasets/detect/coco/), which include 80 pre-trained classes.

| Model                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](http://cocodataset.org) dataset. <br>Reproduce by `yolo val detect data=coco.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val detect data=coco.yaml batch=1 device=0|cpu`

</details>

<details><summary>Detection (Open Image V7)</summary>

See [Detection Docs](https://docs.ultralytics.com/tasks/detect/) for usage examples with these models trained on [Open Image V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/), which include 600 pre-trained classes.

| Model                                                                                     | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ----------------------------------------------------------------------------------------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-oiv7.pt) | 640                   | 18.4                 | 142.4                          | 1.21                                | 3.5                | 10.5              |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-oiv7.pt) | 640                   | 27.7                 | 183.1                          | 1.40                                | 11.4               | 29.7              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-oiv7.pt) | 640                   | 33.6                 | 408.5                          | 2.26                                | 26.2               | 80.6              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-oiv7.pt) | 640                   | 34.9                 | 596.9                          | 2.43                                | 44.1               | 167.4             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-oiv7.pt) | 640                   | 36.3                 | 860.6                          | 3.56                                | 68.7               | 260.6             |

- **mAP<sup>val</sup>** values are for single-model single-scale on [Open Image V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/) dataset. <br>Reproduce by `yolo val detect data=open-images-v7.yaml device=0`
- **Speed** averaged over Open Image V7 val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val detect data=open-images-v7.yaml batch=1 device=0|cpu`

</details>

<details><summary>Segmentation (COCO)</summary>

See [Segmentation Docs](https://docs.ultralytics.com/tasks/segment/) for usage examples with these models trained on [COCO-Seg](https://docs.ultralytics.com/datasets/segment/coco/), which include 80 pre-trained classes.

| Model                                                                                        | size<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640                   | 36.7                 | 30.5                  | 96.1                           | 1.21                                | 3.4                | 12.6              |
| [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640                   | 44.6                 | 36.8                  | 155.7                          | 1.47                                | 11.8               | 42.6              |
| [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640                   | 49.9                 | 40.8                  | 317.0                          | 2.18                                | 27.3               | 110.2             |
| [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640                   | 52.3                 | 42.6                  | 572.4                          | 2.79                                | 46.0               | 220.5             |
| [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640                   | 53.4                 | 43.4                  | 712.1                          | 4.02                                | 71.8               | 344.1             |

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](http://cocodataset.org) dataset. <br>Reproduce by `yolo val segment data=coco-seg.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val segment data=coco-seg.yaml batch=1 device=0|cpu`

</details>

<details><summary>Pose (COCO)</summary>

See [Pose Docs](https://docs.ultralytics.com/tasks/pose/) for usage examples with these models trained on [COCO-Pose](https://docs.ultralytics.com/datasets/pose/coco/), which include 1 pre-trained class, person.

| Model                                                                                                | size<br><sup>(pixels) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------------------------------------------------------------------------------------------------- | --------------------- | --------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640                   | 50.4                  | 80.1               | 131.8                          | 1.18                                | 3.3                | 9.2               |
| [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640                   | 60.0                  | 86.2               | 233.2                          | 1.42                                | 11.6               | 30.2              |
| [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640                   | 65.0                  | 88.8               | 456.3                          | 2.00                                | 26.4               | 81.0              |
| [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640                   | 67.6                  | 90.0               | 784.5                          | 2.59                                | 44.4               | 168.6             |
| [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640                   | 69.2                  | 90.2               | 1607.1                         | 3.73                                | 69.4               | 263.2             |
| [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280                  | 71.6                  | 91.2               | 4088.7                         | 10.04                               | 99.1               | 1066.4            |

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO Keypoints val2017](http://cocodataset.org) dataset. <br>Reproduce by `yolo val pose data=coco-pose.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val pose data=coco-pose.yaml batch=1 device=0|cpu`

</details>

<details><summary>Classification (ImageNet)</summary>

See [Classification Docs](https://docs.ultralytics.com/tasks/classify/) for usage examples with these models trained on [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/), which include 1000 pretrained classes.

| Model                                                                                        | size<br><sup>(pixels) | acc<br><sup>top1 | acc<br><sup>top5 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) at 640 |
| -------------------------------------------------------------------------------------------- | --------------------- | ---------------- | ---------------- | ------------------------------ | ----------------------------------- | ------------------ | ------------------------ |
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224                   | 66.6             | 87.0             | 12.9                           | 0.31                                | 2.7                | 4.3                      |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224                   | 72.3             | 91.1             | 23.4                           | 0.35                                | 6.4                | 13.5                     |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224                   | 76.4             | 93.2             | 85.4                           | 0.62                                | 17.0               | 42.7                     |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224                   | 78.0             | 94.1             | 163.0                          | 0.87                                | 37.5               | 99.7                     |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224                   | 78.4             | 94.3             | 232.0                          | 1.01                                | 57.4               | 154.8                    |

- **acc** values are model accuracies on the [ImageNet](https://www.image-net.org/) dataset validation set. <br>Reproduce by `yolo val classify data=path/to/ImageNet device=0`
- **Speed** averaged over ImageNet val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu`

</details>

## <div align="center">Integrations</div>

Our key integrations with leading AI platforms extend the functionality of Ultralytics' offerings, enhancing tasks like dataset labeling, training, visualization, and model management. Discover how Ultralytics, in collaboration with [Roboflow](https://roboflow.com/?ref=ultralytics), ClearML, [Comet](https://bit.ly/yolov8-readme-comet), Neural Magic and [OpenVINO](https://docs.ultralytics.com/integrations/openvino), can optimize your AI workflow.

<br>
<a href="https://bit.ly/ultralytics_hub" target="_blank">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics active learning integrations"></a>
<br>
<br>

<div align="center">
  <a href="https://roboflow.com/?ref=ultralytics">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-roboflow.png" width="10%" alt="Roboflow logo"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="space">
  <a href="https://cutt.ly/yolov5-readme-clearml">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-clearml.png" width="10%" alt="ClearML logo"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="space">
  <a href="https://bit.ly/yolov8-readme-comet">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-comet.png" width="10%" alt="Comet ML logo"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="space">
  <a href="https://bit.ly/yolov5-neuralmagic">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-neuralmagic.png" width="10%" alt="NeuralMagic logo"></a>
</div>

|                                                           Roboflow                                                           |                                                            ClearML ⭐ NEW                                                            |                                                                        Comet ⭐ NEW                                                                        |                                           Neural Magic ⭐ NEW                                           |
| :--------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: |
| Label and export your custom datasets directly to YOLOv8 for training with [Roboflow](https://roboflow.com/?ref=ultralytics) | Automatically track, visualize and even remotely train YOLOv8 using [ClearML](https://cutt.ly/yolov5-readme-clearml) (open-source!) | Free forever, [Comet](https://bit.ly/yolov8-readme-comet) lets you save YOLOv8 models, resume training, and interactively visualize and debug predictions | Run YOLOv8 inference up to 6x faster with [Neural Magic DeepSparse](https://bit.ly/yolov5-neuralmagic) |

## <div align="center">Ultralytics HUB</div>

Experience seamless AI with [Ultralytics HUB](https://bit.ly/ultralytics_hub) ⭐, the all-in-one solution for data visualization, YOLOv5 and YOLOv8 🚀 model training and deployment, without any coding. Transform images into actionable insights and bring your AI visions to life with ease using our cutting-edge platform and user-friendly [Ultralytics App](https://ultralytics.com/app_install). Start your journey for **Free** now!

<a href="https://bit.ly/ultralytics_hub" target="_blank">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/ultralytics-hub.png" alt="Ultralytics HUB preview image"></a>

## <div align="center">Contribute</div>

We love your input! YOLOv5 and YOLOv8 would not be possible without help from our community. Please see our [Contributing Guide](https://docs.ultralytics.com/help/contributing) to get started, and fill out our [Survey](https://ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey) to send us feedback on your experience. Thank you 🙏 to all our contributors!

<!-- SVG image from https://opencollective.com/ultralytics/contributors.svg?width=990 -->

<a href="https://github.com/ultralytics/yolov5/graphs/contributors">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/image-contributors.png" alt="Ultralytics open-source contributors"></a>

## <div align="center">License</div>

Ultralytics offers two licensing options to accommodate diverse use cases:

- **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/licenses/) open-source license is ideal for students and enthusiasts, promoting open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for more details.
- **Enterprise License**: Designed for commercial use, this license permits seamless integration of Ultralytics software and AI models into commercial goods and services, bypassing the open-source requirements of AGPL-3.0. If your scenario involves embedding our solutions into a commercial offering, reach out through [Ultralytics Licensing](https://ultralytics.com/license).

## <div align="center">Contact</div>

For Ultralytics bug reports and feature requests please visit [GitHub Issues](https://github.com/ultralytics/ultralytics/issues), and join our [Discord](https://ultralytics.com/discord) community for questions and discussions!

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.instagram.com/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="3%" alt="Ultralytics Instagram"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/discord"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>

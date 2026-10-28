# PoseFrame Studio

**A local browser-based tool for annotating human poses on ordered video frames.**

Import a folder of timestamp-named frames, navigate with the left/right arrow keys, use manual COCO-17 skeletons or YOLO26 pose predictions, drag joints to correct them, and write annotations to one automatically updated CSV.

**All frame data stays on your computer.** Ultralytics only needs internet access when downloading model files or installing packages; its inference runs locally.

## 1. Start the website

### Windows (PowerShell or Command Prompt)

Unzip the project, open a terminal in the **poseframe_studio** folder, then run:

```powershell
.\run.bat
```

You can also double-click `run.bat` in File Explorer.

### Ubuntu / WSL / macOS

```bash
cd poseframe_studio
./run.sh
```

The first run installs Flask into a `.venv` virtual environment. Python **3.10 or newer** and an internet connection for this initial installation are required. The browser should open automatically at:

**http://127.0.0.1:8765**

If it doesn't, open that address yourself. Keep the terminal window running; press **Ctrl+C** to stop the app. Your annotations remain saved.

If you use **WSL**, use `./run.sh` in the WSL shell. To import by folder path, use a *Linux-readable path*, e.g. `/mnt/c/Users/yourname/Videos/frames`. The folder-upload option also works in a browser running on Windows.

## 2. Install YOLO26 for automatic labeling (optional)

Manual pose annotation works immediately with the basic installation. To enable auto-labeling, **stop the server**, then run:

Windows:
```powershell
.\install_yolo.bat
```

Ubuntu / WSL / macOS:
```bash
./install_yolo.sh
```

These commands install `ultralytics` (and its PyTorch dependencies) in the same virtual environment. Start the app again with `run.bat` or `./run.sh`.

The default model is **`yolo26x-pose.pt`** (YOLO26x pose), exactly as requested. Ultralytics automatically downloads its pretrained weights on the **first prediction**. This is a very large model and is slow on CPU. The sidebar also offers `yolo26n-pose.pt` (much lighter), `s`, `m`, and `l` when you want faster inference. Models download separately the first time you use each.

A CUDA-compatible NVIDIA GPU may accelerate inference if a suitable GPU-supported PyTorch environment is installed. The default installer does not guarantee CUDA support; see the official PyTorch installation guide if you need a specific GPU/CUDA build.

## 3. Import a directory of frames

Choose either method in **Open folder**:

- **Upload folder:** Browse to a folder from your web browser. Images are copied into `data/frames/` in small batches. This works across Windows/browser/WSL setups and does not need a server filesystem path.
- **Use local folder path:** Paste a full path on the same computer where the Python server is running. Files are read *in place*, with **no copying**. Recommended for large video-frame datasets.

Supported images: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`. Only files directly inside the chosen folder are indexed; nested folders are not traversed.

Frame order: timestamp filenames such as `2026-10-21-14-07-03-000.jpg`, `2026-10-21-14-07-03-100.jpg`, `2026-10-21-14-07-03-200.jpg` sort by their timestamp fields, including any trailing fractional-second/suffix component. Other frame names use natural numeric order (`frame_2` before `frame_10`).

A folder of **sample_frames** is included; import it if you'd like to test the controls immediately.

> **Important:** Opening a *different* folder starts a new working project. Export your old CSV first. Reopening the *same* local-path folder preserves its saved annotations.

## 4. Label poses

1. Use the **← / → arrow keys**, Previous/Next buttons, timeline slider, or click a filename in the left panel.
2. Click **Manual skeleton** to create an editable, front-facing 17-joint outline. Drag individual joint markers to the correct body locations. Click **+ Add** for additional people.
3. Or click **Auto-label with YOLO** to detect all people in the current frame. The result is editable exactly like manual skeletons. Running auto-label again replaces that frame's people, with a confirmation.

   To process the dataset automatically, click **Auto-label unlabeled frames** in the left sidebar. It uses the selected model and confidence threshold, skips labeled/reviewed frames, and saves coordinates and CSV after each frame. Keep the browser tab and server open. **Stop after current frame** preserves completed work; starting again processes the remaining unlabeled frames (including earlier unsuccessful predictions). When the run finishes, the **Unlabeled** section lists frames with no confident keypoints or prediction errors for manual review. Hover a frame to read its full error. Repeated errors or a missing dependency stop the run.
4. Select a joint from the inspector to adjust numeric **X/Y** coordinates; or click **Place point by clicking image** and click its location. For an unlabeled point, set its visibility to `Visible` or `Hidden` before moving it.
5. Set the joint's COCO visibility status as needed: `Visible (2)`, `Hidden/occluded (1)`, or `Missing/unlabeled (0)`.
6. **Ctrl+S** (Cmd+S on macOS) saves the current frame. Moving to another frame also saves any edits automatically. The first YOLO prediction is saved immediately; later corrections save on navigation or Ctrl+S.
7. Click **Export CSV** to download the current file anytime.

**Tip:** Ctrl+Z undoes edits and Ctrl+Shift+Z redoes them. Ctrl+mouse-wheel over the image zooms in/out. The small +/- zoom controls are available below the image.

## 5. Output CSV and COCO convention

The live file is:

```text
poseframe_studio/data/annotations.csv
```

It is rewritten **atomically whenever a frame is saved**. It is also initialized when a folder is imported, so every image appears in the CSV. The export button downloads a copy named `coco_keypoints.csv`.

**CSV structure:** Each person gets **one row per frame**. Columns are:

```text
filename,frame_index,annotation_status,person_id,source,num_keypoints,image_width,image_height,
nose_x,nose_y,nose_v,left_eye_x,left_eye_y,left_eye_v,...,right_ankle_x,right_ankle_y,right_ankle_v
```

That is 8 metadata columns + 17×3 = **59 columns**. All X/Y coordinates use **pixel positions in the original full-resolution image**, not normalized values or positions on the screen.

CSV keypoint triplets match [COCO person-keypoint ordering](https://datatorch.io/resources/coco-keypoints). Their fixed order is:

```text
0 nose                  9 left_wrist
1 left_eye              10 right_wrist
2 right_eye             11 left_hip
3 left_ear              12 right_hip
4 right_ear             13 left_knee
5 left_shoulder         14 right_knee
6 right_shoulder        15 left_ankle
7 left_elbow            16 right_ankle
8 right_elbow
```

COCO visibility `v` semantics:

- `0` → not labeled; saved as `x=0, y=0, v=0`.
- `1` → labeled but occluded; coordinates retained.
- `2` → labeled and visible; coordinates retained.

`num_keypoints` counts values with `v > 0`. Left/right are from **the person's perspective**. COCO includes a **19-edge skeleton**, which is what the overlay draws.

Unannotated frames and explicitly reviewed frames with zero people each get **one metadata-only row** with `person_id` blank, `num_keypoints=0`, and keypoint columns all zero. Their `annotation_status` is `unlabeled` or `reviewed_empty` respectively. Filter to `annotation_status=labeled` when consuming the CSV as person annotations. The `source` column is `manual` or `yolo26`.

**Note:** COCO's original distribution format is a JSON object (`images`, `annotations`, `categories`). This app produces the *requested flat CSV adaptation of COCO coordinates*, not the original COCO JSON format. YOLO's raw joint-confidence score is *not* equivalent to COCO visibility; automatic predictions mark confident keypoints as `2` and weak ones as `0`. Set `1` yourself for occluded joints.

## Files and privacy

```text
poseframe_studio/
├── app.py                  Flask backend, local pose inference, CSV saving
├── index.html              Browser UI
├── static/
│   ├── app.js              Interactive label editor
│   └── style.css           UI styling
├── sample_frames/          Synthetic demo images
├── run.bat / run.sh        Launch scripts
├── install_yolo.bat / .sh Optional YOLO installation
├── requirements.txt
├── requirements-yolo.txt
├── tests/
└── data/                   Created automatically by the app
    ├── project.json        Restore previous project and label state
    ├── annotations.csv     ONE CSV for all frames and people
    └── frames/             Copies created by folder uploads only
```

The local server listens on **127.0.0.1** by default, so other computers cannot access it. The application requires no account, remote service, or cloud upload. Avoid editing `data/project.json` or `data/annotations.csv` manually while the app is running; use the UI, export a copy, or stop the server first.

### Troubleshooting

- **Auto label says Ultralytics not installed:** stop server, run the matching `install_yolo` script, restart it.
- **Windows DLL error (`c10.dll`, WinError 126/127/1114):** install or repair the [Microsoft Visual C++ x64 runtime](https://aka.ms/vc14/vc_redist.x64.exe), restart Windows if requested, then restart `run.bat`. If it persists, run `install_yolo.bat` again; it now checks that PyTorch and Ultralytics actually load.
- **Model takes a long time:** first use downloads weights and initializes PyTorch. `yolo26x-pose.pt` is much heavier than `yolo26n-pose.pt`; try the `n` model on CPU.
- **Web browser does not open:** manually navigate to `http://127.0.0.1:8765`.
- **Path input fails in WSL:** use a WSL path like `/mnt/c/Users/...`, not `C:\Users\...`.
- **Nothing to label:** use `sample_frames` to verify manual mode before running YOLO.
- **For thousands of frames:** prefer **Use local folder path** so images do not need to be copied into the workspace.
- **CSV locked by Excel on Windows:** close Excel before saving; Windows may block an atomic replacement if the file is held open.

### Dependencies & licensing

Flask is licensed under BSD-3-Clause. Ultralytics software and model weights have license terms that may affect commercial distribution, notably AGPL-3.0 or commercial licensing; check [Ultralytics' licensing information](https://www.ultralytics.com/license) for your use case. No Ultralytics code or pretrained model is bundled inside this ZIP.

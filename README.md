# SWYNEX Task 2 - Model Integration: Graphic Design Style Classifier

A small Streamlit web app that integrates the trained Keras model from
[Task 1](https://github.com/maniram02112006/SWYNEX-AI-Graphic-Design-Classifier)
into a working prototype. Upload a poster or graphic design image and the app
predicts its design style.

**Internship:** SWYNEX Technologies - Task 2: Model or API Integration

## What it does

1. Accepts a JPG or PNG image (up to 10 MB).
2. Validates the file (real image, readable, at least 32 x 32 pixels).
3. Preprocesses it exactly like the Task 1 training script.
4. Loads the saved Keras model and predicts one of five styles.
5. Shows the predicted style, the model confidence, and the probability of every style.
6. Shows clear error messages for invalid files.

**Classes (same order as the Task 1 code):** cinematic, minimalist, typography, sports, vintage

## Model details (read from the Task 1 code and the saved model file)

| Item | Value |
|---|---|
| Architecture | MobileNetV2 (ImageNet weights, frozen) + GlobalAveragePooling2D + Dropout(0.2) + Dense(5, softmax) |
| Input | 160 x 160 x 3 RGB image, raw pixel values 0-255 |
| Preprocessing | MobileNetV2 scaling is built into the saved model, so the app does not divide by 255 |
| Output | 5 softmax probabilities, mapped to the class list above |
| Model file | `model/graphic_design_classifier.keras` (about 9.7 MB) |

## Project structure

```text
SWYNEX-Model-or-API-Integration/
|-- app.py                  Streamlit application
|-- requirements.txt        Python packages
|-- README.md               This file
|-- .gitignore
|-- model/
|   `-- graphic_design_classifier.keras   Trained model from Task 1
`-- examples/
    |-- cinematic_01.jpg, minimalist_02.jpg, sports_01.jpg, typography_01.jpg, vintage_01.jpg
    `-- example_outputs.md  Example inputs and the outputs the app produced
```

## How to run (Windows, VS Code terminal)

```powershell
# Create a virtual environment using Python 3.12 (TensorFlow compatible)
py -3.12 -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py
```

Your browser opens automatically at `http://localhost:8501`.

> [!NOTE]
> If PowerShell blocks script execution, run:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
> TensorFlow requires Python 3.10–3.12 on Windows. If Python 3.15 or newer is installed on your computer, ensure you specify `py -3.12` when creating the virtual environment.

## Example inputs and outputs

See [`examples/example_outputs.md`](examples/example_outputs.md). Try the five
images in the `examples/` folder (`cinematic_01.jpg`, `minimalist_02.jpg`, `sports_01.jpg`, `typography_01.jpg`, `vintage_01.jpg`).

## Live Test Verification

All modules have been verified and tested live:
1. **Model Loading:** Successfully loads `model/graphic_design_classifier.keras` (Keras 3 format, MobileNetV2 base).
2. **Preprocessing:** Tested nearest-neighbor resize to 160×160×3 with raw 0–255 pixel range (internal MobileNetV2 scaling intact).
3. **Inference & Mapping:** Tested on all 4 sample images, producing predicted classes matching expected design styles.
4. **Input Validation:** Tested corrupted images, images smaller than 32×32, and files exceeding 10 MB. All error messages handled gracefully.
5. **Streamlit App Server:** Tested startup on `http://localhost:8501`, verified HTTP 200 health check response.

## GitHub Submission Instructions

To publish this project to a new public GitHub repository:

1. Create a new empty public repository on GitHub named:
   `SWYNEX-Model-or-API-Integration`
   *(Do not initialize it with a README, license, or .gitignore since they already exist here)*

2. Open PowerShell in `D:\SWYNEX-Model-or-API-Integration` and run:
   ```powershell
   git init
   git add .
   git commit -m "Complete SWYNEX Task 2: Model Integration Streamlit application"
   git branch -M main
   git remote add origin https://github.com/maniram02112006/SWYNEX-Model-or-API-Integration.git
   git push -u origin main
   ```

## Limitations (please read)

- The Task 1 model was trained on **20 AI-generated illustrative sample images
  (4 per class)**. The dataset's own README says this is not enough for reliable
  training or an honest performance evaluation.
- **No accuracy figure is reported here**, because no proper held-out evaluation exists.
- The example images are from that same sample set, so correct predictions on
  them do **not** show how the model performs on new designs.
- The confidence is a softmax score. It is not a calibrated probability and can
  be high for wrong answers.
- The model always chooses one of the five styles, even for images that are not
  posters or that mix styles.

## Future improvements

- Collect a larger, properly licensed and labelled dataset and evaluate on a held-out test set.
- Add data augmentation and fine-tuning.
- Add an "uncertain / not a poster" option.

## Security

No API keys, passwords, or tokens are used or stored in this project. The model
runs locally.

# ASL Sign Language Recognition – Bi-GRU Model Based on Video

This project aims to automatically classify American Sign Language (ASL) words from video clips. MediaPipe from Google is used to extract body and hand joint points, and a Bi-GRU deep learning model is trained on these features.

## 📁 Technologies Used
- Python
- TensorFlow & Keras
- MediaPipe
- OpenCV
- NumPy, Pandas
- Matplotlib
- Tkinter (GUI)

## 📊 Dataset
- ASLLRP dataset (Rutgers University)
- 639 video samples for 51 different signs
- Split: 70% Training / 20% Validation / 10% Testing
📊 Additional statistics and performance results are available in [assets/Statistics.txt](./assets/Statistics.txt)


## ⚙️ Preprocessing
- Face landmarks and Z-axis removed
- Only shoulders, elbows, wrists, and 42 hand keypoints used
- Coordinates converted to relative (based on shoulders)
- Missing data filled using stack technique

## 🤖 AI Model – Bi-GRU
- Bi-GRU with two layers (128 and 64 units)
- Masking, Dropout, BatchNormalization applied
- Callbacks used: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Class weights used to handle imbalance

## 🔍 Results
|          Model           | Accuracy (Validation) |  Accuracy (Real Videos)    |
|--------------------------|-----------------------|----------------------------|
| RNN (absolute)           |           38%         |            -               |
| LSTM (absolute)          |           41%         |            -               |
| GRU (absolute)           |           44%         |      1/10 correct          |
| GRU (relative)           |           60%         |      3/10 correct          |
| **Bi-GRU (final model)** |        **65.5%**      |   **68.2%** (30/44 videos) |

## 📦 Requirements

Before running the project, install the required packages:

```bash
pip install -r requirements.txt
```

## 🧪 How to Run (GUI)

1. Launch the graphical interface:

```bash
python sign_language_gui.py
```

2. Inside the GUI:

- Click **Load Video** to choose a video file.
- Click **Classify Video** to run prediction.
- The predicted result will appear below.

## 🎬 Sample Videos

Some real ASL sign videos used for testing are included in the [samples](./samples) folder.

You can try them using the GUI:
1. Click "Load Video"
2. Select any of the sample videos
3. Click "Classify Video"


## ⚠️ Important Notes

- Some ASL signs have very similar hand movements, which may cause confusion during classification.
- Examples include:
  - `ANSWER` vs `DIRECT`
  - `BIG` vs `COUCH`
  - `ART` vs `CANCEL`

If the model predicts one of them correctly and the other incorrectly, it's likely due to their close visual similarity in sign movement.



> Make sure the model file (`GRU_model_rel_best.keras`) and label map (`label_map.json`) paths are correctly set in the script.
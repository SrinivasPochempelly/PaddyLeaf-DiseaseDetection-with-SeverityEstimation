# 🌾 Paddy Leaf Disease Detection with Severity Estimation

This project is a web-based application built using **Flask** and **TensorFlow/Keras**, designed to detect diseases in paddy (rice) leaf images. Along with disease classification, it also highlights the affected regions and estimates the severity of the infection.

---

## 📌 Features

- ✅ Paddy leaf verification using color and shape analysis.
- 🌿 Disease detection using a pretrained `EfficientNetB0` model.
- 🔬 Severity estimation based on the infected region in the leaf.
- 🖼️ Visual outputs: original image and disease-affected area.
- 🧠 Model trained on 10 paddy leaf disease classes including **Tungro, Blast, Brown Spot**, etc.
- 🌐 User-friendly web interface to upload and analyze leaf images.

---

## 🧪 Disease Classes

The model can classify the following diseases:
- Bacterial Leaf Blight
- Bacterial Leaf Streak
- Bacterial Panicle Blight
- Blast
- Brown Spot
- Dead Heart
- Downy Mildew
- Hispa
- Normal (Healthy Leaf)
- Tungro

---

## 📷 Sample Workflow

1. Upload a paddy leaf image.
2. App verifies it's a valid leaf based on green region & shape.
3. Disease is predicted using a deep learning model.
4. If diseased, severity is estimated by masking infected areas.
5. Results are displayed with:
   - Disease name
   - Severity level (Low / Medium / High)
   - Original image and affected area visualization

---

## 💻 Tech Stack

| Technology | Usage |
|-----------|-------|
| Python | Core backend logic |
| Flask | Web framework |
| OpenCV | Image preprocessing & masking |
| TensorFlow / Keras | Deep learning model |
| Bootstrap | Responsive frontend |
| HTML, CSS | Web page UI |

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/paddy-leaf-disease-detection.git
cd paddy-leaf-disease-detection

import cv2
import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
import glob

app = Flask(__name__)

# Configure upload and static image folders
UPLOAD_FOLDER = 'static/uploads'
STATIC_IMAGE_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_IMAGE_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Load the model (update this path to your local model file)
MODEL_PATH = 'efficientnetb0_paddy.h5'  # Replace with actual path, e.g., 'efficientnetb0_paddy.h5'
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"❗ Error loading model: {str(e)}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_leaf_image(img_path):
    """
    Check if the image contains a paddy leaf by detecting green areas and analyzing shape.
    Returns True if the image is likely a paddy leaf, False otherwise.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Define green range for leaves
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Calculate the percentage of green area
    green_pixels = cv2.countNonZero(green_mask)
    total_pixels = img.shape[0] * img.shape[1]
    green_percentage = (green_pixels / total_pixels) * 100

    # Lowered threshold: At least 5% of the image should be green
    green_threshold = 5.0
    if green_percentage < green_threshold:
        print(f"❌ Image is not a leaf. Green area: {green_percentage:.2f}% (threshold: {green_threshold}%)")
        return False
    print(f"✅ Green area check passed: {green_percentage:.2f}% (threshold: {green_threshold}%)")

    # Find contours in the green mask to identify leaf shapes
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"❌ No leaf contours detected. Green area: {green_percentage:.2f}%")
        return False
    print(f"✅ Contours detected: {len(contours)}")

    # Analyze the largest contour (likely the main leaf)
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    contour_area_percentage = (contour_area / total_pixels) * 100

    # Lowered threshold: Largest contour should be at least 2% of the image
    contour_area_threshold = 2.0
    if contour_area_percentage < contour_area_threshold:
        print(f"❌ No significant leaf contour detected. Contour area: {contour_area_percentage:.2f}% (threshold: {contour_area_threshold}%)")
        return False
    print(f"✅ Contour area check passed: {contour_area_percentage:.2f}% (threshold: {contour_area_threshold}%)")

    # Get the bounding rectangle of the largest contour to calculate aspect ratio
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h if h != 0 else 0
    # Swap if height is greater than width (for vertical leaves)
    if aspect_ratio < 1 and aspect_ratio != 0:
        aspect_ratio = 1 / aspect_ratio

    # Lowered threshold: Paddy leaves typically have an aspect ratio > 1.5
    aspect_ratio_threshold = 1.5
    # Fallback: Accept if aspect ratio is slightly below threshold but green and contour areas are significant
    if aspect_ratio < aspect_ratio_threshold:
        # return False
        # Fallback check: If green area > 10% and contour area > 5%, accept as paddy leaf
        if green_percentage > 10.0 and contour_area_percentage > 5.0:
            print(f"⚠️ Aspect ratio below threshold ({aspect_ratio:.2f} < {aspect_ratio_threshold}), but accepted due to significant green area ({green_percentage:.2f}%) and contour area ({contour_area_percentage:.2f}%)")
        else:
            print(f"❌ Image is not a paddy leaf. Aspect ratio: {aspect_ratio:.2f} (threshold: {aspect_ratio_threshold})")
            return False

    print(f"✅ Image contains a paddy leaf. Green area: {green_percentage:.2f}%, Contour area: {contour_area_percentage:.2f}%, Aspect ratio: {aspect_ratio:.2f}")
    return True

def predict_disease(img_path, model):
    class_names = ['bacterial_leaf_blight','bacterial_leaf_streak','bacterial_panicle_blight', 'blast','brown_spot','dead_heart','downy_mildew','hispa','normal','tungro']

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return class_names[predicted_class]

def apply_green_mask(img_path, is_normal=False):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Save original image
    cv2.imwrite(os.path.join(STATIC_IMAGE_FOLDER, "original_image.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if is_normal:
        return None, None  # Skip mask generation for normal images

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Green leaf mask
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Sky blue mask (exclude)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Black and dark brown mask (exclude)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 50, 50])  # Low value for dark colors
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)

    # Combine exclusion masks
    exclude_mask = cv2.bitwise_or(blue_mask, dark_mask)

    # Refine green mask by excluding sky blue and dark colors
    refined_green_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(exclude_mask))

    disease_affected_area = cv2.bitwise_and(img, img, mask=refined_green_mask)

    # Save disease-affected image
    cv2.imwrite(os.path.join(STATIC_IMAGE_FOLDER, "disease_affected_area.jpg"), cv2.cvtColor(disease_affected_area, cv2.COLOR_RGB2BGR))

    return disease_affected_area, refined_green_mask

def estimate_severity(img_path):
    leaf_img, mask = apply_green_mask(img_path)  # Refined mask excludes sky blue and dark colors

    hsv_leaf = cv2.cvtColor(leaf_img, cv2.COLOR_RGB2HSV)

    lower_disease = np.array([10, 50, 50])
    upper_disease = np.array([30, 255, 255])

    disease_mask = cv2.inRange(hsv_leaf, lower_disease, upper_disease)
    disease_mask = cv2.bitwise_and(disease_mask, disease_mask, mask=mask)

    disease_area = cv2.countNonZero(disease_mask) * 5
    total_leaf_area = cv2.countNonZero(mask)

    if total_leaf_area == 0:
        print("❗ No leaf area detected. Please check the image.")
        return None, None

    severity_percent = round((disease_area / total_leaf_area) * 100, 2)

    if severity_percent < 30:
        severity = "Low"
    elif severity_percent < 60:
        severity = "Medium"
    else:
        severity = "High"

    return severity, severity_percent

def process_leaf_image(img_path, model):
    # Step 1: Check if the image is a paddy leaf
    if not is_leaf_image(img_path):
        return {"Error": "Please upload an image of a paddy leaf."}

    # Step 2: Proceed with prediction if it's a paddy leaf
    disease = predict_disease(img_path, model)
    print(f"🟢 Predicted Disease: {disease}")

    if disease.lower() == 'normal':
        print("✅ The leaf is healthy. No severity estimation required.")
        apply_green_mask(img_path, is_normal=True)
        return {"Disease": disease, "Severity": "None", "Message": "The leaf is healthy! No disease detected."}
    else:
        print("⚠️ Disease detected. Estimating severity...")
        severity, percent = estimate_severity(img_path)
        if severity is None:
            return {"Disease": disease, "Severity": "Unknown", "SeverityPercent": 0}
        return {"Disease": disease, "Severity": severity, "SeverityPercent": percent}
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file selected")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")

        if file and allowed_file(file.filename):
            # Clean up old uploaded files and images
            for old_file in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*')):
                try:
                    os.remove(old_file)
                except:
                    pass
            for old_image in glob.glob(os.path.join(STATIC_IMAGE_FOLDER, '*.jpg')):
                try:
                    os.remove(old_image)
                except:
                    pass

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                if model is None:
                    return render_template('index.html', error="Model not loaded")

                result = process_leaf_image(file_path, model)

                if "Error" in result:
                    return render_template('index.html', error=result["Error"])

                result_data = {
                'disease': result['Disease'],
                'severity': result.get('Severity', 'Unknown'),
                'severity_percent': result.get('SeverityPercent', 0),
                'message': result.get('Message', ''),
                'original_image': '/static/original_image.jpg',
                'disease_affected_area': '/static/disease_affected_area.jpg' if result['Disease'].lower() != 'normal' else None,
                'green_mask': None  # No longer passing green mask
                }


                return render_template('index.html', result=result_data)

            except Exception as e:
                return render_template('index.html', error=f"Error processing image: {str(e)}")

        return render_template('index.html', error="Invalid file type. Please upload a .jpg, .jpeg, or .png file.")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

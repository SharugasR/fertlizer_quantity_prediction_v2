"""
Flask Web Application for Fertilizer Prediction
Author: AI Assistant
Date: 2025
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from fertilizer_prediction_model import FertilizerPredictionModel
import json
import logging
from werkzeug.utils import secure_filename
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize model
model = None
model_loaded = False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the trained model"""
    global model, model_loaded
    try:
        model = FertilizerPredictionModel()
        model.load_model('fertilizer_prediction_model')
        model_loaded = True
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model_loaded = False

def preprocess_image(image_path):
    """Preprocess uploaded image"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        img = cv2.resize(img, (224, 224))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def create_visualization(prediction, image_path):
    """Create visualization of prediction results"""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(img)
        axes[0].set_title('Uploaded Plant Image')
        axes[0].axis('off')
        
        # Fertilizer type probabilities
        fertilizer_types = list(prediction['all_type_probabilities'].keys())
        probabilities = list(prediction['all_type_probabilities'].values())
        
        bars = axes[1].bar(fertilizer_types, probabilities)
        axes[1].set_title('Fertilizer Type Probabilities')
        axes[1].set_ylabel('Probability')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Highlight the predicted type
        predicted_idx = fertilizer_types.index(prediction['fertilizer_type'])
        bars[predicted_idx].set_color('red')
        
        # Quantity gauge
        quantity = prediction['fertilizer_quantity']
        max_quantity = 500  # Maximum expected quantity
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = 1.0
        
        # Convert quantity to angle
        qty_normalized = min(quantity / max_quantity, 1.0)
        qty_angle = qty_normalized * np.pi
        
        # Plot gauge background
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        axes[2].plot(x, y, 'k-', linewidth=2)
        
        # Plot gauge needle
        axes[2].plot([0, r * np.cos(qty_angle)], [0, r * np.sin(qty_angle)], 'r-', linewidth=3)
        
        # Add labels
        axes[2].text(0, -0.2, f'{quantity:.1f} grams', fontsize=12, ha='center')
        axes[2].text(0, 0.1, 'Fertilizer Quantity', fontsize=10, ha='center')
        axes[2].set_xlim(-1.2, 1.2)
        axes[2].set_ylim(-0.3, 1.2)
        axes[2].axis('off')
        axes[2].set_title('Recommended Quantity')
        
        plt.tight_layout()
        
        # Save to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Convert to base64
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        return img_data
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        return None

def generate_recommendations(prediction):
    """Generate farmer-friendly recommendations"""
    fertilizer_type = prediction['fertilizer_type']
    quantity = prediction['fertilizer_quantity']
    confidence = prediction['confidence']
    
    recommendations = []
    
    # Base recommendation
    recommendations.append(f"Recommended fertilizer: {fertilizer_type}")
    recommendations.append(f"Quantity: {quantity:.1f} grams per kg of soil")
    
    # Confidence-based advice
    if confidence > 0.8:
        recommendations.append("High confidence prediction - proceed with recommendation")
    elif confidence > 0.6:
        recommendations.append("Moderate confidence - consider consulting an expert")
    else:
        recommendations.append("Low confidence - please consult an agricultural expert")
    
    # Fertilizer-specific advice
    fertilizer_advice = {
        'Urea': [
            "Apply Urea in split doses for better efficiency",
            "Water immediately after application to prevent nitrogen loss",
            "Avoid application during hot, dry weather"
        ],
        'DAP': [
            "DAP is excellent for root development",
            "Apply at planting time for best results",
            "Mix well with soil to avoid root burn"
        ],
        'Compost': [
            "Compost improves soil structure and water retention",
            "Apply 2-3 weeks before planting",
            "Can be used as mulch around plants"
        ],
        'NPK': [
            "NPK provides balanced nutrition",
            "Apply according to soil test recommendations",
            "Split application for better nutrient uptake"
        ],
        'Potash': [
            "Potash improves fruit quality and disease resistance",
            "Apply during flowering and fruiting stages",
            "Essential for root and stem strength"
        ],
        'Organic': [
            "Organic fertilizers improve soil health long-term",
            "Apply regularly for sustained benefits",
            "Compost with organic matter for better results"
        ]
    }
    
    if fertilizer_type in fertilizer_advice:
        recommendations.extend(fertilizer_advice[fertilizer_type])
    
    # Quantity-based advice
    if quantity > 300:
        recommendations.append("High quantity needed - consider soil testing")
        recommendations.append("Split application over multiple weeks")
    elif quantity < 100:
        recommendations.append("Low quantity needed - plant is relatively healthy")
        recommendations.append("Monitor plant growth and adjust as needed")
    
    return recommendations

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction endpoint"""
    if not model_loaded:
        flash('Model not loaded. Please contact administrator.', 'error')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(url_for('index'))
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        if file and allowed_file(file.filename):
            try:
                # Secure filename
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Make prediction
                prediction = model.predict_single_image(filepath)
                
                # Generate recommendations
                recommendations = generate_recommendations(prediction)
                
                # Create visualization
                visualization = create_visualization(prediction, filepath)
                
                # Convert image to base64 for display
                with open(filepath, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                
                # Clean up uploaded file
                os.remove(filepath)
                
                return render_template('result.html',
                                     prediction=prediction,
                                     recommendations=recommendations,
                                     image_data=img_data,
                                     visualization=visualization)
                
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                flash(f'Error processing image: {str(e)}', 'error')
                return redirect(url_for('index'))
        else:
            flash('Invalid file type. Please upload an image file.', 'error')
            return redirect(url_for('index'))
    
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save temporary file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            prediction = model.predict_single_image(filepath)
            
            # Clean up
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'prediction': prediction
            })
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/help')
def help():
    """Help page"""
    return render_template('help.html')

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    """Handle 404 error"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 error"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)

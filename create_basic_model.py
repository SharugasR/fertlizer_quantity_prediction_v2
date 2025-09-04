"""
Create a basic model for testing the web application
"""

import numpy as np
import os
from fertilizer_prediction_model import FertilizerPredictionModel

def create_basic_model():
    """Create a basic untrained model for testing"""
    print("Creating basic model for testing...")
    
    # Initialize model
    model = FertilizerPredictionModel(img_size=(224, 224), num_fertilizer_types=8)
    
    # Build model
    model.build_model()
    model.compile_model(learning_rate=0.001)
    
    # Create dummy data to fit the scalers
    dummy_quantities = np.random.uniform(50, 400, 100)
    model.quantity_scaler.fit(dummy_quantities.reshape(-1, 1))
    
    # Create dummy labels for label encoder
    dummy_labels = np.random.choice(model.fertilizer_types, 100)
    model.label_encoder.fit(dummy_labels)
    
    # Save model
    model.save_model('fertilizer_prediction_model')
    
    print("✅ Basic model created and saved!")
    print("Model files created:")
    print("- fertilizer_prediction_model.h5")
    print("- fertilizer_prediction_model_label_encoder.pkl")
    print("- fertilizer_prediction_model_quantity_scaler.pkl")
    print("- fertilizer_prediction_model_fertilizer_types.json")
    
    # Test prediction
    print("\nTesting prediction...")
    try:
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite('test_plant.jpg', dummy_image)
        
        prediction = model.predict_single_image('test_plant.jpg')
        print(f"✅ Prediction test successful!")
        print(f"   Fertilizer: {prediction['fertilizer_type']}")
        print(f"   Quantity: {prediction['fertilizer_quantity']:.2f} grams")
        print(f"   Confidence: {prediction['confidence']:.2f}")
        
        # Clean up
        if os.path.exists('test_plant.jpg'):
            os.remove('test_plant.jpg')
            
    except Exception as e:
        print(f"❌ Prediction test failed: {str(e)}")
    
    return True

if __name__ == "__main__":
    create_basic_model()

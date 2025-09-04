"""
Quick training script to create a basic model for testing
"""

import os
import numpy as np
from fertilizer_prediction_model import FertilizerPredictionModel, create_synthetic_dataset, prepare_data

def quick_train():
    """Create a quick model for testing"""
    print("🚀 Quick Training for Testing")
    print("=" * 40)
    
    # Create small dataset
    print("Creating synthetic dataset...")
    df = create_synthetic_dataset(num_samples=50, output_dir='quick_dataset')
    
    # Prepare data
    print("Preparing data...")
    train_df, val_df, test_df, label_encoder, quantity_scaler = prepare_data(df)
    
    # Initialize model
    print("Initializing model...")
    model = FertilizerPredictionModel(img_size=(224, 224), num_fertilizer_types=8)
    model.build_model()
    model.compile_model(learning_rate=0.001)
    
    # Create data generators
    print("Creating data generators...")
    train_gen, val_gen = model.create_data_generators(train_df, val_df, batch_size=8)
    
    # Quick training (just a few epochs)
    print("Training model (quick test)...")
    try:
        history = model.train(train_gen, val_gen, epochs=2, fine_tune_epochs=1)
        print("✅ Training completed!")
        
        # Save model
        print("Saving model...")
        model.save_model('fertilizer_prediction_model')
        print("✅ Model saved successfully!")
        
        # Test prediction
        print("Testing prediction...")
        test_image = "quick_dataset/images/plant_0001.jpg"
        if os.path.exists(test_image):
            prediction = model.predict_single_image(test_image)
            print(f"✅ Prediction test successful!")
            print(f"   Fertilizer: {prediction['fertilizer_type']}")
            print(f"   Quantity: {prediction['fertilizer_quantity']:.2f} grams")
        
        # Clean up
        import shutil
        if os.path.exists('quick_dataset'):
            shutil.rmtree('quick_dataset')
        
        print("\n🎉 Quick training completed successfully!")
        print("You can now run: python app.py")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    quick_train()

"""
Test script to verify the fertilizer prediction implementation
"""

import os
import sys
import numpy as np
import tensorflow as tf
from fertilizer_prediction_model import FertilizerPredictionModel, create_synthetic_dataset, prepare_data

def test_model_creation():
    """Test if the model can be created successfully"""
    print("Testing model creation...")
    try:
        model = FertilizerPredictionModel(img_size=(224, 224), num_fertilizer_types=8)
        model.build_model()
        model.compile_model(learning_rate=0.001)
        print("✅ Model creation successful!")
        return model
    except Exception as e:
        print(f"❌ Model creation failed: {str(e)}")
        return None

def test_dataset_creation():
    """Test if synthetic dataset can be created"""
    print("\nTesting dataset creation...")
    try:
        df = create_synthetic_dataset(num_samples=10, output_dir='test_dataset')
        print(f"✅ Dataset creation successful! Created {len(df)} samples")
        return df
    except Exception as e:
        print(f"❌ Dataset creation failed: {str(e)}")
        return None

def test_data_preparation(df):
    """Test data preparation pipeline"""
    print("\nTesting data preparation...")
    try:
        train_df, val_df, test_df, label_encoder, quantity_scaler = prepare_data(df)
        print(f"✅ Data preparation successful!")
        print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df, label_encoder, quantity_scaler
    except Exception as e:
        print(f"❌ Data preparation failed: {str(e)}")
        return None, None, None, None, None

def test_model_save_load(model):
    """Test model saving and loading"""
    print("\nTesting model save/load...")
    try:
        # Save model
        model.save_model('test_model')
        print("✅ Model saved successfully!")
        
        # Load model
        new_model = FertilizerPredictionModel()
        new_model.load_model('test_model')
        print("✅ Model loaded successfully!")
        
        # Clean up test files
        import glob
        test_files = glob.glob('test_model*')
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
        print("✅ Test files cleaned up!")
        
        return True
    except Exception as e:
        print(f"❌ Model save/load failed: {str(e)}")
        return False

def test_prediction(model):
    """Test model prediction"""
    print("\nTesting model prediction...")
    try:
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite('test_image.jpg', dummy_image)
        
        # Test prediction
        prediction = model.predict_single_image('test_image.jpg')
        print("✅ Prediction successful!")
        print(f"   Fertilizer Type: {prediction['fertilizer_type']}")
        print(f"   Quantity: {prediction['fertilizer_quantity']:.2f} grams")
        print(f"   Confidence: {prediction['confidence']:.2f}")
        
        # Clean up
        if os.path.exists('test_image.jpg'):
            os.remove('test_image.jpg')
        
        return True
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing FertilizerAI Implementation")
    print("=" * 50)
    
    # Test 1: Model Creation
    model = test_model_creation()
    if model is None:
        print("\n❌ Critical error: Cannot create model. Exiting.")
        return
    
    # Test 2: Dataset Creation
    df = test_dataset_creation()
    if df is None:
        print("\n❌ Critical error: Cannot create dataset. Exiting.")
        return
    
    # Test 3: Data Preparation
    train_df, val_df, test_df, label_encoder, quantity_scaler = test_data_preparation(df)
    if train_df is None:
        print("\n❌ Critical error: Cannot prepare data. Exiting.")
        return
    
    # Test 4: Model Save/Load
    save_load_success = test_model_save_load(model)
    
    # Test 5: Prediction
    prediction_success = test_prediction(model)
    
    # Clean up test dataset
    import shutil
    if os.path.exists('test_dataset'):
        shutil.rmtree('test_dataset')
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"✅ Model Creation: {'PASS' if model else 'FAIL'}")
    print(f"✅ Dataset Creation: {'PASS' if df is not None else 'FAIL'}")
    print(f"✅ Data Preparation: {'PASS' if train_df is not None else 'FAIL'}")
    print(f"✅ Model Save/Load: {'PASS' if save_load_success else 'FAIL'}")
    print(f"✅ Prediction: {'PASS' if prediction_success else 'FAIL'}")
    
    if all([model, df is not None, train_df is not None, save_load_success, prediction_success]):
        print("\n🎉 All tests passed! Implementation is working correctly.")
        print("\nNext steps:")
        print("1. Run: python train_model.py --num-samples 1000 --epochs 30")
        print("2. Run: python app.py")
        print("3. Open: http://localhost:5000")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()

"""
Train the model with realistic synthetic data to improve predictions
"""

import os
import numpy as np
import pandas as pd
import cv2
from fertilizer_prediction_model import FertilizerPredictionModel, prepare_data
import matplotlib.pyplot as plt

def create_realistic_dataset(num_samples=1000, output_dir='realistic_dataset'):
    """
    Create a more realistic synthetic dataset with better patterns
    """
    print(f"Creating realistic dataset with {num_samples} samples...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    # Define realistic patterns based on plant characteristics
    plant_conditions = {
        'healthy_green': {'fertilizer': 'Compost', 'quantity_range': (50, 150)},
        'yellowing_leaves': {'fertilizer': 'Urea', 'quantity_range': (100, 250)},
        'stunted_growth': {'fertilizer': 'NPK', 'quantity_range': (150, 300)},
        'purple_leaves': {'fertilizer': 'Phosphate', 'quantity_range': (80, 200)},
        'brown_edges': {'fertilizer': 'Potash', 'quantity_range': (100, 220)},
        'weak_stems': {'fertilizer': 'DAP', 'quantity_range': (120, 280)},
        'organic_garden': {'fertilizer': 'Organic', 'quantity_range': (60, 180)},
        'hydroponic': {'fertilizer': 'Liquid_Fertilizer', 'quantity_range': (40, 120)}
    }
    
    image_paths = []
    fertilizer_types = []
    quantities = []
    
    for i in range(num_samples):
        # Select plant condition
        condition = np.random.choice(list(plant_conditions.keys()))
        condition_info = plant_conditions[condition]
        
        # Generate image based on condition
        img = generate_plant_image(condition, i)
        
        # Save image
        img_path = f"{output_dir}/images/plant_{i:04d}.jpg"
        cv2.imwrite(img_path, img)
        
        # Set labels based on condition
        fertilizer_type = condition_info['fertilizer']
        quantity = np.random.uniform(*condition_info['quantity_range'])
        
        image_paths.append(img_path)
        fertilizer_types.append(fertilizer_type)
        quantities.append(quantity)
    
    # Create DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'fertilizer_type': fertilizer_types,
        'quantity_grams': quantities
    })
    
    # Save dataset info
    df.to_csv(f"{output_dir}/dataset.csv", index=False)
    
    print(f"✅ Created realistic dataset with {len(df)} samples")
    print("Fertilizer distribution:")
    print(df['fertilizer_type'].value_counts())
    
    return df

def generate_plant_image(condition, seed):
    """
    Generate synthetic plant images based on condition
    """
    np.random.seed(seed)
    
    # Base image (green plant)
    img = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
    
    # Add plant-like patterns
    center_x, center_y = 112, 112
    
    # Create plant shape
    for y in range(224):
        for x in range(224):
            # Distance from center
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Plant stem
            if 50 < dist < 80 and y > center_y:
                img[y, x] = [50, 100, 50]  # Dark green stem
            
            # Plant leaves
            elif dist < 60 and y < center_y + 20:
                if condition == 'healthy_green':
                    img[y, x] = [50, 150, 50]  # Healthy green
                elif condition == 'yellowing_leaves':
                    img[y, x] = [150, 150, 50]  # Yellow-green
                elif condition == 'purple_leaves':
                    img[y, x] = [100, 50, 100]  # Purple
                elif condition == 'brown_edges':
                    img[y, x] = [100, 80, 50]  # Brownish
                else:
                    img[y, x] = [50, 120, 50]  # Default green
    
    # Add condition-specific patterns
    if condition == 'stunted_growth':
        # Make plant smaller
        img = cv2.resize(img, (150, 150))
        img = cv2.resize(img, (224, 224))
    elif condition == 'weak_stems':
        # Add some noise to simulate weakness
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img

def train_model():
    """
    Train the model with realistic data
    """
    print("🚀 Training FertilizerAI Model")
    print("=" * 50)
    
    # Create realistic dataset
    df = create_realistic_dataset(num_samples=2000)
    
    # Prepare data
    print("\nPreparing data...")
    train_df, val_df, test_df, label_encoder, quantity_scaler = prepare_data(df)
    
    # Initialize model
    print("\nInitializing model...")
    model = FertilizerPredictionModel(img_size=(224, 224), num_fertilizer_types=8)
    model.build_model()
    model.compile_model(learning_rate=0.001)
    
    # Create data generators
    print("\nCreating data generators...")
    train_gen, val_gen = model.create_data_generators(train_df, val_df, batch_size=16)
    
    # Train model
    print("\nTraining model...")
    try:
        history = model.train(train_gen, val_gen, epochs=20, fine_tune_epochs=10)
        
        # Save trained model
        print("\nSaving trained model...")
        model.save_model('fertilizer_prediction_model')
        
        # Test predictions
        print("\nTesting predictions...")
        test_predictions = []
        for i in range(5):
            test_img = f"realistic_dataset/images/plant_{i:04d}.jpg"
            if os.path.exists(test_img):
                prediction = model.predict_single_image(test_img)
                test_predictions.append(prediction)
                print(f"Test {i+1}: {prediction['fertilizer_type']} - {prediction['fertilizer_quantity']:.1f}g (Conf: {prediction['confidence']:.1%})")
        
        # Plot training history
        model.plot_training_history(history)
        plt.savefig('training_history.png')
        print("\n✅ Training completed successfully!")
        print("📊 Training history saved as 'training_history.png'")
        
        # Clean up
        import shutil
        if os.path.exists('realistic_dataset'):
            shutil.rmtree('realistic_dataset')
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return False

if __name__ == "__main__":
    train_model()

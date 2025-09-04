"""
Simple training approach without complex data generators
"""

import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from fertilizer_prediction_model import FertilizerPredictionModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def create_simple_dataset(num_samples=500):
    """
    Create a simple dataset with clear patterns
    """
    print(f"Creating simple dataset with {num_samples} samples...")
    
    # Create output directory
    os.makedirs('simple_dataset/images', exist_ok=True)
    
    # Define clear patterns
    patterns = {
        'Urea': {'color': [100, 200, 100], 'quantity_range': (100, 250)},
        'DAP': {'color': [150, 150, 100], 'quantity_range': (120, 280)},
        'Compost': {'color': [80, 120, 80], 'quantity_range': (50, 150)},
        'NPK': {'color': [120, 180, 120], 'quantity_range': (150, 300)},
        'Potash': {'color': [200, 150, 100], 'quantity_range': (100, 220)},
        'Phosphate': {'color': [100, 100, 150], 'quantity_range': (80, 200)},
        'Organic': {'color': [90, 110, 90], 'quantity_range': (60, 180)},
        'Liquid_Fertilizer': {'color': [130, 160, 130], 'quantity_range': (40, 120)}
    }
    
    image_paths = []
    fertilizer_types = []
    quantities = []
    
    for i in range(num_samples):
        # Select fertilizer type
        fert_type = np.random.choice(list(patterns.keys()))
        pattern = patterns[fert_type]
        
        # Generate image with specific color pattern
        img = generate_simple_image(pattern['color'], i)
        
        # Save image
        img_path = f"simple_dataset/images/plant_{i:04d}.jpg"
        cv2.imwrite(img_path, img)
        
        # Set quantity
        quantity = np.random.uniform(*pattern['quantity_range'])
        
        image_paths.append(img_path)
        fertilizer_types.append(fert_type)
        quantities.append(quantity)
    
    # Create DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'fertilizer_type': fertilizer_types,
        'quantity_grams': quantities
    })
    
    df.to_csv('simple_dataset/dataset.csv', index=False)
    
    print(f"✅ Created dataset with {len(df)} samples")
    print("Fertilizer distribution:")
    print(df['fertilizer_type'].value_counts())
    
    return df

def generate_simple_image(base_color, seed):
    """
    Generate simple images with specific color patterns
    """
    np.random.seed(seed)
    
    # Create base image
    img = np.full((224, 224, 3), base_color, dtype=np.uint8)
    
    # Add some variation
    noise = np.random.randint(-30, 30, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add plant-like structure
    center_x, center_y = 112, 112
    
    # Create simple plant shape
    for y in range(224):
        for x in range(224):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Plant area
            if dist < 80:
                # Add some texture
                if (x + y) % 3 == 0:
                    img[y, x] = np.clip(img[y, x] + 20, 0, 255)
    
    return img

def simple_train():
    """
    Simple training without complex data generators
    """
    print("🚀 Simple Training for FertilizerAI")
    print("=" * 40)
    
    # Create dataset
    df = create_simple_dataset(num_samples=800)
    
    # Prepare data
    print("\nPreparing data...")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['fertilizer_type_encoded'] = label_encoder.fit_transform(df['fertilizer_type'])
    
    # Scale quantities
    quantity_scaler = StandardScaler()
    df['quantity_normalized'] = quantity_scaler.fit_transform(df['quantity_grams'].values.reshape(-1, 1)).flatten()
    
    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['fertilizer_type'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['fertilizer_type'])
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = FertilizerPredictionModel(img_size=(224, 224), num_fertilizer_types=8)
    model.build_model()
    model.compile_model(learning_rate=0.001)
    
    # Set the scalers
    model.label_encoder = label_encoder
    model.quantity_scaler = quantity_scaler
    
    # Prepare training data
    print("\nPreparing training data...")
    
    def load_and_preprocess_image(image_path):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        return img_array / 255.0
    
    # Load training data
    X_train = np.array([load_and_preprocess_image(path) for path in train_df['image_path']])
    y_train_type = tf.keras.utils.to_categorical(train_df['fertilizer_type_encoded'], 8)
    y_train_quantity = train_df['quantity_normalized'].values.reshape(-1, 1)
    
    # Load validation data
    X_val = np.array([load_and_preprocess_image(path) for path in val_df['image_path']])
    y_val_type = tf.keras.utils.to_categorical(val_df['fertilizer_type_encoded'], 8)
    y_val_quantity = val_df['quantity_normalized'].values.reshape(-1, 1)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Train model
    print("\nTraining model...")
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
    
    # Train
    history = model.model.fit(
        X_train, [y_train_type, y_train_quantity],
        validation_data=(X_val, [y_val_type, y_val_quantity]),
        epochs=30,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    print("\nSaving trained model...")
    model.save_model('fertilizer_prediction_model')
    
    # Test predictions
    print("\nTesting predictions...")
    test_predictions = []
    for i in range(5):
        test_img = f"simple_dataset/images/plant_{i:04d}.jpg"
        if os.path.exists(test_img):
            prediction = model.predict_single_image(test_img)
            test_predictions.append(prediction)
            print(f"Test {i+1}: {prediction['fertilizer_type']} - {prediction['fertilizer_quantity']:.1f}g (Conf: {prediction['confidence']:.1%})")
    
    # Plot training history
    model.plot_training_history(history)
    import matplotlib.pyplot as plt
    plt.savefig('simple_training_history.png')
    print("\n✅ Training completed successfully!")
    print("📊 Training history saved as 'simple_training_history.png'")
    
    # Clean up
    import shutil
    if os.path.exists('simple_dataset'):
        shutil.rmtree('simple_dataset')
    
    return True

if __name__ == "__main__":
    simple_train()

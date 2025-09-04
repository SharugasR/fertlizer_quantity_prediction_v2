"""
Create realistic synthetic plant images with actual plant structures
"""

import os
import numpy as np
import cv2
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from fertilizer_prediction_model import FertilizerPredictionModel

def create_plant_image(condition, seed, size=(224, 224)):
    """
    Create realistic plant images with actual plant structures
    """
    np.random.seed(seed)
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    # Background (soil)
    img[:, :] = [139, 69, 19]  # Brown soil color
    
    center_x, center_y = size[1] // 2, size[0] // 2
    
    if condition == 'healthy_green':
        # Healthy green plant
        create_healthy_plant(img, center_x, center_y)
        return img, 'Compost', (50, 150)
        
    elif condition == 'yellowing_leaves':
        # Nitrogen deficiency - yellowing leaves
        create_yellowing_plant(img, center_x, center_y)
        return img, 'Urea', (100, 250)
        
    elif condition == 'stunted_growth':
        # General nutrient deficiency
        create_stunted_plant(img, center_x, center_y)
        return img, 'NPK', (150, 300)
        
    elif condition == 'purple_leaves':
        # Phosphorus deficiency
        create_purple_plant(img, center_x, center_y)
        return img, 'Phosphate', (80, 200)
        
    elif condition == 'brown_edges':
        # Potassium deficiency
        create_brown_edges_plant(img, center_x, center_y)
        return img, 'Potash', (100, 220)
        
    elif condition == 'weak_stems':
        # Phosphorus deficiency
        create_weak_stem_plant(img, center_x, center_y)
        return img, 'DAP', (120, 280)
        
    elif condition == 'organic_garden':
        # Organic farming
        create_organic_plant(img, center_x, center_y)
        return img, 'Organic', (60, 180)
        
    elif condition == 'hydroponic':
        # Hydroponic system
        create_hydroponic_plant(img, center_x, center_y)
        return img, 'Liquid_Fertilizer', (40, 120)

def create_healthy_plant(img, center_x, center_y):
    """Create a healthy green plant"""
    # Stem
    cv2.rectangle(img, (center_x-3, center_y), (center_x+3, center_y+40), (34, 139, 34), -1)
    
    # Leaves (healthy green)
    leaves = [
        (center_x-20, center_y-10, 15, 8),  # Left leaf
        (center_x+5, center_y-15, 12, 6),   # Right leaf
        (center_x-15, center_y-25, 10, 5),  # Top left
        (center_x+8, center_y-20, 8, 4),    # Top right
    ]
    
    for x, y, w, h in leaves:
        cv2.ellipse(img, (x, y), (w, h), 0, 0, 360, (34, 139, 34), -1)
        # Add leaf veins
        cv2.line(img, (x, y), (x+w//2, y+h//2), (0, 100, 0), 1)

def create_yellowing_plant(img, center_x, center_y):
    """Create a plant with yellowing leaves (N deficiency)"""
    # Stem
    cv2.rectangle(img, (center_x-3, center_y), (center_x+3, center_y+40), (34, 139, 34), -1)
    
    # Yellowing leaves
    leaves = [
        (center_x-20, center_y-10, 15, 8),
        (center_x+5, center_y-15, 12, 6),
        (center_x-15, center_y-25, 10, 5),
        (center_x+8, center_y-20, 8, 4),
    ]
    
    for x, y, w, h in leaves:
        # Yellow-green color
        cv2.ellipse(img, (x, y), (w, h), 0, 0, 360, (0, 255, 255), -1)
        # Add yellow spots
        cv2.circle(img, (x+w//3, y+h//3), 2, (0, 200, 200), -1)

def create_stunted_plant(img, center_x, center_y):
    """Create a stunted plant (general deficiency)"""
    # Shorter, thinner stem
    cv2.rectangle(img, (center_x-2, center_y), (center_x+2, center_y+25), (34, 139, 34), -1)
    
    # Smaller, fewer leaves
    leaves = [
        (center_x-15, center_y-8, 10, 5),
        (center_x+3, center_y-10, 8, 4),
    ]
    
    for x, y, w, h in leaves:
        cv2.ellipse(img, (x, y), (w, h), 0, 0, 360, (50, 120, 50), -1)

def create_purple_plant(img, center_x, center_y):
    """Create a plant with purple leaves (P deficiency)"""
    # Stem
    cv2.rectangle(img, (center_x-3, center_y), (center_x+3, center_y+40), (34, 139, 34), -1)
    
    # Purple leaves
    leaves = [
        (center_x-20, center_y-10, 15, 8),
        (center_x+5, center_y-15, 12, 6),
        (center_x-15, center_y-25, 10, 5),
        (center_x+8, center_y-20, 8, 4),
    ]
    
    for x, y, w, h in leaves:
        cv2.ellipse(img, (x, y), (w, h), 0, 0, 360, (128, 0, 128), -1)

def create_brown_edges_plant(img, center_x, center_y):
    """Create a plant with brown edges (K deficiency)"""
    # Stem
    cv2.rectangle(img, (center_x-3, center_y), (center_x+3, center_y+40), (34, 139, 34), -1)
    
    # Leaves with brown edges
    leaves = [
        (center_x-20, center_y-10, 15, 8),
        (center_x+5, center_y-15, 12, 6),
        (center_x-15, center_y-25, 10, 5),
        (center_x+8, center_y-20, 8, 4),
    ]
    
    for x, y, w, h in leaves:
        # Green center
        cv2.ellipse(img, (x, y), (w, h), 0, 0, 360, (34, 139, 34), -1)
        # Brown edges
        cv2.ellipse(img, (x, y), (w, h), 0, 0, 360, (139, 69, 19), 2)

def create_weak_stem_plant(img, center_x, center_y):
    """Create a plant with weak stem (P deficiency)"""
    # Thin, weak stem
    cv2.rectangle(img, (center_x-1, center_y), (center_x+1, center_y+35), (34, 139, 34), -1)
    
    # Drooping leaves
    leaves = [
        (center_x-18, center_y-5, 12, 6),
        (center_x+3, center_y-8, 10, 5),
    ]
    
    for x, y, w, h in leaves:
        cv2.ellipse(img, (x, y), (w, h), 0, 0, 360, (50, 100, 50), -1)

def create_organic_plant(img, center_x, center_y):
    """Create an organic garden plant"""
    # Thicker stem
    cv2.rectangle(img, (center_x-4, center_y), (center_x+4, center_y+45), (34, 139, 34), -1)
    
    # Large, healthy leaves
    leaves = [
        (center_x-25, center_y-12, 18, 10),
        (center_x+7, center_y-18, 15, 8),
        (center_x-20, center_y-30, 12, 6),
        (center_x+10, center_y-25, 10, 5),
    ]
    
    for x, y, w, h in leaves:
        cv2.ellipse(img, (x, y), (w, h), 0, 0, 360, (0, 150, 0), -1)
        # Add organic texture
        cv2.circle(img, (x+w//2, y+h//2), 1, (0, 200, 0), -1)

def create_hydroponic_plant(img, center_x, center_y):
    """Create a hydroponic plant"""
    # Clean, straight stem
    cv2.rectangle(img, (center_x-2, center_y), (center_x+2, center_y+50), (34, 139, 34), -1)
    
    # Uniform, clean leaves
    leaves = [
        (center_x-18, center_y-15, 14, 7),
        (center_x+4, center_y-20, 12, 6),
        (center_x-16, center_y-35, 10, 5),
        (center_x+6, center_y-30, 8, 4),
    ]
    
    for x, y, w, h in leaves:
        cv2.ellipse(img, (x, y), (w, h), 0, 0, 360, (0, 180, 0), -1)

def create_realistic_dataset(num_samples=1000):
    """
    Create a realistic dataset with actual plant structures
    """
    print(f"Creating realistic plant dataset with {num_samples} samples...")
    
    # Create output directory
    os.makedirs('realistic_plants/images', exist_ok=True)
    
    conditions = [
        'healthy_green', 'yellowing_leaves', 'stunted_growth', 'purple_leaves',
        'brown_edges', 'weak_stems', 'organic_garden', 'hydroponic'
    ]
    
    image_paths = []
    fertilizer_types = []
    quantities = []
    
    for i in range(num_samples):
        # Select condition
        condition = np.random.choice(conditions)
        
        # Generate plant image
        img, fertilizer_type, quantity_range = create_plant_image(condition, i)
        
        # Save image
        img_path = f"realistic_plants/images/plant_{i:04d}.jpg"
        cv2.imwrite(img_path, img)
        
        # Set quantity
        quantity = np.random.uniform(*quantity_range)
        
        image_paths.append(img_path)
        fertilizer_types.append(fertilizer_type)
        quantities.append(quantity)
    
    # Create DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'fertilizer_type': fertilizer_types,
        'quantity_grams': quantities
    })
    
    df.to_csv('realistic_plants/dataset.csv', index=False)
    
    print(f"✅ Created realistic dataset with {len(df)} samples")
    print("Fertilizer distribution:")
    print(df['fertilizer_type'].value_counts())
    
    return df

def train_with_realistic_data():
    """
    Train model with realistic plant images
    """
    print("🌱 Training with Realistic Plant Images")
    print("=" * 50)
    
    # Create dataset
    df = create_realistic_dataset(num_samples=800)
    
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
        epochs=20,
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
        test_img = f"realistic_plants/images/plant_{i:04d}.jpg"
        if os.path.exists(test_img):
            prediction = model.predict_single_image(test_img)
            test_predictions.append(prediction)
            print(f"Test {i+1}: {prediction['fertilizer_type']} - {prediction['fertilizer_quantity']:.1f}g (Conf: {prediction['confidence']:.1%})")
    
    # Plot training history
    model.plot_training_history(history)
    import matplotlib.pyplot as plt
    plt.savefig('realistic_training_history.png')
    print("\n✅ Training completed successfully!")
    print("📊 Training history saved as 'realistic_training_history.png'")
    
    # Clean up
    import shutil
    if os.path.exists('realistic_plants'):
        shutil.rmtree('realistic_plants')
    
    return True

if __name__ == "__main__":
    train_with_realistic_data()

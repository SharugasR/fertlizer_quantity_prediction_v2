"""
Download and prepare real plant datasets for training
"""

import os
import requests
import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import shutil

def download_plantvillage_dataset():
    """
    Download PlantVillage dataset from Kaggle
    Note: You need to set up Kaggle API first
    """
    print("🌱 Downloading PlantVillage Dataset")
    print("=" * 50)
    
    # Instructions for Kaggle setup
    print("""
    SETUP INSTRUCTIONS:
    1. Go to https://www.kaggle.com/account
    2. Create API token (kaggle.json)
    3. Place kaggle.json in ~/.kaggle/ (or C:/Users/YourName/.kaggle/)
    4. Install: pip install kaggle
    5. Run: kaggle datasets download -d abdallahalidev/plantvillage-dataset
    """)
    
    # Alternative: Manual download
    print("""
    MANUAL DOWNLOAD:
    1. Go to: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
    2. Download the dataset
    3. Extract to 'plantvillage_dataset' folder
    """)
    
    return True

def create_fertilizer_mapping():
    """
    Create mapping from plant diseases to fertilizer recommendations
    Based on nutrient deficiency symptoms
    """
    fertilizer_mapping = {
        # Nitrogen deficiency symptoms
        'Apple___Apple_scab': 'Urea',
        'Apple___Black_rot': 'Urea', 
        'Apple___Cedar_apple_rust': 'Urea',
        'Apple___healthy': 'Compost',
        
        # Phosphorus deficiency symptoms
        'Corn___Common_rust': 'Phosphate',
        'Corn___Northern_Leaf_Blight': 'Phosphate',
        'Corn___healthy': 'Compost',
        
        # Potassium deficiency symptoms
        'Grape___Black_rot': 'Potash',
        'Grape___Esca': 'Potash',
        'Grape___healthy': 'Compost',
        
        # General nutrient deficiency
        'Potato___Early_blight': 'NPK',
        'Potato___Late_blight': 'NPK',
        'Potato___healthy': 'Compost',
        
        # Organic recommendations
        'Tomato___Bacterial_spot': 'Organic',
        'Tomato___Early_blight': 'Organic',
        'Tomato___healthy': 'Compost',
        
        # Liquid fertilizer for specific conditions
        'Strawberry___Leaf_scorch': 'Liquid_Fertilizer',
        'Strawberry___healthy': 'Compost',
        
        # DAP for specific diseases
        'Pepper___Bacterial_spot': 'DAP',
        'Pepper___healthy': 'Compost',
    }
    
    return fertilizer_mapping

def prepare_real_dataset(dataset_path='plantvillage_dataset'):
    """
    Prepare real plant dataset for fertilizer prediction
    """
    print("🌱 Preparing Real Plant Dataset")
    print("=" * 40)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please download PlantVillage dataset first")
        return None
    
    # Create fertilizer mapping
    fertilizer_mapping = create_fertilizer_mapping()
    
    # Collect all images and labels
    image_paths = []
    fertilizer_types = []
    quantities = []
    
    print("Scanning dataset...")
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Extract class from folder name
                class_name = os.path.basename(root)
                
                # Map to fertilizer type
                fertilizer_type = fertilizer_mapping.get(class_name, 'NPK')
                
                # Generate quantity based on disease severity
                if 'healthy' in class_name.lower():
                    quantity = np.random.uniform(50, 150)  # Less fertilizer for healthy plants
                else:
                    quantity = np.random.uniform(100, 300)  # More fertilizer for diseased plants
                
                image_paths.append(os.path.join(root, file))
                fertilizer_types.append(fertilizer_type)
                quantities.append(quantity)
    
    # Create DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'fertilizer_type': fertilizer_types,
        'quantity_grams': quantities
    })
    
    print(f"✅ Found {len(df)} images")
    print("Fertilizer distribution:")
    print(df['fertilizer_type'].value_counts())
    
    # Save dataset
    df.to_csv('real_plant_dataset.csv', index=False)
    print("✅ Dataset saved as 'real_plant_dataset.csv'")
    
    return df

def train_with_real_data():
    """
    Train model with real plant data
    """
    print("🚀 Training with Real Plant Data")
    print("=" * 40)
    
    # Check if dataset exists
    if not os.path.exists('real_plant_dataset.csv'):
        print("❌ Real dataset not found. Please run prepare_real_dataset() first")
        return False
    
    # Load dataset
    df = pd.read_csv('real_plant_dataset.csv')
    print(f"Loaded {len(df)} real plant images")
    
    # Prepare data
    from fertilizer_prediction_model import FertilizerPredictionModel, prepare_data
    
    train_df, val_df, test_df, label_encoder, quantity_scaler = prepare_data(df)
    
    # Initialize model
    model = FertilizerPredictionModel(img_size=(224, 224), num_fertilizer_types=8)
    model.build_model()
    model.compile_model(learning_rate=0.001)
    
    # Set scalers
    model.label_encoder = label_encoder
    model.quantity_scaler = quantity_scaler
    
    # Prepare training data
    import tensorflow as tf
    
    def load_and_preprocess_image(image_path):
        try:
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            return img_array / 255.0
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    # Load training data (sample for demo)
    print("Loading training data...")
    sample_size = min(1000, len(train_df))  # Use sample for demo
    train_sample = train_df.sample(n=sample_size, random_state=42)
    
    X_train = []
    y_train_type = []
    y_train_quantity = []
    
    for idx, row in train_sample.iterrows():
        img_array = load_and_preprocess_image(row['image_path'])
        if img_array is not None:
            X_train.append(img_array)
            y_train_type.append(row['fertilizer_type_encoded'])
            y_train_quantity.append(row['quantity_normalized'])
    
    X_train = np.array(X_train)
    y_train_type = tf.keras.utils.to_categorical(y_train_type, 8)
    y_train_quantity = np.array(y_train_quantity).reshape(-1, 1)
    
    print(f"Training data shape: {X_train.shape}")
    
    # Train model
    print("Training model...")
    history = model.model.fit(
        X_train, [y_train_type, y_train_quantity],
        epochs=10,  # Reduced for demo
        batch_size=16,
        verbose=1
    )
    
    # Save model
    model.save_model('fertilizer_prediction_model_real')
    print("✅ Model trained and saved!")
    
    return True

def main():
    """
    Main function to set up real dataset
    """
    print("🌱 Real Plant Dataset Setup")
    print("=" * 50)
    
    print("""
    OPTIONS:
    1. Download PlantVillage dataset (RECOMMENDED)
    2. Use existing dataset if available
    3. Continue with synthetic data for demo
    """)
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice == '1':
        download_plantvillage_dataset()
    elif choice == '2':
        if os.path.exists('plantvillage_dataset'):
            df = prepare_real_dataset()
            if df is not None:
                train_with_real_data()
        else:
            print("❌ No dataset found. Please download first.")
    else:
        print("Continuing with synthetic data for demo purposes.")

if __name__ == "__main__":
    main()

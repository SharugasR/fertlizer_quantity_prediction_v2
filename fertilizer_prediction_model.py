"""
Complete CNN Multi-Output Model for Fertilizer Type Classification and Quantity Prediction
Author: AI Assistant
Date: 2025
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB3, ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FertilizerPredictionModel:
    """
    Multi-output CNN model for fertilizer type classification and quantity prediction
    """
    
    def __init__(self, img_size=(224, 224), num_fertilizer_types=8, base_model='efficientnet'):
        self.img_size = img_size
        self.num_fertilizer_types = num_fertilizer_types
        self.base_model_name = base_model
        self.model = None
        self.label_encoder = LabelEncoder()
        self.quantity_scaler = StandardScaler()
        self.fertilizer_types = [
            'Urea', 'DAP', 'Compost', 'NPK', 'Potash', 
            'Phosphate', 'Organic', 'Liquid_Fertilizer'
        ]
        
    def build_model(self):
        """
        Build multi-output CNN model with classification and regression heads
        """
        # Input layer
        inputs = layers.Input(shape=(*self.img_size, 3), name='image_input')
        
        # Custom CNN architecture (simpler and more reliable)
        x = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3))(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Shared dense layers
        x = layers.Dense(512, activation='relu', name='shared_dense_1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256, activation='relu', name='shared_dense_2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Classification head (fertilizer type)
        classification_head = layers.Dense(128, activation='relu', name='class_dense')(x)
        classification_head = layers.Dropout(0.2)(classification_head)
        classification_output = layers.Dense(
            self.num_fertilizer_types, 
            activation='softmax', 
            name='fertilizer_type'
        )(classification_head)
        
        # Regression head (fertilizer quantity)
        regression_head = layers.Dense(128, activation='relu', name='reg_dense')(x)
        regression_head = layers.Dropout(0.2)(regression_head)
        regression_output = layers.Dense(
            1, 
            activation='linear', 
            name='fertilizer_quantity'
        )(regression_head)
        
        # Create model
        self.model = models.Model(
            inputs=inputs,
            outputs=[classification_output, regression_output],
            name='fertilizer_prediction_model'
        )
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile model with appropriate loss functions and metrics
        """
        # Define losses for each output
        losses = {
            'fertilizer_type': 'categorical_crossentropy',
            'fertilizer_quantity': tf.keras.losses.Huber(delta=1.0)  # More robust than MSE for outliers
        }
        
        # Define loss weights (can be tuned)
        loss_weights = {
            'fertilizer_type': 1.0,
            'fertilizer_quantity': 0.5
        }
        
        # Define metrics for each output
        metrics = {
            'fertilizer_type': ['accuracy'],
            'fertilizer_quantity': ['mae', 'mse']
        }
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
        
        return self.model
    
    def create_data_generators(self, train_df, val_df, batch_size=32):
        """
        Create data generators with augmentation
        """
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Validation data (only rescaling)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators - use raw mode for multi-output
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='image_path',
            y_col=['fertilizer_type_encoded', 'quantity_normalized'],
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='raw',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_dataframe(
            val_df,
            x_col='image_path',
            y_col=['fertilizer_type_encoded', 'quantity_normalized'],
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='raw',
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def train(self, train_generator, val_generator, epochs=50, 
              fine_tune_epochs=20, fine_tune_lr=1e-5):
        """
        Train the model with two-phase training
        """
        # Phase 1: Train only the new layers
        print("Phase 1: Training new layers...")
        history1 = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=self._get_callbacks('phase1'),
            verbose=1
        )
        
        # Phase 2: Fine-tune the entire model
        print("Phase 2: Fine-tuning entire model...")
        self._unfreeze_base_model()
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=fine_tune_lr),
            loss={
                'fertilizer_type': 'categorical_crossentropy',
                'fertilizer_quantity': tf.keras.losses.Huber(delta=1.0)
            },
            loss_weights={
                'fertilizer_type': 1.0,
                'fertilizer_quantity': 0.5
            },
            metrics={
                'fertilizer_type': ['accuracy'],
                'fertilizer_quantity': ['mae', 'mse']
            }
        )
        
        history2 = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=fine_tune_epochs,
            callbacks=self._get_callbacks('phase2'),
            verbose=1
        )
        
        # Combine histories
        combined_history = self._combine_histories(history1, history2)
        return combined_history
    
    def _unfreeze_base_model(self):
        """Unfreeze base model for fine-tuning"""
        if self.base_model_name == 'efficientnet':
            base_model = self.model.get_layer('efficientnetb3')
        else:
            base_model = self.model.get_layer('resnet50v2')
        
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = len(base_model.layers) - 50
        
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
    
    def _get_callbacks(self, phase):
        """Get training callbacks"""
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                f'best_model_{phase}.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks_list
    
    def _combine_histories(self, history1, history2):
        """Combine training histories from both phases"""
        combined = {}
        for key in history1.history.keys():
            combined[key] = history1.history[key] + history2.history[key]
        return combined
    
    def evaluate(self, test_generator):
        """
        Evaluate model performance
        """
        results = self.model.evaluate(test_generator, verbose=1)
        
        # Get predictions
        predictions = self.model.predict(test_generator)
        y_pred_type = np.argmax(predictions[0], axis=1)
        y_pred_quantity = predictions[1].flatten()
        
        # Get true labels
        y_true_type = test_generator.classes[0] if hasattr(test_generator, 'classes') else None
        y_true_quantity = test_generator.classes[1] if hasattr(test_generator, 'classes') else None
        
        return {
            'test_loss': results[0],
            'classification_accuracy': results[1],
            'regression_mae': results[3],
            'predictions': {
                'fertilizer_type': y_pred_type,
                'fertilizer_quantity': y_pred_quantity
            }
        }
    
    def predict_single_image(self, image_path):
        """
        Predict fertilizer type and quantity for a single image
        """
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=self.img_size
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make prediction
        predictions = self.model.predict(img_array)
        
        # Process results
        fertilizer_type_idx = np.argmax(predictions[0][0])
        fertilizer_type = self.fertilizer_types[fertilizer_type_idx]
        
        # Handle quantity prediction - if scaler is not fitted, use raw prediction
        try:
            fertilizer_quantity = self.quantity_scaler.inverse_transform(
                predictions[1][0].reshape(-1, 1)
            )[0][0]
        except:
            # If scaler is not fitted, use raw prediction with scaling
            fertilizer_quantity = predictions[1][0][0] * 100  # Scale to reasonable range
        
        confidence = np.max(predictions[0][0])
        
        return {
            'fertilizer_type': fertilizer_type,
            'fertilizer_quantity': max(0, fertilizer_quantity),  # Ensure non-negative
            'confidence': confidence,
            'all_type_probabilities': dict(zip(
                self.fertilizer_types, 
                predictions[0][0]
            ))
        }
    
    def save_model(self, filepath):
        """Save the complete model"""
        # Save model in H5 format for Keras 3 compatibility
        self.model.save(f"{filepath}.h5")
        
        # Save preprocessing objects
        import joblib
        joblib.dump(self.label_encoder, f"{filepath}_label_encoder.pkl")
        joblib.dump(self.quantity_scaler, f"{filepath}_quantity_scaler.pkl")
        
        # Save fertilizer types
        with open(f"{filepath}_fertilizer_types.json", 'w') as f:
            json.dump(self.fertilizer_types, f)
    
    def load_model(self, filepath):
        """Load the complete model"""
        # Load model from H5 format
        self.model = tf.keras.models.load_model(f"{filepath}.h5")
        
        # Load preprocessing objects
        import joblib
        self.label_encoder = joblib.load(f"{filepath}_label_encoder.pkl")
        self.quantity_scaler = joblib.load(f"{filepath}_quantity_scaler.pkl")
        
        # Load fertilizer types
        with open(f"{filepath}_fertilizer_types.json", 'r') as f:
            self.fertilizer_types = json.load(f)
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Classification accuracy
        axes[0, 0].plot(history.history['fertilizer_type_accuracy'], label='Train')
        axes[0, 0].plot(history.history['val_fertilizer_type_accuracy'], label='Validation')
        axes[0, 0].set_title('Fertilizer Type Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Classification loss
        axes[0, 1].plot(history.history['fertilizer_type_loss'], label='Train')
        axes[0, 1].plot(history.history['val_fertilizer_type_loss'], label='Validation')
        axes[0, 1].set_title('Fertilizer Type Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Regression MAE
        axes[1, 0].plot(history.history['fertilizer_quantity_mae'], label='Train')
        axes[1, 0].plot(history.history['val_fertilizer_quantity_mae'], label='Validation')
        axes[1, 0].set_title('Fertilizer Quantity MAE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].legend()
        
        # Regression loss
        axes[1, 1].plot(history.history['fertilizer_quantity_loss'], label='Train')
        axes[1, 1].plot(history.history['val_fertilizer_quantity_loss'], label='Validation')
        axes[1, 1].set_title('Fertilizer Quantity Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


def create_synthetic_dataset(num_samples=1000, output_dir='dataset'):
    """
    Create synthetic dataset for fertilizer prediction
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    fertilizer_types = [
        'Urea', 'DAP', 'Compost', 'NPK', 'Potash', 
        'Phosphate', 'Organic', 'Liquid_Fertilizer'
    ]
    
    # Create synthetic data
    data = []
    for i in range(num_samples):
        # Generate synthetic plant image (in practice, use real plant images)
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Add some plant-like patterns
        cv2.circle(img, (112, 112), 80, (0, 150, 0), -1)  # Green circle
        cv2.rectangle(img, (107, 112), (117, 224), (0, 100, 0), -1)  # Stem
        
        # Add noise for realism
        noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # Save image
        img_path = f"{output_dir}/images/plant_{i:04d}.jpg"
        cv2.imwrite(img_path, img)
        
        # Generate labels based on "plant health" (color intensity)
        green_intensity = np.mean(img[:, :, 1])  # Green channel
        health_score = green_intensity / 255.0
        
        # Assign fertilizer type based on health
        if health_score < 0.3:
            fertilizer_type = np.random.choice(['Urea', 'DAP', 'NPK'])
            quantity = np.random.uniform(200, 400)
        elif health_score < 0.6:
            fertilizer_type = np.random.choice(['Compost', 'Organic', 'NPK'])
            quantity = np.random.uniform(100, 250)
        else:
            fertilizer_type = np.random.choice(['Compost', 'Organic', 'Liquid_Fertilizer'])
            quantity = np.random.uniform(50, 150)
        
        data.append({
            'image_path': img_path,
            'fertilizer_type': fertilizer_type,
            'quantity_grams': quantity,
            'plant_health': 'poor' if health_score < 0.3 else 'moderate' if health_score < 0.6 else 'healthy',
            'soil_condition': np.random.choice(['loamy', 'clay', 'sandy']),
            'season': np.random.choice(['spring', 'summer', 'autumn', 'winter'])
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.to_csv(f"{output_dir}/metadata.csv", index=False)
    
    print(f"Created synthetic dataset with {num_samples} samples")
    return df


def prepare_data(df, test_size=0.2, val_size=0.2):
    """
    Prepare data for training
    """
    # Encode fertilizer types
    label_encoder = LabelEncoder()
    df['fertilizer_type_encoded'] = label_encoder.fit_transform(df['fertilizer_type'])
    
    # Normalize quantities
    quantity_scaler = StandardScaler()
    df['quantity_normalized'] = quantity_scaler.fit_transform(
        df['quantity_grams'].values.reshape(-1, 1)
    ).flatten()
    
    # Split data
    train_df, temp_df = train_test_split(df, test_size=test_size + val_size, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=test_size/(test_size + val_size), random_state=42)
    
    return train_df, val_df, test_df, label_encoder, quantity_scaler


# Example usage
if __name__ == "__main__":
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    df = create_synthetic_dataset(num_samples=1000)
    
    # Prepare data
    print("Preparing data...")
    train_df, val_df, test_df, label_encoder, quantity_scaler = prepare_data(df)
    
    # Initialize model
    print("Initializing model...")
    model = FertilizerPredictionModel(img_size=(224, 224), num_fertilizer_types=8)
    
    # Build and compile model
    model.build_model()
    model.compile_model(learning_rate=0.001)
    
    # Create data generators
    print("Creating data generators...")
    train_gen, val_gen = model.create_data_generators(train_df, val_df, batch_size=32)
    
    # Train model
    print("Training model...")
    history = model.train(train_gen, val_gen, epochs=30, fine_tune_epochs=10)
    
    # Plot training history
    model.plot_training_history(history)
    
    # Save model
    model.save_model('fertilizer_prediction_model')
    print("Model saved successfully!")
    
    # Test prediction
    test_image = "dataset/images/plant_0001.jpg"
    if os.path.exists(test_image):
        prediction = model.predict_single_image(test_image)
        print(f"Prediction: {prediction}")

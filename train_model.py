# This file has been renamed from train_real_model.py to train_model.py
# 
# Train the model with realistic synthetic data to improve predictions

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
    
    # ... (rest of the existing code) ...

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
"""
Complete Training Script for Fertilizer Prediction Model
Author: AI Assistant
Date: 2025

This script provides a comprehensive training pipeline for the fertilizer prediction model.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from fertilizer_prediction_model import FertilizerPredictionModel, create_synthetic_dataset, prepare_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Comprehensive model training class"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
        self.results = {}
        
    def create_dataset(self):
        """Create or load dataset"""
        dataset_path = self.config['dataset_path']
        
        if os.path.exists(f"{dataset_path}/metadata.csv"):
            logger.info("Loading existing dataset...")
            df = pd.read_csv(f"{dataset_path}/metadata.csv")
        else:
            logger.info("Creating synthetic dataset...")
            df = create_synthetic_dataset(
                num_samples=self.config['num_samples'],
                output_dir=dataset_path
            )
        
        logger.info(f"Dataset created with {len(df)} samples")
        return df
    
    def prepare_data(self, df):
        """Prepare data for training"""
        logger.info("Preparing data for training...")
        
        # Check if we have the required columns
        required_columns = ['image_path', 'fertilizer_type', 'quantity_grams']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Encode fertilizer types
        label_encoder = LabelEncoder()
        df['fertilizer_type_encoded'] = label_encoder.fit_transform(df['fertilizer_type'])
        
        # Normalize quantities
        quantity_scaler = StandardScaler()
        df['quantity_normalized'] = quantity_scaler.fit_transform(
            df['quantity_grams'].values.reshape(-1, 1)
        ).flatten()
        
        # Split data
        train_df, temp_df = train_test_split(
            df, 
            test_size=self.config['test_size'] + self.config['val_size'], 
            random_state=42,
            stratify=df['fertilizer_type_encoded']
        )
        
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=self.config['test_size']/(self.config['test_size'] + self.config['val_size']), 
            random_state=42,
            stratify=temp_df['fertilizer_type_encoded']
        )
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df, label_encoder, quantity_scaler
    
    def initialize_model(self):
        """Initialize the model"""
        logger.info("Initializing model...")
        
        self.model = FertilizerPredictionModel(
            img_size=self.config['img_size'],
            num_fertilizer_types=self.config['num_fertilizer_types'],
            base_model=self.config['base_model']
        )
        
        # Build and compile model
        self.model.build_model()
        self.model.compile_model(learning_rate=self.config['learning_rate'])
        
        logger.info("Model initialized successfully")
        return self.model
    
    def create_data_generators(self, train_df, val_df):
        """Create data generators"""
        logger.info("Creating data generators...")
        
        train_gen, val_gen = self.model.create_data_generators(
            train_df, val_df, 
            batch_size=self.config['batch_size']
        )
        
        return train_gen, val_gen
    
    def train_model(self, train_gen, val_gen):
        """Train the model"""
        logger.info("Starting model training...")
        
        # Custom callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.train(
            train_gen, val_gen,
            epochs=self.config['epochs'],
            fine_tune_epochs=self.config['fine_tune_epochs'],
            fine_tune_lr=self.config['fine_tune_lr']
        )
        
        logger.info("Model training completed")
        return self.history
    
    def evaluate_model(self, test_df):
        """Evaluate the model"""
        logger.info("Evaluating model...")
        
        # Create test generator
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        test_gen = test_datagen.flow_from_dataframe(
            test_df,
            x_col='image_path',
            y_col=['fertilizer_type_encoded', 'quantity_normalized'],
            target_size=self.config['img_size'],
            batch_size=self.config['batch_size'],
            class_mode='multi_output',
            shuffle=False
        )
        
        # Evaluate
        results = self.model.evaluate(test_gen)
        
        # Get predictions for detailed analysis
        predictions = self.model.model.predict(test_gen)
        y_pred_type = np.argmax(predictions[0], axis=1)
        y_pred_quantity = predictions[1].flatten()
        
        # Get true labels
        y_true_type = test_gen.classes[0]
        y_true_quantity = test_gen.classes[1]
        
        # Classification metrics
        class_report = classification_report(
            y_true_type, y_pred_type,
            target_names=self.model.fertilizer_types,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true_type, y_pred_type)
        
        # Regression metrics
        mae = np.mean(np.abs(y_true_quantity - y_pred_quantity))
        mse = np.mean((y_true_quantity - y_pred_quantity) ** 2)
        rmse = np.sqrt(mse)
        
        self.results = {
            'test_loss': results[0],
            'classification_accuracy': results[1],
            'regression_mae': results[3],
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'regression_metrics': {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse)
            }
        }
        
        logger.info(f"Model evaluation completed - Accuracy: {results[1]:.4f}, MAE: {results[3]:.4f}")
        return self.results
    
    def plot_results(self):
        """Plot training results and evaluation metrics"""
        logger.info("Creating visualization plots...")
        
        # Training history
        if self.history:
            self.model.plot_training_history(self.history)
        
        # Confusion matrix
        if 'confusion_matrix' in self.results:
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                self.results['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=self.model.fertilizer_types,
                yticklabels=self.model.fertilizer_types
            )
            plt.title('Confusion Matrix - Fertilizer Type Classification')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Classification report
        if 'classification_report' in self.results:
            report_df = pd.DataFrame(self.results['classification_report']).transpose()
            report_df = report_df.drop(['support'], axis=1)
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(report_df, annot=True, cmap='YlOrRd', fmt='.3f')
            plt.title('Classification Report - Precision, Recall, F1-Score')
            plt.tight_layout()
            plt.savefig('classification_report.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_results(self):
        """Save training results and model"""
        logger.info("Saving results...")
        
        # Save model
        self.model.save_model(self.config['model_save_path'])
        
        # Save training history
        if self.history:
            with open('training_history.json', 'w') as f:
                json.dump(self.history, f, indent=2)
        
        # Save evaluation results
        with open('evaluation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save configuration
        with open('training_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info("Results saved successfully")
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        try:
            # Create dataset
            df = self.create_dataset()
            
            # Prepare data
            train_df, val_df, test_df, label_encoder, quantity_scaler = self.prepare_data(df)
            
            # Initialize model
            self.initialize_model()
            
            # Create data generators
            train_gen, val_gen = self.create_data_generators(train_df, val_df)
            
            # Train model
            self.train_model(train_gen, val_gen)
            
            # Evaluate model
            self.evaluate_model(test_df)
            
            # Plot results
            self.plot_results()
            
            # Save results
            self.save_results()
            
            logger.info("Training pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise

def get_default_config():
    """Get default training configuration"""
    return {
        'dataset_path': 'dataset',
        'num_samples': 1000,
        'img_size': (224, 224),
        'num_fertilizer_types': 8,
        'base_model': 'efficientnet',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 30,
        'fine_tune_epochs': 10,
        'fine_tune_lr': 1e-5,
        'test_size': 0.2,
        'val_size': 0.2,
        'early_stopping_patience': 10,
        'lr_patience': 5,
        'model_save_path': 'fertilizer_prediction_model'
    }

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Fertilizer Prediction Model')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of synthetic samples to create')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--base-model', type=str, default='efficientnet', 
                       choices=['efficientnet', 'resnet'], help='Base model architecture')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Override with command line arguments
    if args.num_samples:
        config['num_samples'] = args.num_samples
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.base_model:
        config['base_model'] = args.base_model
    
    # Print configuration
    logger.info("Training Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Run training pipeline
    trainer.run_training_pipeline()

if __name__ == "__main__":
    main()

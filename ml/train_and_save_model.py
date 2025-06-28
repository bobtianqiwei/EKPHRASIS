#EKPHRASIS by Bob Tianqi Wei, Shayne Shen, UC Berkeley, 2024

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def train_composition_model():
    # Load the pre-trained VGG16 model without the top classification layer
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the convolutional layers of VGG16 to prevent them from being updated during training
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers
    x = base_model.output
    x = Flatten()(x)  # Flatten the feature maps
    x = Dense(128, activation='relu')(x)  # Fully connected layer
    predictions = Dense(1, activation='sigmoid')(x)  # Output layer with sigmoid activation for binary classification

    # Define the new model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Display the model architecture
    model.summary()

    # Set the path to the dataset
    data_dir = "../dataset/balance/Bob's classes"  # Path to your dataset

    # Use ImageDataGenerator to automatically split the dataset into training and validation sets
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # 80% for training, 20% for validation

    # Load training data
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),  # VGG16 requires input images to be 224x224
        batch_size=32,
        class_mode='binary',  # Binary classification task
        subset='training'  # Use 80% of the data for training
    )

    # Load validation data
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation'  # Use 20% of the data for validation
    )

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=10  # Set the number of epochs according to your dataset
    )

    # Save the trained model
    model.save('composition_model.h5')
    print("Model saved as composition_model.h5")

if __name__ == "__main__":
    train_composition_model() 
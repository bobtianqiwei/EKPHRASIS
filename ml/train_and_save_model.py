# train_and_save_model.py developed by Bob Tianqi Wei
# EKPHRASIS by Bob Tianqi Wei, Shayne Shen, UC Berkeley, 2024
# Train one model per vocabulary. Dataset: dataset/<vocabulary_id>/class_0, class_1.
# Usage: python train_and_save_model.py <vocabulary_id>   e.g. python train_and_save_model.py visual_balance

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys

# Vocabulary id must match CRITERIA in model_server.py and a folder under dataset/
CLASS_ORDER = ['class_0', 'class_1']  # class_0 = less, class_1 = more (app shows high = More)

def train_vocabulary_model(vocabulary_id: str):
    if not vocabulary_id or vocabulary_id.strip() == '':
        print("Usage: python train_and_save_model.py <vocabulary_id>")
        print("  e.g. python train_and_save_model.py visual_balance")
        print("  Dataset must exist: ../dataset/<vocabulary_id>/class_0 and class_1")
        sys.exit(1)
    vocabulary_id = vocabulary_id.strip()

    # Paths: dataset and output model (under ml/)
    ml_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(ml_dir)
    data_dir = os.path.join(project_root, 'dataset', vocabulary_id)
    if not os.path.isdir(data_dir):
        print(f"Dataset directory not found: {data_dir}")
        sys.exit(1)
    for c in CLASS_ORDER:
        p = os.path.join(data_dir, c)
        if not os.path.isdir(p):
            print(f"Missing class folder: {p}")
            sys.exit(1)

    # Model: VGG16 + head
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Data
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        classes=CLASS_ORDER,
        subset='training'
    )
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        classes=CLASS_ORDER,
        subset='validation'
    )

    # Calculate class weights for imbalanced dataset
    train_samples = train_generator.samples
    class_0_count = sum(train_generator.classes == 0)
    class_1_count = sum(train_generator.classes == 1)
    weight_0 = (1 / class_0_count) * (train_samples / 2.0)
    weight_1 = (1 / class_1_count) * (train_samples / 2.0)
    class_weight = {0: weight_0, 1: weight_1}
    print(f"Class counts - 0: {class_0_count}, 1: {class_1_count}")
    print(f"Using class weights - 0: {weight_0:.2f}, 1: {weight_1:.2f}")

    # Train
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        class_weight=class_weight,
        epochs=10
    )

    # Save under ml/models/
    models_dir = os.path.join(ml_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, f'{vocabulary_id}.h5')
    model.save(out_path)
    print(f"Model saved: {out_path}")
    print("Register this vocabulary in ml/model_server.py CRITERIA if not already there.")

if __name__ == "__main__":
    vocabulary_id = sys.argv[1] if len(sys.argv) > 1 else ""
    train_vocabulary_model(vocabulary_id)

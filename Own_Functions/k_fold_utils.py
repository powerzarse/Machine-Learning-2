# k_fold_utils.py
import numpy as np
from tensorflow.keras.models import clone_model
from sklearn.model_selection import KFold
from .subgenerator_utils import create_subgenerator  # Importiere die create_subgenerator Funktion

def k_fold_cross_validation(original_model, train_generator, n_splits=5, epochs=5):
    dataset_size = train_generator.n
    num_classes = len(train_generator.class_indices)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_performance = []
    fold_histories = []

    for train_indices, val_indices in kf.split(range(dataset_size)):
        train_subgen = create_subgenerator(train_generator, train_indices)
        val_subgen = create_subgenerator(train_generator, val_indices)

        model = clone_model(original_model)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(train_subgen, epochs=epochs, validation_data=val_subgen, steps_per_epoch=len(train_indices)//train_generator.batch_size, validation_steps=len(val_indices)//train_generator.batch_size)

        fold_histories.append(history.history)
        scores = model.evaluate(val_subgen, verbose=0, steps=len(val_indices)//train_generator.batch_size)
        fold_performance.append(scores[1])

    average_performance = np.mean(fold_performance)
    return fold_performance, average_performance, fold_histories

import concurrent.futures
from tensorflow.keras.models import clone_model
from .subgenerator_utils import create_subgenerator  # Importiere die create_subgenerator Funktion
from sklearn.model_selection import KFold
from .subgenerator_utils import create_subgenerator  # Importiere die create_subgenerator Funktion
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_and_evaluate_fold(fold_num, train_indices, val_indices, original_model, train_generator, epochs):
    train_subgen = create_subgenerator(train_generator, train_indices)
    val_subgen = create_subgenerator(train_generator, val_indices)

    model = clone_model(original_model)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    checkpoint_path = f"model_fold_{fold_num}.h5"
    model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

    history = model.fit(
        train_subgen, 
        epochs=epochs, 
        validation_data=val_subgen, 
        steps_per_epoch=len(train_indices)//train_generator.batch_size, 
        validation_steps=len(val_indices)//train_generator.batch_size, 
        callbacks=[early_stopping, model_checkpoint],  
        workers=6,
        verbose=1
    )

    scores = model.evaluate(val_subgen, verbose=0, steps=len(val_indices)//train_generator.batch_size)
    return fold_num, scores[1], history.history

def k_fold_cross_validation_parallel(original_model, train_generator, n_splits=5, epochs=5):
    dataset_size = train_generator.n
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    tasks = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for fold, (train_indices, val_indices) in enumerate(kf.split(range(dataset_size))):
            tasks.append(executor.submit(train_and_evaluate_fold, fold+1, train_indices, val_indices, original_model, train_generator, epochs))

    fold_performance = []
    fold_histories = []
    for future in concurrent.futures.as_completed(tasks):
        fold_num, score, history = future.result()
        fold_performance.append(score)
        fold_histories.append(history)
        print(f"Fold {fold_num} abgeschlossen mit Score: {score}")

    average_performance = np.mean(fold_performance)
    return fold_performance, average_performance, fold_histories

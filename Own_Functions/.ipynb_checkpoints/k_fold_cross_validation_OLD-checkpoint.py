def k_fold_cross_validation(original_model, train_generator, n_splits=5, epochs=5):
    # Extracting data and labels from the generator
    dataset_size = train_generator.n
    num_classes = len(train_generator.class_indices)
    batch_size = train_generator.batch_size
    target_size = train_generator.target_size

    X = np.zeros((dataset_size, target_size[0], target_size[1], 3), dtype=np.float32)
    y = np.zeros((dataset_size, num_classes), dtype=np.float32)

    for i in range(len(train_generator)):
        batch_x, batch_y = train_generator.next()
        X[i * batch_size:(i + 1) * batch_size] = batch_x
        y[i * batch_size:(i + 1) * batch_size] = batch_y

    # K-Fold Cross-Validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_performance = []

    for train_indices, val_indices in kf.split(X):
        # Splitting data
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        # Cloning the original model structure
        model = clone_model(original_model)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Training the model
        model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))

        # Evaluating the model
        scores = model.evaluate(X_val, y_val, verbose=0)
        fold_performance.append(scores[1])  # Assuming [1] is accuracy

    # Average performance
    average_performance = np.mean(fold_performance)
    return fold_performance, average_performance
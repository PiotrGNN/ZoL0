model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    # Compile the model with appropriate optimizer and loss function
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae', 'mse']
    )

    # Add early stopping to prevent overfitting
    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train, 
        epochs=epochs, 
        verbose=1, 
        validation_split=0.2,
        callbacks=[early_stopping]
    )
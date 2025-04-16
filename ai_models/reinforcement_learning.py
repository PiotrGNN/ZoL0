model = Sequential()
        model.add(Dense(64, input_dim=input_dim, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))

        # Dodanie kompilacji modelu przed treningiem z pełną konfiguracją
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mse', 'accuracy']
        )
        
        # Sprawdzenie czy model został poprawnie skompilowany
        if not getattr(model, "_is_compiled", False):
            logging.warning("Model nie został poprawnie skompilowany - próba ponownej kompilacji")
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Dodanie callback'a dla early stopping
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
model = Sequential()
        model.add(Dense(64, input_dim=input_dim, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))

        # Dodanie kompilacji modelu przed treningiem
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        model.fit(X_train, y_train, epochs=epochs, verbose=1, validation_split=0.2)
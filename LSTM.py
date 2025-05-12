import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Hyperparameters
max_number_words = 50000  # Maximum vocabulary size
batch_size = 64
epochs = 15

# Load the dataset
df = pd.read_csv('/Users/riteshchandra/Desktop/Course Work /Course Work - Semester - III/AIT636/Movie_Reviews.csv')
print(df.head())

# Bar chart for class sizes
sns.countplot(data=df, x='label')
plt.xlabel('Label')
plt.title('Size of classes')
plt.show()

# Tokenize the text
tokenizer = Tokenizer(num_words=max_number_words,
                      filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
                      lower=True)
tokenizer.fit_on_texts(df['text'].values)

# Print the size of the lexicon
print('Found %s unique tokens.' % len(tokenizer.word_index))

# Convert labels to categorical format
y = pd.get_dummies(df['label']).values
print('Shape of label tensor:', y.shape)

# Hyperparameter combinations
combinations = [
    (64, 75, 50), (64, 75, 75), (64, 75, 100),
    (64, 100, 50), (64, 100, 75), (64, 100, 100),
    (128, 75, 50), (128, 75, 75), (128, 75, 100),
    (128, 100, 50), (128, 100, 75), (128, 100, 100),
    (256, 75, 50), (256, 75, 75), (256, 75, 100),
    (256, 100, 50), (256, 100, 75), (256, 100, 100)
]

# To store the results for plotting
results = {}
for LSTM_units, embedding_dim, max_phrase_length in combinations:
    print(f"\nRunning model for LSTM_units={LSTM_units}, embedding_dim={embedding_dim}, max_phrase_length={max_phrase_length}")

    # Adjust input data based on max_phrase_length
    x = tokenizer.texts_to_sequences(df['text'].values)
    x = pad_sequences(x, maxlen=max_phrase_length)
    print('Shape of data tensor:', x.shape)

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

    # Build the model
    model = Sequential()
    model.add(Embedding(input_dim=max_number_words,
                        output_dim=embedding_dim,
                        input_length=max_phrase_length))
    model.add(SpatialDropout1D(0.1))
    model.add(LSTM(units=LSTM_units, activation='relu', dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=2, activation='softmax'))

    # Build the model with input shape to avoid unbuilt layer warnings
    model.build(input_shape=(None, max_phrase_length))

    # Print model summary
    print("\nModel Summary:")
    model.summary()

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=1)

    # Save results
    key = f"LSTM:{LSTM_units}, Embedding:{embedding_dim}, Length:{max_phrase_length}"
    results[key] = {
        'train_accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy']
    }
    print(f"Completed: {key}")

    # Plot the graph for the current configuration
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"Accuracy vs Epochs for {key}")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

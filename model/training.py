from msilib import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from pathlib import Path
import os
import numpy as np


# prepare dataset
DATA_PATH = Path(r"C:\Users\Klomm\Desktop\AI\emotional_detection\model\dataset")
NO_SEQUENCE = 30
SEQUENCE_LENGTH = 30
actions = np.array(["yes", "no"])

sequences, labels = [], []
for action_idx, action_name in enumerate(actions):
    for sequence in range(NO_SEQUENCE):
        windows = []
        for frame_num in range(SEQUENCE_LENGTH):
            npy_path = DATA_PATH / action_name / str(sequence) / f"{frame_num}.npy"
            res = np.load(npy_path)
        sequences.append(windows)
        labels.append(action_idx)

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# split datset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# create model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# train model
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

# evaluate model
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

confusion_matrix = multilabel_confusion_matrix(ytrue, yhat)
score = accuracy_score(ytrue, yhat)

print(f"confusion matrix: {confusion_matrix}")
print(f"score: {score}")

# save model
model.save("action.h5")
print("save action.h5 model")
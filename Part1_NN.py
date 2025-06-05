from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD
from keras.regularizers import l2
import joblib

EPOCHS = 100
BATCH_SIZE = 32
r = 0.01

data = pd.read_csv('/content/alzheimers_disease_data (1).csv', skiprows=1, header=None)

X = data.drop(columns=[0, 33, 34])
y = data[33]

scaler = StandardScaler()
X = scaler.fit_transform(X)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_accuracies = []
cv_losses = []
cv_mse = []

all_train_losses = []
all_val_losses = []
all_train_acc = []
all_val_acc = []


for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"🔵 Training Fold {fold}/5...")

    # Διαχωρισμός training & validation sets
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 🔹 6. Κατασκευή Νευρωνικού Δικτύου
    model = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(64, activation='relu', kernel_regularizer=l2(r)),
        Dense(32, activation='relu', kernel_regularizer=l2(r)),
        Dense(16, activation='relu', kernel_regularizer=l2(r)),
        Dense(1, activation='sigmoid', kernel_regularizer=l2(r))
    ])

    # 🔹 7. Compiling του Μοντέλου
    opt = SGD(learning_rate=0.01, momentum=0.6)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # 🔹 8. Εκπαίδευση (Training) του Μοντέλου
    history = model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

    # 🔹 9. Αξιολόγηση στο Validation Set
    val_ce, val_accuracy = model.evaluate(X_val, y_val)
    print(f"✅ Validation - Accuracy: {val_accuracy:.4f}, CE-Loss: {val_ce:.4f}")

    cv_accuracies.append(val_accuracy)
    cv_losses.append(val_ce)

    all_train_losses.append(history.history['loss'])
    all_val_losses.append(history.history['val_loss'])
    all_train_acc.append(history.history['accuracy'])
    all_val_acc.append(history.history['val_accuracy'])


mean_cv_accuracy = np.mean(cv_accuracies)
mean_cv_losses = np.mean(cv_losses)

# 🔹 Bar plot of validation accuracy per fold
plt.figure(figsize=(8, 4))
plt.bar(range(1, len(cv_accuracies)+1), cv_accuracies, color='skyblue')
plt.axhline(mean_cv_accuracy, color='red', linestyle='--', label=f'Mean Accuracy = {mean_cv_accuracy:.4f}')
plt.title('Validation Accuracy per Fold')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(range(1, len(cv_accuracies)+1))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


print(f"\n📊 Μέση ακρίβεια στο CV: {mean_cv_accuracy:.4f}")
print(f"\n📊 Μέσα losses στο CV: {mean_cv_losses:.4f}")

# 🔹 Υπολογισμός μέσου όρου ανά epoch
avg_train_loss = np.mean(all_train_losses, axis=0)
avg_val_loss = np.mean(all_val_losses, axis=0)
avg_train_acc = np.mean(all_train_acc, axis=0)
avg_val_acc = np.mean(all_val_acc, axis=0)

epochs_range = range(1, len(avg_train_loss) + 1)

# 🔹 Γράφημα Loss
plt.figure(figsize=(10, 4))
plt.plot(epochs_range, avg_train_loss, label='Μ.Ο. Training Loss', color='blue')
plt.plot(epochs_range, avg_val_loss, label='Μ.Ο. Validation Loss', color='orange')
plt.title('Μέσος Όρος Cross-Entropy Loss ανά Εποχή')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 🔹 Γράφημα Accuracy
plt.figure(figsize=(10, 4))
plt.plot(epochs_range, avg_train_acc, label='Μ.Ο. Training Accuracy', color='green')
plt.plot(epochs_range, avg_val_acc, label='Μ.Ο. Validation Accuracy', color='red')
plt.title('Μέσος Όρος Accuracy ανά Εποχή')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Αποθήκευση του μοντέλου μετά την εκπαίδευση
model.save('best_model.keras')

# Αποθήκευση των scaler/encoder για να μεταχειριστείς τα δεδομένα με τον ίδιο τρόπο
joblib.dump(scaler, 'scaler.pkl')

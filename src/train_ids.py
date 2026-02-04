import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# PATH
# =========================
DATASET_PATH = r"C:\Users\moham\.cache\kagglehub\datasets\rohulaminlabid\iotid20-dataset\versions\2\IoT Network Intrusion Dataset.csv"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATASET_PATH)

# =========================
# DROP NON-USEFUL COLUMNS
# =========================
DROP_COLS = [
    "Flow_ID", "Src_IP", "Dst_IP", "Timestamp",
    "Cat", "Sub_Cat"
]

for col in DROP_COLS:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# =========================
# CLEAN DATA
# =========================
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# =========================
# LABEL ENCODING
# =========================
label_encoder = LabelEncoder()
df["Label"] = label_encoder.fit_transform(df["Label"])

# Check the mapping
print("Label mapping:")
for original, encoded in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(f"  {original} -> {encoded}")

X = df.drop("Label", axis=1)
y = df["Label"]

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SAVE SCALER AND LABEL ENCODER
joblib.dump(scaler, "scaler.save")
joblib.dump(label_encoder, "label_encoder.save")  # Save label encoder too!

# =========================
# IDS MODEL (DEEP LEARNING)
# =========================
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=1024,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# =========================
# EVALUATION
# =========================
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# =========================
# SAVE MODEL
# =========================
model.save("iot_ids_model.keras")  # Use newer format
print("\nModel, scaler, and label encoder saved.")
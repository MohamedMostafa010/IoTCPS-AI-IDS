import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# =========================
# LOAD MODEL & SCALER & LABEL ENCODER
# =========================
model = load_model("iot_ids_model.keras")  # Fixed: loading .keras file
scaler = joblib.load("scaler.save")
label_encoder = joblib.load("label_encoder.save")

# Print the label mapping for reference
print("Label mapping from saved encoder:")
for original, encoded in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(f"  {original} -> {encoded}")

# =========================
# LOAD DATA
# =========================
DATASET_PATH = r"C:\Users\moham\.cache\kagglehub\datasets\rohulaminlabid\iotid20-dataset\versions\2\IoT Network Intrusion Dataset.csv"
df = pd.read_csv(DATASET_PATH)

# =========================
# PREPROCESSING (MUST MATCH TRAINING)
# =========================
DROP_COLS = ["Flow_ID", "Src_IP", "Dst_IP", "Timestamp", "Cat", "Sub_Cat"]
for col in DROP_COLS:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# CRITICAL: Use the saved label encoder to match training
df["Label"] = label_encoder.transform(df["Label"])

X = df.drop("Label", axis=1)
y = df["Label"]

# Get category columns BEFORE dropping them
df_with_cats = pd.read_csv(DATASET_PATH)
df_with_cats = df_with_cats.loc[df.index]  # Align with cleaned df

# =========================
# IDS DETECTION FUNCTION
# =========================
def detect_intrusion(sample, cat, sub_cat):
    sample = np.array(sample).reshape(1, -1)
    sample = scaler.transform(sample)
    prediction = model.predict(sample, verbose=0)[0][0]
    
    # Debug: print prediction probability
    print(f"Prediction: {prediction:.4f} (1=Normal, 0=Anomaly)", end=" - ")
    
    if prediction > 0.5:
        return f"NORMAL TRAFFIC → "
    else:
        return f"ATTACK DETECTED → {cat} ({sub_cat})"

# =========================
# TEST WITH MIXED SAMPLES
# =========================
print("\n" + "=" * 60)
print("Testing IDS on mixed samples:")
print("=" * 60)

num_samples = 10
normal_idx = y[y == 1].index  # 1 = Normal in our mapping
attack_idx = y[y == 0].index  # 0 = Anomaly/Attack in our mapping

# Ensure we have enough samples
num_normal = min(num_samples // 2, len(normal_idx))
num_attack = min(num_samples // 2, len(attack_idx))

normal_samples = normal_idx[:num_normal]
attack_samples = attack_idx[:num_attack]

mixed_idx = list(normal_samples) + list(attack_samples)

correct = 0
total = len(mixed_idx)

print(f"\nTesting {num_normal} Normal samples and {num_attack} Attack samples:\n")

for i, idx in enumerate(mixed_idx):
    sample = X.loc[idx]
    cat = df_with_cats.loc[idx, "Cat"]
    sub_cat = df_with_cats.loc[idx, "Sub_Cat"]
    true_label = "Normal" if y.loc[idx] == 1 else "Attack"
    result = detect_intrusion(sample, cat, sub_cat)
    
    # Check if prediction is correct
    predicted_normal = "NORMAL" in result
    is_correct = (predicted_normal and true_label == "Normal") or (not predicted_normal and true_label == "Attack")
    
    if is_correct:
        correct += 1
        status = "✓ CORRECT"
    else:
        status = "✗ WRONG"
    
    print(f"Sample {i+1}: {result} [True: {true_label}] {status}")

print("\n" + "=" * 60)
print(f"Accuracy: {correct}/{total} = {(correct/total)*100:.1f}%")
print("=" * 60)

# =========================
# BULK TEST FOR ACCURACY CHECK
# =========================
print("\n\nBulk Testing Accuracy:")
print("=" * 60)

# Take a larger test sample
test_size = min(1000, len(df))
if len(df) > test_size:
    test_indices = np.random.choice(df.index, test_size, replace=False)
    X_test = X.loc[test_indices]
    y_test = y.loc[test_indices]
    
    X_test_scaled = scaler.transform(X_test)
    predictions = (model.predict(X_test_scaled, verbose=0) > 0.5).astype(int)
    
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy on {test_size} random samples: {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Anomaly  Normal")
    print(f"Actual Anomaly  {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"Actual Normal   {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    print("\n" + classification_report(y_test, predictions, 
                                     target_names=["Anomaly", "Normal"]))
    
    if cm[1,0] > 0:
        print(f"WARNING: {cm[1,0]} normal samples are being classified as attacks!")
    if cm[0,1] > 0:
        print(f"WARNING: {cm[0,1]} attack samples are being classified as normal!")

# =========================
# TEST SINGLE SAMPLES EXPLICITLY
# =========================
print("\n" + "=" * 60)
print("Testing specific samples explicitly:")
print("=" * 60)

# Find one clear attack and one clear normal sample
attack_sample_idx = attack_idx[0]
normal_sample_idx = normal_idx[0]

print("\n1. Testing a known ATTACK sample:")
attack_sample = X.loc[attack_sample_idx]
attack_cat = df_with_cats.loc[attack_sample_idx, "Cat"]
attack_subcat = df_with_cats.loc[attack_sample_idx, "Sub_Cat"]
result = detect_intrusion(attack_sample, attack_cat, attack_subcat)
print(f"   Result: {result}")
print(f"   Expected: ATTACK DETECTED → {attack_cat} ({attack_subcat})")

print("\n2. Testing a known NORMAL sample:")
normal_sample = X.loc[normal_sample_idx]
normal_cat = df_with_cats.loc[normal_sample_idx, "Cat"]
normal_subcat = df_with_cats.loc[normal_sample_idx, "Sub_Cat"]
result = detect_intrusion(normal_sample, normal_cat, normal_subcat)
print(f"   Result: {result}")
print(f"   Expected: NORMAL TRAFFIC → {normal_cat} ({normal_subcat})")
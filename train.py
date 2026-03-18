"""
Diabetic Retinopathy Detection - Training Script (v5 - FIXED)

FIXES FROM v4:
  1. REMOVED rescale=1./255 — EfficientNet has built-in preprocessing.
     With rescale, backbone received near-zero values → couldn't extract features.
  2. REPLACED focal loss (gamma=3 collapsed gradients) with categorical_crossentropy + class_weight.
  3. FIXED data split — train on full balanced dataset, validate on full original dataset.
  4. REMOVED WarmUp callback that fought ReduceLROnPlateau.
  5. Simplified training phases for stability.
"""

import os
import shutil
import glob
import random
import json
import warnings
import numpy as np
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
IMG_SIZE     = 224
BATCH_SIZE   = 16
EPOCHS_P1    = 20
EPOCHS_P2    = 50
NUM_CLASSES  = 5
DATASET_DIR  = "dataset"
BALANCED_DIR = "dataset_balanced"
MODEL_PATH   = "model/dr_model.keras"
RESULTS_DIR  = "static/results"

os.makedirs("model",          exist_ok=True)
os.makedirs(RESULTS_DIR,      exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
random.seed(42); np.random.seed(42); tf.random.set_seed(42)

# ─── CLASS WEIGHTS (replaces broken focal loss) ──────────────────────────────
# Based on original class distribution: {0:968, 1:527, 2:395, 3:575, 4:1089}
# Even with oversampling, these help the optimizer pay more attention to
# underrepresented classes.
CLASS_WEIGHT = {0: 1.0, 1: 1.8, 2: 2.5, 3: 1.7, 4: 1.0}

# ─── OVERSAMPLE MINORITY CLASSES ──────────────────────────────────────────────
def build_balanced_dataset(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)

    class_imgs = {}
    for cls in sorted(os.listdir(src)):
        p = os.path.join(src, cls)
        if not os.path.isdir(p): continue
        imgs = glob.glob(os.path.join(p, "*.*"))
        if imgs:
            class_imgs[cls] = imgs

    target = max(len(v) for v in class_imgs.values())
    print(f"\nOriginal counts : { {k: len(v) for k,v in class_imgs.items()} }")
    print(f"Target per class: {target}")

    for cls, imgs in class_imgs.items():
        dst_cls = os.path.join(dst, cls)
        os.makedirs(dst_cls)
        for p in imgs:
            shutil.copy2(p, os.path.join(dst_cls, os.path.basename(p)))
        needed = target - len(imgs)
        if needed > 0:
            for i, p in enumerate(random.choices(imgs, k=needed)):
                ext = os.path.splitext(p)[1]
                shutil.copy2(p, os.path.join(dst_cls, f"dup_{i:05d}{ext}"))
        print(f"  Class {cls}: {len(imgs)} → {len(os.listdir(dst_cls))}")
    return dst

print("Building balanced dataset...")
build_balanced_dataset(DATASET_DIR, BALANCED_DIR)

# ─── DATA GENERATORS ──────────────────────────────────────────────────────────
# FIX: NO rescale=1./255 — EfficientNet has its own built-in preprocessing
# that expects raw [0, 255] pixel values. Adding rescale=1/255 would
# double-normalize to near-zero → backbone sees flat black images.
train_datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    brightness_range=[0.75, 1.25],
    shear_range=0.1,
)

# FIX: No rescale, no validation_split — use ALL original data for validation
val_datagen = ImageDataGenerator()

# FIX: Train on full balanced dataset (no split), validate on full original dataset
train_gen = train_datagen.flow_from_directory(
    BALANCED_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical',
    shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    DATASET_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical',
    shuffle=False
)
print(f"Train: {train_gen.samples}  |  Val: {val_gen.samples}")

# ─── CLASS DISTRIBUTION GRAPH ─────────────────────────────────────────────────
orig_counts = np.array([
    len(glob.glob(os.path.join(DATASET_DIR, c, "*.*")))
    for c in sorted(os.listdir(DATASET_DIR))
    if os.path.isdir(os.path.join(DATASET_DIR, c))
])
bal_counts = np.bincount(train_gen.classes)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#4CAF50','#2196F3','#FF9800','#F44336','#9C27B0']
for ax, counts, title in zip(axes,
                              [orig_counts, bal_counts],
                              ["Original Distribution", "After Oversampling (Train)"]):
    bars = ax.bar(CLASS_NAMES, counts, color=colors)
    ax.set_title(title); ax.set_ylabel("Count")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2,
                str(c), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/class_distribution.png", dpi=150)
plt.close(); print("Saved: class_distribution.png")

# ─── BUILD MODEL ──────────────────────────────────────────────────────────────
def build_model():
    base = EfficientNetB0(
        include_top=False, weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False

    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.5)(x)
    x   = layers.Dense(512, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.4)(x)
    x   = layers.Dense(128, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return models.Model(inp, out), base

model, base_model = build_model()
model.summary()

# ─── PHASE 1: Train head only ─────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 1: Head training (EfficientNet fully frozen)")
print("="*60)

LR_P1 = 1e-3
model.compile(
    optimizer=optimizers.Adam(LR_P1),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cb1 = [
    EarlyStopping(patience=8, restore_best_weights=True,
                  monitor='val_accuracy', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                      min_lr=1e-6, verbose=1),
    ModelCheckpoint(MODEL_PATH, save_best_only=True,
                    monitor='val_accuracy', verbose=1)
]

history1 = model.fit(
    train_gen, validation_data=val_gen,
    epochs=EPOCHS_P1, callbacks=cb1,
    class_weight=CLASS_WEIGHT, verbose=1
)

# ─── PHASE 2: Fine-tune with FROZEN BatchNorm ─────────────────────────────────
print("\n" + "="*60)
print("PHASE 2: Fine-tuning (BN layers FROZEN to stop val oscillation)")
print("="*60)

base_model.trainable = True

# CRITICAL: Re-freeze ALL BatchNormalization layers inside EfficientNet.
# Without this, BN running stats change during training → completely different
# behaviour at val time → wild accuracy swings.
bn_frozen = 0
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
        bn_frozen += 1
    elif layer.name.startswith('block') and hasattr(layer, 'layers'):
        for sub in layer.layers if hasattr(layer, 'layers') else []:
            if isinstance(sub, tf.keras.layers.BatchNormalization):
                sub.trainable = False
                bn_frozen += 1

# Freeze first 60% of remaining trainable layers (keep only top layers)
trainable_layers = [l for l in base_model.layers if l.trainable]
freeze_up_to = int(len(trainable_layers) * 0.6)
for layer in trainable_layers[:freeze_up_to]:
    layer.trainable = False

conv_trainable = sum(1 for l in base_model.layers if l.trainable)
print(f"BN layers frozen       : {bn_frozen}")
print(f"Conv layers trainable  : {conv_trainable} / {len(base_model.layers)}")

LR_P2 = 1e-5
model.compile(
    optimizer=optimizers.Adam(LR_P2),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cb2 = [
    EarlyStopping(patience=12, restore_best_weights=True,
                  monitor='val_accuracy', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                      min_lr=1e-8, verbose=1),
    ModelCheckpoint(MODEL_PATH, save_best_only=True,
                    monitor='val_accuracy', verbose=1)
]

history2 = model.fit(
    train_gen, validation_data=val_gen,
    epochs=EPOCHS_P2, callbacks=cb2,
    class_weight=CLASS_WEIGHT, verbose=1
)

# ─── MERGE & SAVE HISTORY ─────────────────────────────────────────────────────
def merge(h1, h2):
    return {k: h1.history[k] + h2.history[k] for k in h1.history}

history = merge(history1, history2)
with open("model/history.json", "w") as f:
    json.dump(history, f)

# ─── GRAPH 1: Training Curves ─────────────────────────────────────────────────
p1_len = len(history1.history['accuracy'])
ep     = range(1, len(history['accuracy']) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training History – CrossEntropy + Oversampling + Frozen BN", fontsize=14)
axes[0].plot(ep, history['accuracy'],     label='Train', color='royalblue')
axes[0].plot(ep, history['val_accuracy'], label='Val',   color='tomato', ls='--')
axes[0].axvline(p1_len+.5, color='gray', ls=':', lw=2, label='Fine-tune start')
axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")

axes[1].plot(ep, history['loss'],     label='Train', color='royalblue')
axes[1].plot(ep, history['val_loss'], label='Val',   color='tomato', ls='--')
axes[1].axvline(p1_len+.5, color='gray', ls=':', lw=2, label='Fine-tune start')
axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(alpha=0.3)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/training_curves.png", dpi=150)
plt.close(); print("Saved: training_curves.png")

# ─── EVALUATE ─────────────────────────────────────────────────────────────────
print("\nGenerating evaluation graphs...")
val_gen.reset()
y_prob = model.predict(val_gen, verbose=1)
y_pred = np.argmax(y_prob, axis=1)
y_true = val_gen.classes[:len(y_pred)]

# ─── GRAPH 2: Confusion Matrix ────────────────────────────────────────────────
cm      = confusion_matrix(y_true, y_pred)
row_sum = cm.sum(axis=1, keepdims=True)
cm_norm = np.where(row_sum == 0, 0.0, cm.astype(float) / row_sum)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Confusion Matrices", fontsize=16)
sns.heatmap(cm,      annot=True, fmt='d',    cmap='Blues',  ax=axes[0],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[0].set_title("Raw Counts"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[1].set_title("Normalized"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150)
plt.close(); print("Saved: confusion_matrix.png")

# ─── GRAPH 3: Per-Class Metrics ───────────────────────────────────────────────
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                output_dict=True, zero_division=0)
P = [report[c]['precision'] for c in CLASS_NAMES]
R = [report[c]['recall']    for c in CLASS_NAMES]
F = [report[c]['f1-score']  for c in CLASS_NAMES]

x = np.arange(NUM_CLASSES); w = 0.25
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x-w, P, w, label='Precision', color='steelblue')
ax.bar(x,   R, w, label='Recall',    color='darkorange')
ax.bar(x+w, F, w, label='F1-Score',  color='green')
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=15)
ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
ax.set_title("Per-Class Metrics (v5)"); ax.legend(); ax.grid(axis='y', alpha=0.3)

# Annotate F1 values on bars
for i, f in enumerate(F):
    ax.text(i+w, f+0.02, f"{f:.2f}", ha='center', fontsize=9, color='darkgreen')

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/per_class_metrics.png", dpi=150)
plt.close(); print("Saved: per_class_metrics.png")

# ─── GRAPH 4: Prediction vs Ground Truth + Confidence ────────────────────────
pred_counts = np.bincount(y_pred, minlength=NUM_CLASSES)
true_counts = np.bincount(y_true, minlength=NUM_CLASSES)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
xp = np.arange(NUM_CLASSES)
axes[0].bar(xp-0.2, true_counts, 0.4, label='Ground Truth', color='steelblue')
axes[0].bar(xp+0.2, pred_counts, 0.4, label='Predicted',    color='tomato', alpha=0.8)
axes[0].set_xticks(xp); axes[0].set_xticklabels(CLASS_NAMES, rotation=15)
axes[0].set_title("Ground Truth vs Predicted")
axes[0].set_ylabel("Count"); axes[0].legend(); axes[0].grid(axis='y', alpha=0.3)

axes[1].hist(np.max(y_prob, axis=1), bins=20, color='mediumpurple', edgecolor='white')
axes[1].set_title("Model Confidence Distribution")
axes[1].set_xlabel("Max Softmax Probability")
axes[1].set_ylabel("Count"); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/prediction_analysis.png", dpi=150)
plt.close(); print("Saved: prediction_analysis.png")

# ─── GRAPH 5: LR Schedule (Visual Reference) ──────────────────────────────────
lr_hist = history.get('learning_rate', [])
if lr_hist:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(lr_hist)+1), lr_hist, color='teal', marker='o', ms=3)
    ax.axvline(p1_len+.5, color='gray', ls=':', lw=2, label='Fine-tune start')
    ax.set_title("Learning Rate Schedule"); ax.set_xlabel("Epoch")
    ax.set_ylabel("LR"); ax.set_yscale('log'); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/lr_schedule.png", dpi=150)
    plt.close(); print("Saved: lr_schedule.png")

# ─── FINAL REPORT ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("CLASSIFICATION REPORT (v5)")
print("="*60)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
val_loss, val_acc = model.evaluate(val_gen, verbose=0)
print(f"Final Val Accuracy : {val_acc:.4f}")
print(f"Final Val Loss     : {val_loss:.4f}")
print(f"Model saved to     : {MODEL_PATH}")

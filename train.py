import tensorflow as tf


full_ds = tf.keras.utils.image_dataset_from_directory(
    "data",
    shuffle=True,
    seed=42,
    image_size=(128, 128),
    batch_size=16
)

class_names = full_ds.class_names

total_batches = tf.data.experimental.cardinality(full_ds).numpy()

train_size = int(0.6 * total_batches)
val_size   = int(0.2 * total_batches)

train_ds = full_ds.take(train_size)
val_ds   = full_ds.skip(train_size).take(val_size)
test_ds  = full_ds.skip(train_size + val_size)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),
    tf.keras.layers.Rescaling(1/255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='linear')
])
# model.summary()
model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

import matplotlib.pyplot as plt

# Plot Training vs Validation Loss & Accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()
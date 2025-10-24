# mnist_cnn_tf.py


# 2. One-hot not required if using sparse_categorical_crossentropy


# 3. Build model
model = models.Sequential([
layers.Input(shape=(28,28,1)),
layers.Conv2D(32, 3, activation='relu'),
layers.Conv2D(64, 3, activation='relu'),
layers.MaxPooling2D(pool_size=(2,2)),
layers.Dropout(0.25),
layers.Flatten(),
layers.Dense(128, activation='relu'),
layers.Dropout(0.5),
layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])


model.summary()


# 4. Train
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)]
model.fit(x_train, y_train, batch_size=128, epochs=12, validation_split=0.1, callbacks=callbacks)


# 5. Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")


# 6. Visualize predictions on 5 samples
import matplotlib.pyplot as plt
indices = np.random.choice(len(x_test), 5, replace=False)
preds = model.predict(x_test[indices])
pred_labels = np.argmax(preds, axis=1)


fig, axes = plt.subplots(1,5, figsize=(12,3))
for ax, img, true, pred in zip(axes, x_test[indices], y_test[indices], pred_labels):
ax.imshow(img.squeeze(), cmap='gray')
ax.set_title(f"T:{true} P:{pred}")
ax.axis('off')
plt.tight_layout()
plt.show()


# Notes: With this architecture and ~10 epochs, expect >99% training accuracy and typically >=98-99% test accuracy.
# To reliably exceed 95% test accuracy, ensure training runs for enough epochs and use dropout/early stopping.
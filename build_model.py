import math
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

# preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/home/jstephen/Desktop/EyeDIseaseDetection/dataset/train',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    '/home/jstephen/Desktop/EyeDIseaseDetection/dataset/test',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

# transfer learning
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))


base_model.trainable = False

# learning rate scheduler
def lr_schedule(epoch):
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 5
    lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

# early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# compilation
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# training
history = model.fit(train_generator, epochs=15, validation_data=test_generator, callbacks=[lr_scheduler, early_stopping])

# evaluation
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')

# save the model
model.save('/home/jstephen/Desktop/EyeDIseaseDetection/dataset/model.h5')
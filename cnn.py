import os

import cv2
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.src.callbacks import LearningRateScheduler, EarlyStopping
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.models import load_model

label_names = ['peace', 'clarity', 'thumbs_down', 'thumbs_up', 'fist']

lr = 1e-3


def lr_schedule(epoch):
    global lr
    if epoch % 3 == 0 and epoch != 0:
        lr *= 0.1  # reduce learning rate after 5 epochs
    return lr


def get_model(train=True):
    if os.path.exists('model2.h5') and train is False:
        print("Loading existing model")
        return load_model('model2.h5')

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(512, 512, 1), activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(5, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    return model


def train_model(model, x_train, y_train, x_test, y_test, save_name):
    lr_scheduler = LearningRateScheduler(lr_schedule)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test),
              callbacks=[lr_scheduler, early_stopping])
    model.save(save_name)
    return model


def test_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def predict(model, image):
    image = np.array(image).reshape(1, 512, 512, 1)
    prediction = model.predict(image, verbose=0)
    return label_names[np.argmax(prediction)]


def main():
    # read all the directories in dataset
    directories = os.listdir("dataset")
    print(directories)

    images = []
    labels = []
    for category_id, category in enumerate(directories):
        category_path = os.path.join("dataset", category)
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.array(img).reshape(512, 512, 1)
            images.append(img)
            labels.append(category_id)

    images = np.array(images)
    labels = np.array(labels).reshape(-1)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels,
                                                        shuffle=True)

    model = get_model(train=True)
    model = train_model(model, X_train, y_train, X_test, y_test, save_name="model2.h5")

    test_model(model, X_test, y_test)


if __name__ == "__main__":
    main()

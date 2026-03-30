import data as data_module
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def prepare_data(module):
    """Load arrays from the provided module and prepare them for training.

    Expects the module to provide: x_train, y_train, x_valid, y_valid.
    Returns: (x_train, y_train), (x_valid, y_valid)
    """
    x_train = getattr(module, 'x_train')
    y_train = getattr(module, 'y_train')
    x_valid = getattr(module, 'x_valid')
    y_valid = getattr(module, 'y_valid')

    x_train = x_train.astype('float32')
    x_valid = x_valid.astype('float32')

    if x_train.ndim == 3:
        x_train = x_train[..., None]
    if x_valid.ndim == 3:
        x_valid = x_valid[..., None]

    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_valid = tf.keras.utils.to_categorical(y_valid, num_classes)

    return (x_train, y_train), (x_valid, y_valid)


def build_model(input_shape=(28, 28, 1), num_classes=10):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.SGD(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train(epochs, save_path=None):
    (x_train, y_train), (x_valid, y_valid) = prepare_data(data_module)
    model = build_model(input_shape=x_train.shape[1:], num_classes=y_train.shape[1])
    model.summary()
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_valid, y_valid))

    if save_path:
        # save the whole model (architecture + weights + optimizer state)
        model.save(save_path)

    return model, history


if __name__ == '__main__':
    # TensorFlow will add the correct extension for the saved model format
    save_file = 'saved_model.h5'
    train(epochs=15, save_path=save_file)


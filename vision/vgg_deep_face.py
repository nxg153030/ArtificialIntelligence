from tensorflow.python.keras import Sequential, layers


# Original paper: https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf
class VGGDeepFace:

    def __init__(self, classes=2, include_top=True):
        model = Sequential()
        # block 1
        model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1_1',
                                input_shape=(224, 224, 3)))
        model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1_2'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1'))

        # block 2
        model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2_1'))
        model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2_2'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2'))

        # block 3
        model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3_1'))
        model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3_2'))
        model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3_3'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3'))

        # block 4
        model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4_1'))
        model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4_2'))
        model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4_3'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool4'))

        # block 5
        model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv5_1'))
        model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv5_2'))
        model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv5_3'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5'))

        # Classification block
        if include_top:
            model.add(layers.Conv2D(4096, (7, 7), activation='relu', name='fc6'))
            model.add(layers.Conv2D(4096, (1, 1), activation='relu', name='fc7'))
            model.add(layers.Conv2D(2622, (1, 1), activation='relu', name='fc8'))
            model.add(layers.Flatten(name='Flatten'))
            model.add(layers.Dense(classes, activation='softmax', name='predictions'))
        self.model = model







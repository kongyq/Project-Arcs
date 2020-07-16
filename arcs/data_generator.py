import numpy as np
from tensorflow import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, input_list1, input_list2, labels, batch_size=32, dim=1000,
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list1 = input_list1
        self.list2 = input_list2
        self.shuffle = shuffle

        self.indexes = np.arange(len(self.labels))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list1) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty(self.batch_size, dtype=int)

        output1 = np.array([self.list1[i] for i in indexes], dtype='float32')
        output2 = np.array([self.list2[i] for i in indexes], dtype='float32')

        return [output1, output2], np.array([self.labels[i] for i in indexes], dtype=int)
        #
        # # Generate data
        # for i, ID in enumerate(indexes):
        #     # Store sample
        #     X[i,] = np.load('data/' + ID + '.npy')
        #
        #     # Store class
        #     y[i] = self.labels[ID]
        #
        # return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
import keras.callbacks
import keras.layers as KL
from keras import Model
from keras.optimizers import adam_v2


class Plain5(object):
    def __init__(self, model_path=None, input_shape=None):
        self.model = None
        self.input_shape = input_shape
        if model_path is not None:
            # TODO: loading from the file
            pass
        else:
            self.model = self.build_model()

    def build_model(self):
        input_layer = KL.Input(self.input_shape, name='input')
        x = KL.Conv1D(8, 3, padding='same', name='Conv1')(input_layer)
        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv1D(8, 3, padding='same', name='Conv2')(x)
        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv1D(8, 3, padding='same', name='Conv3')(x)
        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)

        x = KL.Dense(20, activation='relu', name='dense')(x)
        x = KL.Dense(1, activation='sigmoid', name='output')(x)
        model = Model(input_layer, x)
        return model

    def fit(self, x, y, x_val, y_val, epoch, batch_size):
        self.model.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=0.01 * (batch_size / 256)))
        checkpoint = keras.callbacks.ModelCheckpoint(filepath='checkpoints/plain5.hdf5', monitor='val_loss',
                                                     mode="min", save_best_only=True)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                   patience=1000, verbose=0, mode='auto')
        lr_decay = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, min_delta=1e-6)
        callbacks = [checkpoint, early_stop, lr_decay]
        history = self.model.fit(x, y, validation_data=(x_val, y_val), epochs=epoch, verbose=1,
                                 callbacks=callbacks, batch_size=batch_size)
        return history


class Residual5(object):
    def __init__(self, model_path=None, input_shape=None):
        self.model = None
        self.input_shape = input_shape
        if model_path is not None:
            # TODO: loading from the file
            pass
        else:
            self.model = self.build_model()

    def build_model(self):
        input_layer = KL.Input(self.input_shape, name='input')
        fx = KL.Conv1D(8, 3, padding='same', name='Conv1')(input_layer)
        fx = KL.BatchNormalization()(fx)
        x = KL.Activation('relu')(fx)

        fx = KL.Conv1D(8, 3, padding='same', name='Conv2')(x)
        fx = KL.BatchNormalization()(fx)
        fx = KL.Activation('relu')(fx)
        x = fx + x

        fx = KL.Conv1D(8, 3, padding='same', name='Conv3')(x)
        fx = KL.BatchNormalization()(fx)
        fx = KL.Activation('relu')(fx)
        x = fx + x

        x = KL.Dense(20, activation='relu', name='dense')(x)
        x = KL.Dense(1, activation='sigmoid', name='output')(x)
        model = Model(input_layer, x)
        return model

    def fit(self, x, y, x_val, y_val, epoch, batch_size):
        self.model.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=0.01 * (batch_size / 256)))
        checkpoint = keras.callbacks.ModelCheckpoint(filepath='checkpoints/res5.hdf5', monitor='val_loss',
                                                     mode="min", save_best_only=True)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                   patience=1000, verbose=0, mode='auto')
        lr_decay = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, min_delta=1e-6)
        callbacks = [checkpoint, early_stop, lr_decay]
        history = self.model.fit(x, y, validation_data=(x_val, y_val), epochs=epoch, verbose=1,
                                 callbacks=callbacks, batch_size=batch_size)
        return history


class ShortCut5(object):
    def __init__(self, model_path=None, input_shape=None):
        self.model = None
        self.input_shape = input_shape
        if model_path is not None:
            # TODO: loading from the file
            pass
        else:
            self.model = self.build_model()

    def build_model(self):
        input_layer = KL.Input(self.input_shape, name='input')
        x_raw = KL.Conv1D(8, 3, padding='same', name='Conv1')(input_layer)
        fx1 = KL.BatchNormalization()(x_raw)
        fx1 = KL.Activation('relu')(fx1)

        fx2 = KL.Conv1D(8, 3, padding='same', name='Conv2')(fx1)
        fx2 = KL.BatchNormalization()(fx2)
        fx2 = KL.Activation('relu')(fx2)

        fx3 = KL.Conv1D(8, 3, padding='same', name='Conv3')(fx2)
        fx3 = KL.BatchNormalization()(fx3)
        fx3 = KL.Activation('relu')(fx3)
        x = KL.Concatenate(axis=2)([x_raw, fx1, fx2, fx3])

        x = KL.Dense(20, activation='relu', name='dense')(x)
        x = KL.Dense(1, activation='sigmoid', name='output')(x)
        model = Model(input_layer, x)
        return model

    def fit(self, x, y, x_val, y_val, epoch, batch_size):
        self.model.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=0.01 * (batch_size / 256)))

        checkpoint = keras.callbacks.ModelCheckpoint(filepath='checkpoints/shortcut5.hdf5', monitor='val_loss',
                                                     mode="min", save_best_only=True)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                   patience=1000, verbose=0, mode='auto')
        lr_decay = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, min_delta=1e-6)
        callbacks = [checkpoint, early_stop, lr_decay]
        history = self.model.fit(x, y, validation_data=(x_val, y_val), epochs=epoch, verbose=1,
                                 callbacks=callbacks, batch_size=batch_size)
        return history


class ShortCut11(object):
    def __init__(self, model_path=None, input_shape=None):
        self.model = None
        self.input_shape = input_shape
        if model_path is not None:
            # TODO: loading from the file
            pass
        else:
            self.model = self.build_model()

    def build_model(self):
        input_layer = KL.Input(self.input_shape, name='input')
        x_raw = KL.Conv1D(8, 3, padding='same', name='Conv1_1')(input_layer)
        x = KL.BatchNormalization()(x_raw)
        x = KL.Activation('relu')(x)
        x = KL.Conv1D(8, 3, padding='same', name='Conv1_2')(x)
        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)
        x = KL.Conv1D(8, 3, padding='same', name='Conv1_3')(x)
        x = KL.BatchNormalization()(x)
        fx1 = KL.Activation('relu')(x)

        x = KL.Conv1D(8, 3, padding='same', name='Conv2_1')(fx1)
        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)
        x = KL.Conv1D(8, 3, padding='same', name='Conv2_2')(x)
        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)
        x = KL.Conv1D(8, 3, padding='same', name='Conv2_3')(x)
        x = KL.BatchNormalization()(x)
        fx2 = KL.Activation('relu')(x)

        x = KL.Conv1D(8, 3, padding='same', name='Conv3_1')(fx2)
        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)
        x = KL.Conv1D(8, 3, padding='same', name='Conv3_2')(x)
        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)
        x = KL.Conv1D(8, 3, padding='same', name='Conv3_3')(x)
        x = KL.BatchNormalization()(x)
        fx3 = KL.Activation('relu')(x)
        x = KL.Concatenate(axis=2)([x_raw, fx1, fx2, fx3])

        x = KL.Dense(200, activation='relu', name='dense1')(x)
        x = KL.Dense(1, activation='sigmoid', name='output')(x)
        model = Model(input_layer, x)
        return model

    def fit(self, x, y, x_val, y_val, epoch, batch_size):
        self.model.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=0.01 * (batch_size / 256)))
        checkpoint = keras.callbacks.ModelCheckpoint(filepath='checkpoints/shortcut11.hdf5', monitor='val_loss',
                                                     mode="min", save_best_only=True)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-6,
                                                   patience=200, verbose=0, mode='auto')
        lr_decay = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                     patience=25, min_delta=1e-6)
        callbacks = [checkpoint, early_stop, lr_decay]
        history = self.model.fit(x, y, validation_data=(x_val, y_val), epochs=epoch, verbose=1,
                                 callbacks=callbacks, batch_size=batch_size)
        return history


class Plain11(object):
    def __init__(self, model_path=None, input_shape=None):
        self.model = None
        self.input_shape = input_shape
        if model_path is not None:
            # TODO: loading from the file
            pass
        else:
            self.model = self.build_model()

    def build_model(self):
        input_layer = KL.Input(self.input_shape, name='input')
        x = KL.Conv1D(8, 3, padding='same', name='Conv1_1')(input_layer)
        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)
        x = KL.Conv1D(8, 3, padding='same', name='Conv1_2')(x)
        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)
        x = KL.Conv1D(8, 3, padding='same', name='Conv1_3')(x)
        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv1D(8, 3, padding='same', name='Conv2_1')(x)
        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)
        x = KL.Conv1D(8, 3, padding='same', name='Conv2_2')(x)
        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)
        x = KL.Conv1D(8, 3, padding='same', name='Conv2_3')(x)
        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv1D(8, 3, padding='same', name='Conv3_1')(x)
        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)
        x = KL.Conv1D(8, 3, padding='same', name='Conv3_2')(x)
        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)
        x = KL.Conv1D(8, 3, padding='same', name='Conv3_3')(x)
        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)

        x = KL.Dense(200, activation='relu', name='dense1')(x)
        x = KL.Dense(1, activation='sigmoid', name='output')(x)
        model = Model(input_layer, x)
        return model

    def fit(self, x, y, x_val, y_val, epoch, batch_size):
        self.model.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=0.01 * (batch_size / 256)))
        checkpoint = keras.callbacks.ModelCheckpoint(filepath='checkpoints/plain11.hdf5', monitor='val_loss',
                                                     mode="min", save_best_only=True)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-6,
                                                   patience=200, verbose=0, mode='auto')
        lr_decay = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                     patience=25, min_delta=1e-6)
        callbacks = [checkpoint, early_stop, lr_decay]
        history = self.model.fit(x, y, validation_data=(x_val, y_val), epochs=epoch, verbose=1,
                                 callbacks=callbacks, batch_size=batch_size)
        return history


if __name__ == '__main__':
    # plain5 = Plain5(model_path=None, input_shape=(1, 102))
    # plain11 = Plain11(model_path=None, input_shape=(1, 102))
    residual5 = Residual5(model_path=None, input_shape=(1, 102))
    short5 = ShortCut5(model_path=None, input_shape=(1, 102))

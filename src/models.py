import tensorflow as tf

def lenet1d():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(5000, 12), dtype='float32'),
        tf.keras.layers.Conv1D(6, kernel_size=5, activation='relu', padding='valid', strides=1),
        tf.keras.layers.AveragePooling1D(pool_size=2, padding='valid'),
        tf.keras.layers.Conv1D(16, kernel_size=5, activation='relu', padding='valid', strides=1),
        tf.keras.layers.AveragePooling1D(pool_size=2, padding='valid', strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dense(7, activation='sigmoid')
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# def lenet1d(input_shape, n_classes):
#     input = tf.keras.Input(shape=input_shape, dtype='float32')
#     x = input

#     x = tf.keras.layers.Conv1D(6, kernel_size=5, activation='relu', padding='valid', strides=1)(x)
#     x = tf.keras.layers.AveragePooling1D(pool_size=2, padding='valid')(x)

#     x = tf.keras.layers.Conv1D(16, kernel_size=5, activation='relu', padding='valid', strides=1)(x)
#     x = tf.keras.layers.AveragePooling1D(pool_size=2, padding='valid', strides=2)(x)

#     x = tf.keras.layers.Dense(120, activation='relu')(x)
#     x = tf.keras.layers.Dense(84, activation='relu')(x)

#     x = tf.keras.layers.Dense(n_classes, activation='sigmoid')(x)

#     model = tf.keras.Model(input, x)
#     return model

def biggerlenet1d(number_of_leads, num_classes, sample_length): 
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(sample_length, number_of_leads), dtype='float32'),
        tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='valid', strides=1),
        tf.keras.layers.AveragePooling1D(pool_size=2, padding='valid'),
        tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='valid', strides=1),
        tf.keras.layers.AveragePooling1D(pool_size=2, padding='valid', strides=2),
        tf.keras.layers.Conv1D(128, kernel_size=5, activation='relu', padding='valid', strides=1),
        tf.keras.layers.AveragePooling1D(pool_size=2, padding='valid', strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])
    
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


from tensorflow.keras import layers
import tensorflow.keras as keras

def vgg_block(input, cnn_units):
    output = input
    output = layers.Conv1D(cnn_units, 3, padding='same', activation='relu')(output)
    output = layers.Conv1D(cnn_units, 3, padding='same', activation='relu')(output)
    output = layers.MaxPooling1D(2, padding='same')(output)
    return output

def crt_net(
        number_of_leads,
        cnn_units=128,
        vgg_blocks=1,
        rnn_units=64,
        transformer_encoders=4,
        att_dim=64,
        att_heads=8,
        fnn_units=64,
        num_classes=6
    ):
    input = layers.Input(shape=(None, number_of_leads))
    output = input

    for _ in range(vgg_blocks):
        output = vgg_block(output, cnn_units)

    output = layers.Bidirectional(layers.GRU(rnn_units, return_sequences=True), merge_mode='sum')(output)

    if transformer_encoders > 0:
        output = output + nlp.layers.SinePositionEncoding(max_wavelength=10000)(output)

        for _ in range(transformer_encoders):
            output = nlp.layers.TransformerEncoder(att_dim, att_heads)(output)

        output = layers.GlobalAveragePooling1D()(output)
        
    output = layers.Dropout(0.2)(output)
    output = layers.Dense(fnn_units, activation='relu')(output)
    output = layers.Dense(fnn_units, activation='relu')(output)

    output = layers.Dense(num_classes, activation='sigmoid')(output)
    model = keras.Model(input, output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


def crt_net_vggnet(n_blocks, cnn_units):
    input = tf.keras.Input(shape=None)
    x = input

    for _ in range(n_blocks):
            x = tf.keras.layers.Conv1D(cnn_units, 3, padding='same', activation='relu')(x)
            x = tf.keras.layers.Conv1D(cnn_units, 3, padding='same', activation='relu')(x)
            x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)

    return tf.keras.Model(input, x)

def crt_net_bigru(rnn_units):
    input = tf.keras.Input(shape=None)
    x = input

    gru_layer = tf.keras.layers.GRU(rnn_units, return_sequences=True)
    x = tf.keras.layers.Bidirectional(gru_layer, merge_mode='sum')(x)

    return tf.keras.Model(input, x)

# def crt_net_transformer():
#     input = tf.keras.Input(shape=None)

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_ffn, d_model, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.fully_connected1 = tf.keras.layers.Dense(d_ffn)
        self.fully_connected2 = tf.keras.layers.Dense(d_model)
        self.activation = tf.keras.layers.ReLU()
    
    def call(self, x):
        x_fc1 = self.fully_connected1(x)
        x_fc2 = self.fully_connected2(x_fc1)
        return self.activation(x_fc2);

class AddNormalization(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x, add_x):
        return self.layer_norm(x + add_x)

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, n_heads, d_keys, d_values, d_model, d_ffn, dropout_rate, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.multihead_attention = tf.keras.layers.MultiHeadAttention(n_heads, d_keys, d_values, d_model)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.add_norm1 = AddNormalization(d_ffn, d_model)
        self.feed_forward = FeedForward(d_ffn, d_model)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.add_norm2 = AddNormalization(d_ffn, d_model)

    def call(self, x, pad_mask, training):
        mha_x = self.multihead_attention(x, x, x, pad_mask)
        mha_x = self.dropout1(mha_x, trainigng=training)
        add_norm_x = self.add_norm1(x, mha_x)
        ffn_x = self.feed_forward(add_norm_x)
        ffn_x = self.dropout2(ffn_x, training=training)
        return self.add_norm2(add_norm_x, ffn_x)



def crt_net_modular(
        classifier_module: tf.keras.Model,
        cnn_module: tf.keras.Model = None,
        rnn_module: tf.keras.Model = None,
        transformer_module: tf.keras.Model = None):
    input = tf.keras.Input(shape=None)
    x = input

    if cnn_module is not None:
        x = cnn_module(x)
    
    if rnn_module is not None:
        x = rnn_module(x)

    if transformer_module is not None:
        x = transformer_module(x)

    output = classifier_module(x)

    model = tf.keras.Model(input, output)
    return model
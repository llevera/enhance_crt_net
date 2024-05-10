import tensorflow as tf
import keras_nlp
#import keras_rkwv

# Example usage on crt_net_original_alt():
#
#   Beat classification on MIT-BIH: 200 samples on 2 leads
#     [batch, 200,   2]
#     [batch, 100, 128] After single VGGBlock (length downsampled, feature dimension increased)
#     [batch, 100, 256] After BiGRU and concat (feature dimension doubled)
#     [batch, 100, 256] After transformer encoder stack
#     [batch,      256] After global pooling (remove sequence dimension)
#     [batch,       10] After final dense layer
#     [batch,        5] After output layer
#
#   Rhythm classification on CPSC: 3000 samples on 12 leads
#     [batch, 3000,  12]
#     [batch, 1500, 128] After 1st VGGBlock
#     [batch,  750, 128] After 2nd VGGBlock
#     [batch,  375, 128] After 3rd VGGBlock
#     [batch,  188, 128] After 4th VGGBlock
#     [batch,   94, 128] After 5th VGGBlock
#     [batch,   94, 256] After BiGRU and concat
#     [batch,   94, 256] After transformer encoder stack
#     [batch,       256] After global pooling
#     [batch,        18] After final dense layer
#     [batch,         9] After output layer

# notes:
# removed dense layers and dropout. this was already present for transformers
# use concat merge mode for bidirectional gru.
# increased rnn units to 128 (same as units for cnn)
# transformer feedforward is 256 units not 64
def crt_net_original(
        n_classes,
        n_vgg_blocks=1,
        binary=False,
        use_focal=False,
        metrics=['accuracy', 'f1'],
        d_model=128,
        d_ffn=None
    ):
    """
    n_vgg_blocks: more blocks for more downsampling of signal length.
    - 1 -> single-qrs beats (beat classification)
    - 5 -> multiple beats, e.g. 6s signal (rhythm classification)

    model_compiler_helper args:
    - binary: use for binary classification or multi-label (binary) classification
    - use_focal: use for imbalanced classes
    - see more information in `model_compile_helper()` definition.
    """
    input = tf.keras.Input(shape=None)
    x = input
    x = VGGNet(d_model, n_blocks=n_vgg_blocks)(x)
    x = BiGRU(d_model)(x)
    x = StackedTransformerEncoder(4, 8, d_model * 2 if d_ffn is None else d_ffn)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # x = tf.keras.layers.Dropout(0.2)(x)
    # x = tf.keras.layers.Dense(d_ffn, activation='relu')(x) # These 2 dense layers might not be the actual CRT-Net arch.
    # x = tf.keras.layers.Dense(d_ffn, activation='relu')(x) # Because each transformerencoder has a 2 layer feed-forward network.

    # x = tf.keras.layers.Dense(4 * n_classes, activation='relu')(x) # You might consider using this still to compare.

    model = model_compile_helper(input, x, n_classes, binary, use_focal, metrics)
    return model

def crt_net_original_alt(
        n_classes,
        n_vgg_blocks=1,
        binary=False,
        use_focal=False,
        metrics=['accuracy', 'f1'],
        d_model=128,
        d_ffn=None
    ):
    """
    The provided CRT-Net models.py has some alterations which may be the result of tuning the model:
    - Leaky ReLU (alpha=0.3) activation instead of ReLU.
    - Dropout (rate=0.2) after every VGG block and the BiGRU layer.
    - Sine position encoding uses max position encoding of 2048, instead of default 10000
    - Additional dropout between transformer encoders and global pooling
    - Additional dense layer before output (units=4*n_classes, SeLU activation)
    """
    
    input = tf.keras.Input(shape=None)
    x = input
    x = VGGNet(d_model, n_blocks=n_vgg_blocks, activation=tf.keras.layers.LeakyReLU, activation_kwargs=dict(alpha=0.3), dropout=0.2, use_stride=True)(x)
    x = BiGRU(d_model, activation=tf.keras.layers.LeakyReLU, activation_kwargs=dict(alpha=0.3), dropout=0.2)(x)
    x = StackedTransformerEncoder(4, 8, d_model * 2 if d_ffn is None else d_ffn, max_wavelength=2048)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(4 * n_classes, activation='selu')(x)

    model = model_compile_helper(input, x, n_classes, binary, use_focal, metrics)
    return model

def model_compile_helper(input, x, n_classes, binary, use_focal, metrics):
    """
    Attaches output layer and compiles the model.

    binary: use for binary classification or multi-label (binary) classification
    - False -> Softmax output activation, binary cross entropy loss
    - True -> Sigmoid output activation, categorical cross entropy loss
    
    use_focal: use for imbalanced classes
    - False -> binary_crossentropy or categorical_crossentropy loss
    - True -> binary_focal_crossentropy or categorical_focal_crossentropy loss
    """

    activation = 'softmax'
    loss = 'categorical_focal_crossentropy' if use_focal else 'categorical_crossentropy'

    if binary:
        activation = 'sigmoid'
        loss = 'binary_focal_crossentropy' if use_focal else 'binary_crossentropy'

    output = tf.keras.Layers.Dense(n_classes, activation=activation)(x)
    model = tf.keras.Model(input, output)
    model.compile(optimizer="adam", loss=loss, metrics=metrics)
    return model

class VGGBlock(tf.keras.layers.Layer):
    """
    default:
    - Conv1D -> ReLU -> Conv1D -> ReLU -> MaxPooling1D [-> dropout]

    use_stride: (use large kernel with stride=2 to downsample sequence length, instead of using max pooling)
    - Conv1D -> ReLU -> Conv1D -> ReLU -> Conv1D(kernel 24, stride 2) -> ReLU [-> dropout]
    """
    def __init__(
            self,
            d_cnn,
            activation=tf.keras.layers.ReLU,
            activation_kwargs=dict(),
            dropout=0.0,
            use_stride=False,
            **kwargs
        ):
        super(VGGBlock, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv1D(d_cnn, 3, padding='same')
        self.activation1 = activation(**activation_kwargs)
        self.conv2 = tf.keras.layers.Conv1D(d_cnn, 3, padding='same')
        self.activation2 = activation(**activation_kwargs)
        self.pooling = (tf.keras.layers.Conv1D(d_cnn, 24, strides=2, padding='same')
            ) if use_stride else tf.keras.layers.MaxPooling1D(2, padding='same')
        self.activation3 = activation(**activation_kwargs) if use_stride else None
        self.dropout = tf.keras.layers.Dropout(dropout) if dropout > 0.0 else None

    def call(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.pooling(x)
        if self.activation3 is not None:
            x = self.activation3(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
    
class VGGNet(tf.keras.layers.Layer):
    """
    n_blocks=1
    - (sig len: 200) -> VGGBlock -> (sig len: 100)

    n_blocks=5
    - (sig len: 1250) -> VGGBlock -> VGGBlock -> VGGBlock -> VGGBlock -> VGGBlock -> (sig len: 40)
    """
    def __init__(
            self,
            d_cnn,
            n_blocks=1,
            activation=tf.keras.layers.ReLU,
            activation_kwargs=dict(),
            dropout=0.0,
            use_stride=False,
            **kwargs
        ):
        super(VGGNet, self).__init__(**kwargs)
        self.blocks = [ BiGRU(
            d_cnn / (2**i),
            activation=activation,
            activation_kwargs=activation_kwargs,
            dropout=dropout,
            use_stride=use_stride
            ) for i in range(n_blocks) ]

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class BiGRU(tf.keras.layers.Layer):
    """
    merge_mode=concat doubles feature space (d_rnn: 128 -> out dim: 256)
    - i.e. 128 features are forward GRU pass, 128 are backward GRU pass.
    """
    def __init__(
            self,
            d_rnn,
            return_sequences=True,
            merge_mode='concat',
            activation=tf.keras.layers.ReLU,
            activation_kwargs=dict(),
            dropout=0.0,
            **kwargs
        ):
        super(BiGRU, self).__init__(**kwargs)
        self.gru = tf.keras.layers.GRU(d_rnn, return_sequences=return_sequences)
        self.bigru = tf.keras.layers.Bidirectional(self.gru, merge_mode=merge_mode)
        self.activation = activation(**activation_kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout) if dropout > 0.0 else None

    def call(self, x):
        x = self.bigru(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class StackedTransformerEncoder(tf.keras.layers.Layer):
    """
    n_blocks=4
    - x -> x + SinePositionEncoding -> TransformerEncoder -> TransformerEncoder -> TransformerEncoder -> TransformerEncoder

    TransformerEncoder
    - x -> MultiHeadAttention -> Dropout -> x + Normalization -> x' -> FeedForward -> Dropout -> x' + Normalization
    """
    def __init__(self, n_blocks, n_heads, d_ffn, max_wavelength=10000, **kwargs):
        super(StackedTransformerEncoder, self).__init__(**kwargs)
        self.pos_encoding = keras_nlp.layers.SinePositionEncoding(max_wavelength=max_wavelength)
        self.transformers = [
            keras_nlp.layers.TransformerEncoder(d_ffn, n_heads) for _ in range(n_blocks)
        ]

    def call(self, x):
        x = x + self.pos_encoding(x)
        for transformer in self.transformers:
            x = transformer(x)
        return x

# class FeedForward(tf.keras.layers.Layer):
#     def __init__(self, d_ffn, d_model, **kwargs):
#         super(FeedForward, self).__init__(**kwargs)
#         self.fully_connected1 = tf.keras.layers.Dense(d_ffn)
#         self.fully_connected2 = tf.keras.layers.Dense(d_model)
#         self.activation = tf.keras.layers.ReLU()
    
#     def call(self, x):
#         x_fc1 = self.fully_connected1(x)
#         x_fc2 = self.fully_connected2(x_fc1)
#         return self.activation(x_fc2);

# class AddNormalization(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(AddNormalization, self).__init__(**kwargs)
#         self.layer_norm = tf.keras.layers.LayerNormalization()

#     def call(self, x, add_x):
#         return self.layer_norm(x + add_x)

# class TransformerEncoder(tf.keras.layers.Layer):
#     def __init__(self, n_heads, d_keys, d_values, d_model, d_ffn, dropout_rate, **kwargs):
#         super(TransformerEncoder, self).__init__(**kwargs)
#         self.multihead_attention = tf.keras.layers.MultiHeadAttention(n_heads, d_keys, d_values, d_model)
#         self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
#         self.add_norm1 = AddNormalization(d_ffn, d_model)
#         self.feed_forward = FeedForward(d_ffn, d_model)
#         self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
#         self.add_norm2 = AddNormalization(d_ffn, d_model)

#     def call(self, x, pad_mask, training):
#         mha_x = self.multihead_attention(x, x, x, pad_mask)
#         mha_x = self.dropout1(mha_x, training=training)
#         add_norm_x = self.add_norm1(x, mha_x)
#         ffn_x = self.feed_forward(add_norm_x)
#         ffn_x = self.dropout2(ffn_x, training=training)
#         return self.add_norm2(add_norm_x, ffn_x)

import tensorflow as tf
import keras_nlp
import tensorflow_probability # just making sure this is installed
from src.keras_rwkv.models.v4 import RwkvBackboneModified
from tensorflow.keras import backend as K
from keras.metrics import F1Score


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
        input_shape,
        n_vgg_blocks=1,
        binary=False,
        use_focal=False,
        metrics=['accuracy', 'f1_score'],
        d_model=128,
        d_ffn=None,
        learning_rate=0.001,
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
    input = tf.keras.Input(shape=input_shape)
    x = input
    x = VGGNet(d_model, n_blocks=n_vgg_blocks)(x)
    x = BiGRU(d_model)(x)
    x = StackedTransformerEncoder(4, 8, d_model * 2 if d_ffn is None else d_ffn)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(2 * n_classes, activation='relu')(x) # You might consider using this still to compare.

    model = model_compile_helper(input, x, n_classes, binary, use_focal, metrics, learning_rate)
    return model

def crt_net_original_alt(
        n_classes,
        input_shape,
        n_vgg_blocks=1,
        binary=False,
        use_focal=False,
        metrics=['accuracy', 'f1_score'],
        d_model=128,
        d_ffn=None,
        learning_rate=0.001,
    ):
    """
    The provided CRT-Net models.py has some alterations which may be the result of tuning the model:
    - Leaky ReLU (alpha=0.3) activation instead of ReLU.
    - Dropout (rate=0.2) after every VGG block and the BiGRU layer.
    - Sine position encoding uses max position encoding of 2048, instead of default 10000
    - Additional dropout between transformer encoders and global pooling
    - Additional dense layer before output (units=2*n_classes, SeLU activation)
    """
    
    input = tf.keras.Input(shape=input_shape)
    x = input
    x = VGGNet(d_model, n_blocks=n_vgg_blocks, activation=tf.keras.layers.LeakyReLU, activation_kwargs=dict(alpha=0.3), dropout=0.2, use_stride=True)(x)
    x = BiGRU(d_model, activation=tf.keras.layers.LeakyReLU, activation_kwargs=dict(alpha=0.3), dropout=0.2)(x)
    x = StackedTransformerEncoder(4, 8, d_model * 2 if d_ffn is None else d_ffn, max_wavelength=2048, dropout=0.2)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(2 * n_classes, activation='selu')(x)

    model = model_compile_helper(input, x, n_classes, binary, use_focal, metrics, learning_rate)
    return model

def crt_net_modular(
        n_classes,
        input_shape,
        cnn_type='vggnet',
        rnn_type='bigru',
        att_type='transformer',
        alternate_arch=False,
        n_vgg_blocks=1,
        rkwv_stack_multiplier=1,
        extra_dense=False,
        use_selu=True,
        binary=False,
        use_focal=False,
        metrics=['accuracy', 'f1_score'],
        d_model=128,
        learning_rate=0.001,
    ):
    input = tf.keras.Input(shape=input_shape)
    x = input

    if cnn_type == 'vggnet':
        if alternate_arch:
            x = VGGNet(d_model, n_blocks=n_vgg_blocks, activation=tf.keras.layers.LeakyReLU, activation_kwargs=dict(alpha=0.3), dropout=0.2, use_stride=True)(x)
        else:
            x = VGGNet(d_model, n_blocks=n_vgg_blocks)(x)
    elif cnn_type == 'squeezenet':
        x = SqueezeNet()(x)
    elif cnn_type == 'cnnsvm':
        raise NotImplementedError()
    elif cnn_type == 'none':
        pass  # No CNN block
        
    if rnn_type == 'bigru':
        if alternate_arch:
            x = BiGRU(d_model, activation=tf.keras.layers.LeakyReLU, activation_kwargs=dict(alpha=0.3), dropout=0.2)(x)
        else:
            x = BiGRU(d_model)(x)
    elif rnn_type == 'none':
        pass  # No RNN block
    
    if att_type == 'transformer':
        if alternate_arch:
            x = StackedTransformerEncoder(4, 8, d_model * 2, max_wavelength=2048, dropout=0.2)(x)
        else:
            x = StackedTransformerEncoder(4, 8, d_model * 2)(x)
    elif att_type == 'our_transformer':
        if alternate_arch:
            x = StackedTransformerEncoderCustom(4, 8, d_model * 2, max_wavelength=2048, dropout=0.2)(x)
        else:
            x = StackedTransformerEncoderCustom(4, 8, d_model * 2)(x)
    elif att_type == 'rwkv':
        if alternate_arch:
            x = StackedRWKV(n_blocks=4 * rkwv_stack_multiplier, d_ffn=d_model * 2, dropout=0.2)(x)
        else:
            x = StackedRWKV(n_blocks=4 * rkwv_stack_multiplier, d_ffn=d_model * 2)(x)
    elif att_type == 'none':
        pass  # No Transformer block


    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    dense_activation = 'selu' if alternate_arch and use_selu else 'relu'

    if extra_dense:
        x = tf.keras.layers.Dense(2 * d_model, activation=dense_activation)(x)
        x = tf.keras.layers.Dense(d_model, activation=dense_activation)(x)

    x = tf.keras.layers.Dense(2 * n_classes, activation=dense_activation)(x)
    
    model = model_compile_helper(input, x, n_classes, binary, use_focal, metrics, learning_rate)
    return model

def model_compile_helper(input, x, n_classes, binary, use_focal, metrics, learning_rate):
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if binary:
        activation = 'sigmoid'
        loss = 'binary_focal_crossentropy' if use_focal else 'binary_crossentropy'

    output = tf.keras.layers.Dense(n_classes, activation=activation)(x)
    model = tf.keras.Model(input, output)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
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
        self.blocks = [ VGGBlock(
            d_cnn,
            activation=activation,
            activation_kwargs=activation_kwargs,
            dropout=dropout,
            use_stride=use_stride
            ) for _ in range(n_blocks) ]

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
    def __init__(self, n_blocks, n_heads, d_ffn, max_wavelength=10000, dropout=0.0, **kwargs):
        super(StackedTransformerEncoder, self).__init__(**kwargs)
        self.pos_encoding = keras_nlp.layers.SinePositionEncoding(max_wavelength=max_wavelength)
        self.transformers = [
            keras_nlp.layers.TransformerEncoder(d_ffn, n_heads) for _ in range(n_blocks)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout) if dropout > 0.0 else None

    def call(self, x):
        x = x + self.pos_encoding(x)
        for transformer in self.transformers:
            x = transformer(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
    
class StackedTransformerEncoderCustom(tf.keras.layers.Layer):
    def __init__(self, n_blocks, n_heads, d_model, max_wavelength=10000, dropout=0.0, **kwargs):
        super(StackedTransformerEncoderCustom, self).__init__(**kwargs)
        self.pos_encoding = keras_nlp.layers.SinePositionEncoding(max_wavelength=max_wavelength)
        self.transformers = [
            TransformerEncoder(d_model, n_heads, dropout=dropout) for _ in range(n_blocks)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout) if dropout > 0.0 else None

    def call(self, x):
        x = x + self.pos_encoding(x)
        for transformer in self.transformers:
            x = transformer(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
    
class StackedRWKV(tf.keras.layers.Layer):
    def __init__(self, n_blocks, d_ffn, max_wavelength=10000, dropout=0.0, **kwargs):
        super(StackedRWKV, self).__init__(**kwargs)
        self.pos_encoding = keras_nlp.layers.SinePositionEncoding(max_wavelength=max_wavelength)
        self.rwkv = RwkvBackboneModified(hidden_dim=d_ffn, num_layers=n_blocks)
        self.dropout = tf.keras.layers.Dropout(dropout) if dropout > 0.0 else None

    def call(self, x):
        x = x + self.pos_encoding(x)
        x = self.rwkv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_ffn, d_model, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.fully_connected1 = tf.keras.layers.Dense(d_ffn)
        self.fully_connected2 = tf.keras.layers.Dense(d_model)
        self.activation = tf.keras.layers.ReLU()
    
    def call(self, x):
        x = self.fully_connected1(x)
        x = self.fully_connected2(x)
        return self.activation(x);

class AddNormalization(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, add_x):
        return self.layer_norm(x + add_x)

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, dropout, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.multihead_attention = tf.keras.layers.MultiHeadAttention(n_heads, d_model, dropout=dropout)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(4 * d_model, d_model)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.add_norm2 = AddNormalization()

    def call(self, x):
        mha_x = self.multihead_attention(x, x)
        mha_x = self.dropout1(mha_x)
        x = self.add_norm1(mha_x, x)

        ffn_x = self.feed_forward(x)
        ffn_x = self.dropout2(ffn_x)
        x = self.add_norm2(ffn_x, x)
        return x

class FireModule(tf.keras.layers.Layer):
    def __init__(self, squeeze_channels, expand_channels, **kwargs):
        super(FireModule, self).__init__(**kwargs)
        self.squeeze = tf.keras.layers.Conv1D(squeeze_channels, 1, activation='relu')
        self.expand1x1 = tf.keras.layers.Conv1D(expand_channels, 1, activation='relu')
        self.expand3x3 = tf.keras.layers.Conv1D(expand_channels, 3, padding='same', activation='relu')

    def call(self, x):
        x = self.squeeze(x)
        return tf.keras.layers.concatenate([self.expand1x1(x), self.expand3x3(x)])


class SqueezeNet(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SqueezeNet, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv1D(96, 7, strides=2, activation='relu', padding='same')
        self.maxpool1 = tf.keras.layers.MaxPooling1D(3, strides=2, padding='same')
        self.fire2 = FireModule(16, 64)
        self.fire3 = FireModule(16, 64)
        self.fire4 = FireModule(32, 128)
        self.maxpool4 = tf.keras.layers.MaxPooling1D(3, strides=2, padding='same')
        self.fire5 = FireModule(32, 128)
        self.fire6 = FireModule(48, 192)
        self.fire7 = FireModule(48, 192)
        self.fire8 = FireModule(64, 256)
        self.maxpool8 = tf.keras.layers.MaxPooling1D(3, strides=2, padding='same')
        self.fire9 = FireModule(64, 256)

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool4(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool8(x)
        x = self.fire9(x)
        return x

def create_crtnet_original(number_of_leads=1, num_classes=5, multilabel=False, learning_rate=0.001):
    tf.keras.backend.clear_session()
    return crt_net_original(
        n_classes=num_classes,
        input_shape=(None,number_of_leads),
        n_vgg_blocks=5, # increased signal length so more CNN blocks to downsample (3000 / 2**5 -> 94)
        binary=multilabel, # set this to true if using multilabel output (disables softmax and categorical cross entropy). CPSC can be multilabel.
        use_focal=True, # addresses significant class imbalance (enables focal cross entropy)
        metrics=['accuracy',F1Score()], # May be better to evaluate on F1 score if using early stopping
        d_model=128, # default feature dim size (d_ffn set to 2*d_model)
        learning_rate=learning_rate
    )

def create_crtnet_our_transformer(number_of_leads=1, num_classes=5, multilabel=False, learning_rate=0.001):
    tf.keras.backend.clear_session()
    return crt_net_modular(
        n_classes=num_classes,
        input_shape=(None,number_of_leads),
        n_vgg_blocks=5, # increased signal length so more CNN blocks to downsample (3000 / 2**5 -> 94)
        binary=multilabel, # set this to true if using multilabel output (disables softmax and categorical cross entropy). CPSC can be multilabel.
        use_focal=True, # addresses significant class imbalance (enables focal cross entropy)
        metrics=['accuracy',F1Score()], # May be better to evaluate on F1 score if using early stopping
        d_model=128, # default feature dim size (d_ffn set to 2*d_model)
        learning_rate=learning_rate,
        alternate_arch=True,
        att_type='our_transformer'
    )

def create_crtnet_dense(number_of_leads=1, num_classes=5, multilabel=False, learning_rate=0.001):
    tf.keras.backend.clear_session()
    return crt_net_modular(
        n_classes=num_classes,
        input_shape=(None,number_of_leads),
        n_vgg_blocks=5, # increased signal length so more CNN blocks to downsample (3000 / 2**5 -> 94)
        binary=multilabel, # set this to true if using multilabel output (disables softmax and categorical cross entropy). CPSC can be multilabel.
        use_focal=True, # addresses significant class imbalance (enables focal cross entropy)
        metrics=['accuracy', F1Score()], # May be better to evaluate on F1 score if using early stopping
        d_model=128, # default feature dim size (d_ffn set to 2*d_model)
        learning_rate=learning_rate,
        alternate_arch=True,
        extra_dense=True,
        att_type='transformer'
    )

def create_crtnet_no_attn(number_of_leads=1, num_classes=5, multilabel=False, learning_rate=0.001):
    tf.keras.backend.clear_session()
    return crt_net_modular(
        n_classes=num_classes,
        input_shape=(None,number_of_leads),
        n_vgg_blocks=5, # increased signal length so more CNN blocks to downsample (3000 / 2**5 -> 94)
        binary=multilabel, # set this to true if using multilabel output (disables softmax and categorical cross entropy). CPSC can be multilabel.
        use_focal=True, # addresses significant class imbalance (enables focal cross entropy)
        metrics=['accuracy',F1Score()], # May be better to evaluate on F1 score if using early stopping
        d_model=128, # default feature dim size (d_ffn set to 2*d_model)
        learning_rate=learning_rate,
        alternate_arch=True,
        att_type='none'
    )

def create_crtnet_dense_noselu(number_of_leads=1, num_classes=5, multilabel=False, learning_rate=0.001):
    tf.keras.backend.clear_session()
    return crt_net_modular(
        n_classes=num_classes,
        input_shape=(None,number_of_leads),
        n_vgg_blocks=5, # increased signal length so more CNN blocks to downsample (3000 / 2**5 -> 94)
        binary=multilabel, # set this to true if using multilabel output (disables softmax and categorical cross entropy). CPSC can be multilabel.
        use_focal=True, # addresses significant class imbalance (enables focal cross entropy)
        metrics=['accuracy',F1Score()], # May be better to evaluate on F1 score if using early stopping
        d_model=128, # default feature dim size (d_ffn set to 2*d_model)
        learning_rate=learning_rate,
        alternate_arch=True,
        extra_dense=True,
        att_type='transformer',
        use_selu=False
    )

def create_crtnet_original_vgg1(number_of_leads=1, num_classes=5, multilabel=False, learning_rate=0.001):
    tf.keras.backend.clear_session()
    return crt_net_original(
        n_classes=num_classes,
        input_shape=(None,number_of_leads),
        n_vgg_blocks=1, # increased signal length so more CNN blocks to downsample (3000 / 2**5 -> 94)
        binary=multilabel, # set this to true if using multilabel output (disables softmax and categorical cross entropy). CPSC can be multilabel.
        use_focal=True, # addresses significant class imbalance (enables focal cross entropy)
        metrics=['accuracy', F1Score()], # May be better to evaluate on F1 score if using early stopping
        d_model=128, # default feature dim size (d_ffn set to 2*d_model)
        learning_rate=learning_rate
    )

def create_crtnet_alternate_vgg1(number_of_leads=1, num_classes=5, multilabel=False, learning_rate=0.001):
    tf.keras.backend.clear_session()
    return crt_net_original_alt(
        n_classes=num_classes,
        input_shape=(None,number_of_leads),
        n_vgg_blocks=1, # increased signal length so more CNN blocks to downsample (3000 / 2**5 -> 94)
        binary=multilabel, # set this to true if using multilabel output (disables softmax and categorical cross entropy). CPSC can be multilabel.
        use_focal=True, # addresses significant class imbalance (enables focal cross entropy)
        metrics=['accuracy', F1Score()], # May be better to evaluate on F1 score if using early stopping
        d_model=128, # default feature dim size (d_ffn set to 2*d_model)
        learning_rate=learning_rate
    )

def create_crtnet_alternate(number_of_leads=1, num_classes=5, multilabel=False, learning_rate=0.001):
    tf.keras.backend.clear_session()
    return crt_net_original_alt(
        n_classes=num_classes,
        input_shape=(None,number_of_leads),
        n_vgg_blocks=5, # increased signal length so more CNN blocks to downsample (3000 / 2**5 -> 94)
        binary=multilabel, # set this to true if using multilabel output (disables softmax and categorical cross entropy). CPSC can be multilabel.
        use_focal=True, # addresses significant class imbalance (enables focal cross entropy)
        metrics=['accuracy', F1Score()], # May be better to evaluate on F1 score if using early stopping
        d_model=128, # default feature dim size (d_ffn set to 2*d_model)
        learning_rate=learning_rate
    )

def create_crtnet_squeezenet(number_of_leads=1, num_classes=5, multilabel=False, learning_rate=0.001):
    tf.keras.backend.clear_session()
    return crt_net_modular(
        n_classes=num_classes,
        input_shape=(None,number_of_leads),
        rkwv_stack_multiplier=4,
        n_vgg_blocks=5, # increased signal length so more CNN blocks to downsample (3000 / 2**5 -> 94)
        binary=multilabel, # set this to true if using multilabel output (disables softmax and categorical cross entropy). CPSC can be multilabel.
        use_focal=True, # addresses significant class imbalance (enables focal cross entropy)
        metrics=['accuracy',F1Score()], # May be better to evaluate on F1 score if using early stopping
        d_model=128, # default feature dim size (d_ffn set to 2*d_model)
        cnn_type='squeezenet',
        alternate_arch=True,
        learning_rate=learning_rate
    )

def create_crtnet_no_cnn(number_of_leads=1, num_classes=5, multilabel=False, learning_rate=0.001):
    tf.keras.backend.clear_session()
    return crt_net_modular(
        n_classes=num_classes,
        input_shape=(None, number_of_leads),
        cnn_type='none',  
        alternate_arch=True,
        n_vgg_blocks=5,
        binary=multilabel,
        use_focal=True,
        metrics=['accuracy', F1Score()],
        d_model=128,
        learning_rate=learning_rate
    )

def create_crtnet_no_rnn(number_of_leads=1, num_classes=5, multilabel=False, learning_rate=0.001):
    tf.keras.backend.clear_session()
    return crt_net_modular(
        n_classes=num_classes,
        input_shape=(None, number_of_leads),
        rnn_type='none',  # No RNN
        alternate_arch=True,
        n_vgg_blocks=5,
        binary=multilabel,
        use_focal=True,
        metrics=['accuracy', F1Score()],
        d_model=128,
        learning_rate=learning_rate
    )

def create_crtnet_just_rwkv(number_of_leads=1, num_classes=5, multilabel=False, learning_rate=0.001):
    tf.keras.backend.clear_session()
    return crt_net_modular(
        n_classes=num_classes,
        input_shape=(None,number_of_leads),
        rkwv_stack_multiplier=4,
        n_vgg_blocks=5, # increased signal length so more CNN blocks to downsample (3000 / 2**5 -> 94)
        binary=multilabel, # set this to true if using multilabel output (disables softmax and categorical cross entropy). CPSC can be multilabel.
        use_focal=True, # addresses significant class imbalance (enables focal cross entropy)
        metrics=['accuracy',F1Score()], # May be better to evaluate on F1 score if using early stopping
        d_model=128, # default feature dim size (d_ffn set to 2*d_model)
        att_type='rwkv',
        rnn_type='none',  # No RNN
        alternate_arch=True,
        learning_rate=learning_rate
    )

def create_crtnet_just_transformer(number_of_leads=1, num_classes=5, multilabel=False, learning_rate=0.001):
    tf.keras.backend.clear_session()
    return crt_net_modular(
        n_classes=num_classes,
        input_shape=(None,number_of_leads),
        rkwv_stack_multiplier=4,
        n_vgg_blocks=5, # increased signal length so more CNN blocks to downsample (3000 / 2**5 -> 94)
        binary=multilabel, # set this to true if using multilabel output (disables softmax and categorical cross entropy). CPSC can be multilabel.
        use_focal=True, # addresses significant class imbalance (enables focal cross entropy)
        metrics=['accuracy',F1Score()], # May be better to evaluate on F1 score if using early stopping
        d_model=128, # default feature dim size (d_ffn set to 2*d_model)
        att_type='transformer',
        rnn_type='none',  
        cnn_type='none',  
        alternate_arch=True,
        learning_rate=learning_rate
    )

def create_crtnet_rwkv(number_of_leads=1, num_classes=5, multilabel=False, learning_rate=0.001):
    tf.keras.backend.clear_session()
    return crt_net_modular(
        n_classes=num_classes,
        input_shape=(None,number_of_leads),
        rkwv_stack_multiplier=4,
        n_vgg_blocks=5, # increased signal length so more CNN blocks to downsample (3000 / 2**5 -> 94)
        binary=multilabel, # set this to true if using multilabel output (disables softmax and categorical cross entropy). CPSC can be multilabel.
        use_focal=True, # addresses significant class imbalance (enables focal cross entropy)
        metrics=['accuracy',F1Score()], # May be better to evaluate on F1 score if using early stopping
        d_model=128, # default feature dim size (d_ffn set to 2*d_model)
        att_type='rwkv',
        alternate_arch=True,
        learning_rate=learning_rate
    )

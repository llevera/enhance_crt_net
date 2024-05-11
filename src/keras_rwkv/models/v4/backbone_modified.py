import os
import copy

from keras_nlp.src.models import Backbone # modified
#from keras_nlp.layers.modeling.reversible_embedding import ReversibleEmbedding # modified
from keras_nlp.src.utils.python_utils import classproperty # modified
#from ...backend import keras # modified
import tensorflow as tf # modified
from tensorflow import keras # modified
from ...layers.block import ChannelMixBlock, HybridMixBlock

@keras.utils.register_keras_serializable("keras_rwkv")
class RwkvBackboneModified(Backbone):  # pylint:disable=abstract-method
    def __init__(
        self,
        # vocabulary_size: int,
        hidden_dim: int,
        num_layers: int,
        ffn_pre: bool = False,
        layer_norm_epsilon: float = 1e-5,
        use_original_cuda_wkv: bool = False,
        parallel_wkv: bool = True,
        **kwargs,
    ):
        # Inputs
        # token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        # padding_mask = keras.Input(shape=(None,), dtype="int32", name="padding_mask")

        # Inputs (modified)
        input = keras.Input(shape=(None, hidden_dim))
        x = input

        # Embed tokens, positions.
        # token_embedding_layer = ReversibleEmbedding(
        #     input_dim=vocabulary_size,
        #     output_dim=hidden_dim,
        #     name="token_embedding",
        #     embeddings_initializer=keras.initializers.RandomUniform(-1e-4, 1e-4),
        #     tie_weights=False,
        # )
        # x = token_embedding_layer(token_ids)
        x = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon, name="input_norm"
        )(x)
        if ffn_pre:
            x = ChannelMixBlock(
                0, num_layers, layer_norm_epsilon=layer_norm_epsilon, name="block0"
            )(x)
        else:
            x = HybridMixBlock(
                0,
                num_layers,
                layer_norm_epsilon=layer_norm_epsilon,
                use_original_cuda_wkv=use_original_cuda_wkv,
                parallel_wkv=parallel_wkv,
                name="block0",
            )(x)
        for i in range(1, num_layers):
            x = HybridMixBlock(
                i,
                num_layers,
                layer_norm_epsilon=layer_norm_epsilon,
                use_original_cuda_wkv=use_original_cuda_wkv,
                parallel_wkv=parallel_wkv,
                name=f"block{i}",
            )(x)
        sequence_output = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon, name="output_norm"
        )(x)

        # Instantiate using Functional API Model constructor
        super().__init__(
            # inputs={
            #     "token_ids": token_ids,
            #     "padding_mask": padding_mask,
            # },
            inputs=input, # (modified)
            outputs=sequence_output,
            **kwargs,
        )

        # All references to `self` below this line
        # self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.ffn_pre = ffn_pre
        # self.token_embedding = token_embedding_layer # (modified)
        self.use_original_cuda_wkv = use_original_cuda_wkv
        self.parallel_wkv = parallel_wkv

    def get_config(self):
        config = super().get_config()
        config.update(
            # vocabulary_size=self.vocabulary_size,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            ffn_pre=self.ffn_pre,
            use_original_cuda_wkv=self.use_original_cuda_wkv,
            parallel_wkv=self.parallel_wkv,
        )
        return config


import tensorflow as tf
from omegaconf import DictConfig


class GRUEncoder(tf.keras.Model):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.enc_emb = tf.keras.layers.Embedding(
            cfg.data.src.vocab_size,
            cfg.model.enc.embed_size
        )
        self.enc_gru = tf.keras.layers.GRU(
            cfg.model.enc.rnn.units,
            return_state=True,
            return_sequences=True,
            recurrent_initializer="glorot_uniform"
        )

    def call(self, src_tokens, state=None, training=False):
        embed_enc = self.enc_emb(src_tokens)
        enc_outputs, enc_states = self.enc_gru(
            embed_enc, initial_state=state
        )
        return enc_outputs, enc_states


class GRUDecoder(tf.keras.Model):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.dec_emb = tf.keras.layers.Embedding(
            cfg.data.src.vocab_size,
            cfg.model.dec.embed_size
        )
        self.dec_gru = tf.keras.layers.GRU(
            cfg.model.dec.rnn.units,
            return_state=True,
            return_sequences=True,
            recurrent_initializer="glorot_uniform"
        )
        self.fc = tf.keras.layers.Dense(cfg.data.tar.vocab_size)

    def call(self, tar_tokens, state=None, training=False):
        embed_dec = self.dec_emb(tar_tokens)
        dec_outputs, dec_states = self.dec_gru(
            embed_dec, initial_state=state
        )
        final_outputs = self.fc(dec_outputs)
        return final_outputs, dec_states, None 
    

def get_stacked_model(cfg: DictConfig):
    if cfg.model.name == "Stacked_RNN":
        encoder = GRUEncoder(cfg)
        decoder = GRUDecoder(cfg)
    else:
        raise NotImplementedError()

    return encoder, decoder

def loss_function(
    real, 
    pred, 
    loss_object=tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
    )
):
    # delete [pad] loss part with masks.
    mask = tf.math.logical_not(
        tf.math.equal(real, 0)
    )
    _loss = loss_object(real, pred)

    mask = tf.cast(mask, dtype=_loss.dtype)
    _loss *= mask

    return tf.reduce_mean(_loss)
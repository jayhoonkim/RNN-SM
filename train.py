import os
import random
import time
import logging

import numpy as np
import tensorflow as tf
from typing import Optional, List, Dict, Tuple
from omegaconf import DictConfig
from utils.config_utils_tf import get_optimizer_element
from conf.made_conf import made_conf
import utils.data_utils as dutils
from model.net import get_stacked_model, loss_function
    
step = 0
steps_per_epoch = 350


@tf.function
def _step(src, tar, enc_hidden, teacher_forcing_ratio=0.5):

    enc_output, enc_hidden = encoder(src, enc_hidden)    

    dec_hidden = enc_hidden

    # add start token
    dec_input = tf.expand_dims(
        [tar_tokenizer. word_index["&"]] * src.shape[0], # multiple with batch_size
        1
    ) # [B, 1]

    outputs = []
    loss = 0

    for t in range(1, tar.shape[1]):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden) # [B, 1, V_SZ]

        outputs.append(predictions[:, 0]) # [B, V_SZ]
        final_outs = tf.argmax(predictions, 2) # [B, 1]

        ground_truth = tf.expand_dims(tar[:, t], 1) # [B, 1]

        loss += loss_function(ground_truth, predictions)

        if random.random() < teacher_forcing_ratio: # teacher forcing case
            dec_input = ground_truth
        else: # no teacher
            dec_input = final_outs

    return loss, outputs

@tf.function
def train_step(src, tar, enc_hidden, teacher_forcing_ratio=0.5):
    with tf.GradientTape() as tape:
        loss, outputs = _step(src, tar, enc_hidden, teacher_forcing_ratio)

    batch_loss = (loss / int(tar.shape[1])) # divide with seq_len

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss, outputs

@tf.function
def eval_step(src, tar, enc_hidden):
    loss, outputs = _step(src, tar, enc_hidden, 0.0)
    batch_loss = (loss / int(tar.shape[1])) # divide with seq_len
    return batch_loss, outputs


if __name__ == '__main__':
    src_tensor, tar_tensor, src_tokenizer, tar_tokenizer, infos = dutils.made_datasets('./data/google-10000-english.txt')
    cfg = made_conf(infos)
    
    dataset = tf.data.Dataset.from_tensor_slices((src_tensor, tar_tensor))
    train_batch_size = cfg.train.train_batch_size
    val_batch_size = cfg.train.val_batch_size
    train_dataloader = dataset.batch(train_batch_size, drop_remainder=True)
    val_dataloader = dataset.batch(val_batch_size, drop_remainder=True)
    
    encoder, decoder = get_stacked_model(cfg)
    optimizer, scheduler = get_optimizer_element(
        cfg.opt.optimizer, cfg.opt.lr_scheduler
    )
    checkpoint_prefix = cfg.log.checkpoint_filepath
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        encoder=encoder,
        decoder=decoder,
    )
    
    
    train_dataloader = dataset.batch(train_batch_size, drop_remainder=True)
    val_dataloader = dataset.batch(val_batch_size, drop_remainder=True)
    val_dataloader = iter(val_dataloader)
    
    for epoch in range(cfg.train.max_epochs):
        start = time.time()
        total_epoch_loss = 0

        for (batch, (cur_src, cur_tar)) in enumerate(
            train_dataloader.take(steps_per_epoch)
        ): # batch iter
            enc_hidden = tf.zeros((
                cfg.train.train_batch_size,
                cfg.model.enc.rnn.units
            ))

            batch_loss, outputs = train_step(cur_src, cur_tar, enc_hidden)
            total_epoch_loss += batch_loss

            if batch % 100 == 0 or steps_per_epoch == batch:
                print("Epoch {} Batch {} Train Loss {:.4f}".format(
                    epoch+1,
                    batch,
                    batch_loss.numpy()
                ))

            step += 1

        # save model per 50 epoch
        if (epoch + 1) % 50 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        train_loss = total_epoch_loss / steps_per_epoch
        print("Epoch {} Train Loss {:.4f}".format(epoch+1, train_loss))

        # validation step
        enc_hidden = tf.zeros((
                cfg.train.val_batch_size,
                cfg.model.enc.rnn.units
        ))
        cur_src, cur_tar = next(val_dataloader)
        val_loss, outputs = eval_step(cur_src, cur_tar, enc_hidden)
        print("Epoch {} Val Loss {:.4f}".format(epoch+1, val_loss))

        # token -> text & logging
        preds = tf.stack(outputs, axis=1)
        preds = tf.argmax(preds, axis=2)
        preds = [p.numpy() for p in preds]

        # save model per 2 epoch
        if (epoch + 1) % 50 == 0:
            src_texts = src_tokenizer.sequences_to_texts(cur_src.numpy())
            tar_texts = tar_tokenizer.sequences_to_texts(cur_tar.numpy())
            pred_texts = tar_tokenizer.sequences_to_texts(preds)
            print(src_texts)
            print(tar_texts)
            print(pred_texts)

        print(f"Time taken for 1 epoch {time.time() - start} sec \n")
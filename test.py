import re
import pickle
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Optional, List, Dict, Tuple
from omegaconf import OmegaConf
from utils.config_utils_tf import get_optimizer_element
from conf.made_conf import made_conf
from model.net import get_stacked_model, loss_function

    
beam_length = 10
arg_ratio = 0.25
input_length = 3

@tf.function
def _step(src, enc_hidden, tar_tokenizer, encoder, decoder):
    enc_output, enc_hidden = encoder(src, enc_hidden)    

    dec_hidden = enc_hidden

    # add start token
    dec_input = tf.expand_dims(
        [tar_tokenizer. word_index["&"]] * src.shape[0], # multiple with batch_size
        1
    ) # [B, 1]

    outputs = []
    loss = 0

    for t in range(1, input_length+2):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden) # [B, 1, V_SZ]

        outputs.append(predictions[:, 0]) # [B, V_SZ]
        final_outs = tf.argmax(predictions, 2) # [B, 1]
        dec_input = final_outs
    return loss, outputs

def predict(sample, cfg, src_tokenizer, tar_tokenizer, encoder, decoder):
    tmp = ''
    pred_texts = ['_']
    result = []
    state = False

    for i in tqdm(range(len(sample)-input_length+1)):
        for get in pred_texts:
            if '!' in get:
                state = True

            if get[:-1].replace(" ", "") == sample[i:i+input_length].lower():
                state = False
                if len(tmp) == 0:
                    tmp = sample[i-1:i+input_length-1].lower() + get.replace(" ", "")[-2]
                else:
                    tmp = tmp + get.replace(" ", "")[-2]
                break
        
        if state:
            result.append(tmp)
            tmp = ''

        enc_hidden = tf.zeros((
                cfg.train.train_batch_size,
                cfg.model.enc.rnn.units
            ))
        tensor = src_tokenizer.texts_to_sequences(sample[i:i+input_length].lower())
        tensor = pad_sequences(tensor, padding="post")      
        tensor = tensor.reshape(1, input_length)
        tensor = tf.concat([tensor] * cfg.train.train_batch_size, 0)
        tensor = tf.convert_to_tensor(tensor)
        _, preds = _step(tensor, enc_hidden, tar_tokenizer, encoder, decoder)

        preds = tf.stack(preds, axis=1)
        values, indices = tf.nn.top_k(preds[0], beam_length)
        values = np.array(values)
        indices = np.array(indices)
        preds = tf.argmax(preds, axis=2)
        preds = [p.numpy() for p in preds]
        
        for i in range(1, values.shape[1]):
            for j in range(values.shape[0]):
                if values[j, 0] < values[j, i] + values[j, i]*arg_ratio:
                    preds[i][j] = indices[j, i]

        pred_texts = tar_tokenizer.sequences_to_texts(preds)
        pred_texts = list(dict.fromkeys(pred_texts))
        
    for idx, val in enumerate(result):
        result[idx] = re.sub("[^a-zA-Z]+", "", val)
    
    return list(filter(None, result))

if __name__ == '__main__':
    # data configuration
    with open("./conf/config.yaml", 'r') as f:
        cfg = OmegaConf.load(f)
        
    samples = []
    answer = []
    with open('./data/test_sample.txt', 'r') as f:
        sample_num = f.readline()
        for i in range(int(sample_num)):
            samples.append(f.readline().rstrip("\n"))
            answer.append(f.readline().rstrip("\n").split(' '))
        
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
    save_path = './result/model-1'
    checkpoint.restore(save_path)
    
    with open('./result/src_tokenizer.pickle', 'rb') as handle:
        src_tokenizer = pickle.load(handle)

    with open('./result/tar_tokenizer.pickle', 'rb') as handle:
        tar_tokenizer = pickle.load(handle)
    
    predicts = []    
    for sample in samples:
        predicts.append(predict(sample, cfg, src_tokenizer, tar_tokenizer, encoder, decoder))
    
    for i in range(len(samples)):
        print(f"input is: {samples[i]}")
        print(f"answer is: {answer[i]}")
        print(f"predict is: {predicts[i]}\n")
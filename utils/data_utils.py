import csv
import re
import pickle
import unicodedata

from typing import Optional, List, Dict
from collections import defaultdict
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

max_cut_length = 5 
use_length = 3
num_examples = 30000

def load_data(path):
    f = open(path, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    dicdata = []
    for line in rdr:
        dicdata.append(line[0])
    f.close()
    return dicdata

def CuttingWord(data):
    seq_data = defaultdict(list)
    
    for word in data:
        wordList = list(word)
        for i in range(2, max_cut_length + 1):
            for j in range(0, len(word)-i+1):
                inputTemp = [''.join(wordList[j:i+j])]

                if j+i >= len(word):
                    inputTemp.append('!'*i)
                else:
                    inputTemp.append(''.join(wordList[j+1:j+i+1]))
                seq_data[i].append(inputTemp)

    seq_data = seq_data[use_length]
    seq_data =  [list(t) for t in set(tuple(element) for element in seq_data)]

    return seq_data

def unicode_to_ascii(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    w = re.sub(r"[^a-zA-Z?.!,_¿']+", " ", w)
    w = w.strip()

    w = "&" + w + "#"
    return w

def create_dataset(path: str, num_examples: Optional[int]=None):
    lines = path
    
    word_pairs = [list(map(preprocess_sentence, w)) for w in path]

    return [ [ i for i in list(x[0])[1:-1] ] for x in word_pairs ], [ [ i for i in x[1] ] for x in word_pairs ]

def tokenize(lang):
    lang_tokenizer = Tokenizer(filters="")
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = pad_sequences(tensor, padding="post")

    return tensor, lang_tokenizer

def load_dataset(path, num_examples=None):
    src_lang, tar_lang = create_dataset(path, num_examples) # en, sp

    src_tensor, src_tokenizer = tokenize(src_lang)
    tar_tensor, tar_tokenizer = tokenize(tar_lang)

    return src_tensor, tar_tensor, src_tokenizer, tar_tokenizer

def made_datasets(path):
    data = load_data(path)
    data = data + ['button', 'test', 'mqtt', 'gw_test', 'paho']
    
    seq_data = CuttingWord(data)
    
    src_tensor, tar_tensor, src_tokenizer, tar_tokenizer = load_dataset(
        seq_data, num_examples
    )

    max_tar_len, max_src_len = tar_tensor.shape[1], src_tensor.shape[1]

    src_vocab_size = len(src_tokenizer.word_index) + 1
    tar_vocab_size = len(tar_tokenizer.word_index) + 1

    with open('./result/src_tokenizer.pickle', 'wb') as handle:
        pickle.dump(src_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./result/tar_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tar_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return src_tensor, tar_tensor, src_tokenizer, tar_tokenizer, \
        [max_tar_len, max_src_len, src_vocab_size, tar_vocab_size]
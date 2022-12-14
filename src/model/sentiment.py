import sys
import os
import re
import torch.nn as nn

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

import numpy as np

def make_embeddings(path):
    embeddings_index = {}
    f = open(path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    return embeddings_index



def embed_sequence(seq, embeddings_index, d):
  ret = np.zeros((len(seq), d))
  for i, word in enumerate(seq):
    emb = embeddings_index.get(clean_str(word))
    if emb is not None:
      ret[i] = emb
    else:
        print('{} not in index'.format(clean_str(word)))
  return ret


class GloveLayer(nn.Module):
    def __init__(self, d, glove_path, max_length=30):
        super().__init__()

        self.d = d
        self.embeddings_index = make_embeddings(glove_path)
        self.maxlen = max_length
    
    def embed_sequence(self, x):
        return embed_sequence(x, self.embeddings_index, self.d)
        
    def forward(self, x):
        emb = np.zeros((len(x), self.maxlen, self.d))
        for i, a in enumerate(x):
            seq = self.embed_sequence(a)
            #PADDING
            l = len(seq)
            if l > self.maxlen:
                seq = seq[0:(self.maxlen - 1)]
            else:  
                padlen = self.maxlen - l
                pad = np.zeros((padlen, self.d))
                print(pad.shape)
                print(seq.shape)
                seq = np.concatenate((seq, pad))
            ###

            emb[i] = seq

        return emb #SHAPE (B, K, D). B batchsize, K maxlen of seq, D dimensions

if __name__ == '__main__':
    #embeddings_index = make_embeddings()
    texts = [['Peanut', 'Butter'], ['Jelly'], ['Time']]

    # for text in texts:
    #     ret = embed_sequence(text)
    #     print(ret)

    layer = GloveLayer(50, os.path.join(os.getcwd(), 'glove.6B.50d.txt'))

    ret = layer(texts)
    print(ret)



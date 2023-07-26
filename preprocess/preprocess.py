import numpy as np
import pandas as pd

def count(text):
    #count words in review
    return len(text.split())
def delta_score(i, k):
    #delta between average user rating and average product rating
    return abs(i - k)
def average_word_length(sentence):
    words = sentence.split()
    return sum(len(word) for word in words) / len(words)
def feature_minus(f1, f2):
    return f1 - f2
def feature_multiply(f1, f2):
    return f1 * f2
def feature_divide(f1, f2):
    if f2==0:
        return f1
    else:
        return f1 / f2
def feature_plus(f1, f2):
    return f1 + f2
def preprocess(path):
    df = pd.read_csv(path, compression='gzip')
    max_range = df.drop(['id1', 'id2', 'id3', 'text','label'], axis=1).shape[1]+1
    for i in range(1, max_range):
        for k in range(1, max_range):
            if i == k:
                pass
            else:
                #adding new numerical features based on existing numerical features
                df['fmin{}'.format(str(i)+str(k))]=df.apply(lambda x: feature_minus(x['f{}'.format(int(i))], 
                                                                                    x['f{}'.format(int(k))]), axis=1)
                df['fplus{}'.format(str(i)+str(k))]=df.apply(lambda x: feature_plus(x['f{}'.format(int(i))], 
                                                                                    x['f{}'.format(int(k))]), axis=1)
                df['fmult{}'.format(str(i)+str(k))]=df.apply(lambda x: feature_multiply(x['f{}'.format(int(i))], 
                                                                                        x['f{}'.format(int(k))]), axis=1)
                df['fdiv{}'.format(str(i)+str(k))]=df.apply(lambda x: feature_divide(x['f{}'.format(int(i))], 
                                                                                     x['f{}'.format(int(k))]), axis=1)
    df['rand_f'] = np.random.rand(df.shape[0]) #random feature for calculating feature importance
    df['w_count'] = df.text.apply(count)
    df['d_score'] = df.apply(lambda x: delta_score(x['f3'], x['f6']), axis=1)
    df['av_w_len'] = df.text.apply(average_word_length)
    return df
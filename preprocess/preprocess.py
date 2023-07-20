import numpy as np
import pandas as pd

def count(text):
    return len(text.split())
def delta_score(i, k):
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
def main():
    df = pd.read_csv('../data/wb_school_task_2.csv.gzip', compression='gzip')
    df['w_count'] = df.text.apply(count)
    df['d_score'] = df.apply(lambda x: delta_score(x['f3'], x['f6']), axis=1)
    df['av_w_len'] = df.text.apply(average_word_length)
    for i in range(1, 9):
        for k in range(1, 9):
            if i == k:
                pass
            else:
                df['fmin{}'.format(str(i)+str(k))]=df.apply(lambda x: feature_minus(x['f{}'.format(int(i))], 
                                                                                    x['f{}'.format(int(k))]), axis=1)
                df['fplus{}'.format(str(i)+str(k))]=df.apply(lambda x: feature_plus(x['f{}'.format(int(i))], 
                                                                                    x['f{}'.format(int(k))]), axis=1)
                df['fmult{}'.format(str(i)+str(k))]=df.apply(lambda x: feature_multiply(x['f{}'.format(int(i))], 
                                                                                        x['f{}'.format(int(k))]), axis=1)
                df['fdiv{}'.format(str(i)+str(k))]=df.apply(lambda x: feature_divide(x['f{}'.format(int(i))], 
                                                                                     x['f{}'.format(int(k))]), axis=1)
    df['rand_f'] = np.random.rand(df.shape[0])            
    return df
if __name__ == '__main__':
    main()
from itertools import chain

def chainer(s, sep='\t'):
    return list(chain.from_iterable(s.str.split(sep)))
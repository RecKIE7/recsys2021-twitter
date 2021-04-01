from itertools import chain

def chainer(s, sep='\t'):
    return list(chain.from_iterable(s.str.split(sep)))

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string], '')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
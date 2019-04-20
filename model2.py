import numpy as np
import data2

#model parameters
hidden_size = 32
seq_length = 10
learning_rate = 1e-1

Wxh = np.random.randn(data.vocab_size, hidden_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(hidden_size, data.vocab_size) * 0.01
bh = np.zeros((1, hidden_size))
by = np.zeros((1, data.vocab_size))


def forward(x,y,hprev):
    xs, ys, hs = {}, {}, {} #Empty dicts to keep related info
    hs[-1] = np.copy(hprev)
    for t in range(len(x)):
        xs[t] = np.zeros((1,data.vocab_size)) # for one-hot-row vector representation
        xs[t][0][inputs[t]] = 1 # placing the t-th input in one-hot-row vector representation
        hs[t] = np.tanh(np.dot(xs[t], Wxh) + np.dot(hs[t-1], Whh) + bh) # hidden state
        ys[t] = np.dot(hs[t], Why) + by #output
    return hs, ys

def loss(hs,ys):
    ps = {}
    for t in range(len(ys)):
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
        loss += -np.log(ps[t][0][targets[t]])

import numpy as np
import model


Wxh, Whh, Why, bh, by = model.init_model() #model parameters



def Loss(inputs, targets, hprev):
    x, h, y, p = {}, {}, {}, {} #Empty dicts
    loss = 0.0

    h[-1] = np.copy(hprev)
     #forward pass
    for t in range(len(inputs)):
        x[t] = np.zeros((1,data.vocab_size)) # for one-hot-row vector representation
        x[t][0][inputs[t]] = 1 # placing the t-th input in one-hot-row vector representation
        h[t] = np.tanh(np.dot(x[t], Wxh) + np.dot(h[t-1], Whh) + bh) # hidden state
        y[t] = np.dot(h[t], Why) + by #output
        p[t] = np.exp(y[t]) / np.sum(np.exp(y[t])) # probabilities for output chars
        loss += -np.log(p[t][0][targets[t]]) # softmax loss

    #bakward pass
    dWxh, dWhh, dWy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhprev = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
    #output probabilities
        dy = np.copy(ps[t])
        dy[0][targets[t]] -= 1 # backprop for y
        dby += dy
        dWy += np.dot(hs[t].T, dy)
        dh = np.dot(dy, Why.T) + dhprev # backprop for h
        dth = (1 - hs[t] ** 2) * dh # backprop for tanh
        dbh += dth #backprop for bh
        dWxh += np.dot(xs[t].T, dth) #backprop for Wxh
        dWhh += np.dot(hs[t-1].T, dth)
        dhprev = np.dot(dth, Whh.T)
    return loss, dWxh, dWhh, dWy, dbh, dby, hs[len(inputs)-1]

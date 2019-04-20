import numpy as np
import data


class Model:
    def _init_(self,hidden_size,seq_length,learning_rate):
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate


#model parameters
#hidden_size = 32
#seq_length = 10
#learning_rate = 1e-1
    def init_model(self):
        Wxh = np.random.randn(data.vocab_size, self.hidden_size) * 0.01 #input to hidden
        Whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01 #input to hidden
        Why = np.random.randn(self.hidden_size, data.vocab_size) * 0.01 #input to hidden
        bh = np.zeros((1, self.hidden_size))
        by = np.zeros((1, data.vocab_size))
        return Wxh, Whh, Why, bh, by


    def Loss(inputs, targets, hprev):
        x, h, y, p = {}, {}, {}, {} #Empty dicts
        loss = 0.0

        h[-1] = np.copy(hprev)
     #forward pass
    for t in range(len(inputs)):
        x[t] = np.zeros((1,data.vocab_size)) # encode in 1-of-k representation (we place a 0 vector as the t-th input)
        x[t][0][inputs[t]] = 1 # Inside that t-th input we use the integer in "inputs" list to  set the correct
        h[t] = np.tanh(np.dot(x[t], Wxh) + np.dot(h[t-1], Whh) + bh) # hidden state
        y[t] = np.dot(h[t], Why) + by # unnormalized log probabilities for next chars
        p[t] = np.exp(y[t]) / np.sum(np.exp(y[t])) # probabilities for next chars
        loss += -np.log(p[t][0][targets[t]]) # softmax (cross-entropy loss)

    #bakward pass
    dWxh, dWhh, dWy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhprev = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
    #output probabilities
        dy = np.copy(ps[t])
        #derive our first gradient
        dy[0][targets[t]] -= 1 # backprop into y

        dby += dy
        dWy += np.dot(hs[t].T, dy)
        dh = np.dot(dy, Why.T) + dhprev # backprop into h

        dth = (1 - hs[t] ** 2) * dh # backprop through tanh nonlinearity
        dbh += dth #derivative of hidden bias

        dWxh += np.dot(xs[t].T, dth) #derivative of input to hidden layer weight
        dWhh += np.dot(hs[t-1].T, dth)
        #derivative of hidden layer to hidden layer weight
        dhprev = np.dot(dth, Whh.T)
    return loss, dWxh, dWhh, dWy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
    x = np.zeros((1, data.vocab_size))
  #customize it for our seed char
    x[0][seed_ix] = 1
  #list to store generated chars
    ixes = []
  #for as many characters as we want to generate
    for t in range(n):
    #a hidden state at a given time step is a function
    #of the input at the same time step modified by a weight matrix
    #added to the hidden state of the previous time step
    #multiplied by its own hidden state to hidden state matrix.
        h = np.tanh(np.dot(x, Wxh) + np.dot(h, Whh) + bh)
    #compute output (unnormalised)
        y = np.dot(h, Why) + by
    ## probabilities for next chars
        p = np.exp(y) / np.sum(np.exp(y))
    #pick one with the highest probability
        ix = np.random.choice(range(data.vocab_size), p=p.ravel())
    #create a vector
        x = np.zeros((1, data.vocab_size))
    #customize it for the predicted char
        x[0][ix] = 1
    #add it to the list
        ixes.append(ix)

    txt = ''.join(data.idx_to_char[ix] for ix in ixes)
    print ('----\n %s \n----' % (txt, ))
hprev = np.zeros((1, self.hidden_size)) # reset RNN memory
        #predict the 200 next characters given 'a'
sample(hprev,data.char_to_idx['a'],200)

#inputs = [data.char_to_idx[ch] for ch in data.book[0: 0 + seq_length]]
#print ("inputs: ", inputs)
#targets = [data.char_to_idx[ch] for ch in data.book[0+1: 0 + seq_length+1]]
#print ("targets: ", targets)



n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0 / data.vocab_size) * seq_length # loss at iteration 0
while n<=1000*100:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  # check "How to feed the loss function to see how this part works
  if p+seq_length+1 >= len(data.book) or n == 0:
      hprev = np.zeros((1, hidden_size)) # reset RNN memory
      p = 0 # go from start of data
  inputs = [data.char_to_idx[ch] for ch in data.book[p:p+seq_length]]

  targets = [data.char_to_idx[ch] for ch in data.book[p+1:p+seq_length+1]]

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = Loss(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001

  # sample from the model now and then
  if n % 1000 == 0:
    print ('iter %d, loss: %f' % (n, smooth_loss)) # print progress
    sample(hprev, inputs[0], 200)

  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter


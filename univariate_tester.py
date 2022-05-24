import numpy as np
import itertools


def s(x):
    return 1 / (1 + np.exp(-x))


# i gate
w_ix = -1.45868564
w_ih = 4.58483267
b_i = 2.26598692

# f gate
w_fx = -5.45733309
w_fh = -1.51896131
b_f = 1.6409204

# o gate
w_ox = -1.30956185
w_oh = 2.43434882
b_o = 4.05669594

# g
w_gx = -4.32242584
w_gh = 2.39521885
b_g = 0.70014334

################################

# The below code runs through all length 14 binary strings and throws an error 
# if the LSTM fails to predict the correct parity

cnt = 0
for X in itertools.product([0, 1], repeat=14):
    c = 0
    h = 0
    cnt += 1
    for x in X:
        i = s(w_ih * h + w_ix * x + b_i)
        f = s(w_fh * h + w_fx * x + b_f)
        g = np.tanh(w_gh * h + w_gx * x + b_g)
        o = s(w_oh * h + w_ox * x + b_o)
        c = f * c + i * g
        h = o * np.tanh(c)
    if np.sum(X) % 2 != int(h > 0.5):
        print("Failure", cnt, X, int(h > 0.5), np.sum(X) % 2 == int(h > 0.5))
        break
    if cnt % 1000 == 0:
        print(cnt)

"""
Shuffle the wikipedia mohameds wikipedia sentences
"""
import random
from time import time

def sec_to_hms(seconds):
    m,s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h,m,s

start = time()
print("(%i:%i:%i) Reading..." % sec_to_hms(time()-start))
sentences = open('data/en.tok.txt', 'r').readlines() 
print("(%i:%i:%i) Shuffling..." % sec_to_hms(time()-start))
random.shuffle(sentences)
print("(%i:%i:%i) Writing..." % sec_to_hms(time()-start))
with open('data/shuffled.en.tok.txt', 'w') as f:
    for sentence in sentences:
        f.write(sentence)
del sentences # make sure it's garbage collected
print("Done. Total time : (%i:%i:%i) hours" % sec_to_hms(time()-start))
"""Data import helpers for Dependency RNN"""
from __future__ import print_function
import numpy as np
import re
import collections
import random
from spacy import English

"""Constant defs"""
nlp = English()
split_delims = [' ', '.',';',':', '%', '"', '$', '^', ',']
label2int = dict() # keep running dictionary of labels

"""Function Defs"""
# drop any data item that contains an oov word
def is_oov(sentence, vocab_set):
    for token in sentence:
        if token.text not in vocab_set:
            return True
    return False

def split(string, delimiters=split_delims, maxsplit=0):
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)

def convert_raw_x(line):
    """Convert raw line of semeval data into a useable form
    
    Convert to a triple of (list(raw words), e1_index, e2_index)
    """
    s = line.strip()
    #print(s)
    s = s[s.index('"')+1: -(s[::-1].index('"')+1)] # get s between first " and last "
    # we will assume that the first token follow the <e1> , <e2> tags are the entity words.  
    # note this is a big assumption and hopefully phrases will be in subtrees or in heads of the parse trees
    # TODO: this can be addressed by making it a 5-tuple with the endpoints also encoded
    s = [ w for w in split(s) if w is not '' ]
    for i in range(len(s)):
        # deal with e1's
        if '<e1>' in s[i]:
            e1_index = i
            s[i] = s[i].replace('<e1>', '')
        if '</e1>' in s[i]:
            #e1_index = i
            s[i] = s[i].replace('</e1>', '')
        # eal with e2's
        if '<e2>' in s[i]:
            e2_index = i
            s[i] = s[i].replace('<e2>', '')
        if '</e2>' in s[i]:
            #e2_index = i
            s[i] = s[i].replace('</e2>', '')
        
    # drop extraneous elements from re.split
    # also turn it into a spacy sentence
    s = nlp(u' '.join([ w.lower() for w in s ])) 
    return (s, e1_index, e2_index)
    
label2int = dict() # keep running dictionary of labels
def convert_raw_y(label_line, label2int):
    """Convert raw line of semeval labels into a useable form (ints)"""
#     print("Raw Y: %r" % line[:])
    line = label_line.strip()
    if line in label2int:
        return label2int[line]
    else:
        label2int[line] = len(label2int.keys())
        return label2int[line]
    
def lookup_inverse_relation(relation_int, int2label, label2int):
    if relation_int in int2label:
        label = int2label[relation_int]
        if label == 'Other':
            return relation_int
        else:
            rel, ex, ey = re.split("\(|,|\)", label)[:3]
            label = ''.join([rel, '(', ey, ',', ex,')'])
            return label2int[label]
    else:
        print("Label with that index doesn't exist")
        
def load_semeval_data():
    """Load in SemEval 2010 Task 8 Training file and return lists of tuples:
    
    Tuple form =  (spacy(stripped sentence), index of e1, index of e2)"""
    training_txt_file = 'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    validation_index = 8000 - 891# len data - len valid - 1 since we start at 0
    train = {'x':[], 'y':[]}
    valid = {'x':[], 'y':[]}
    text = open(training_txt_file, 'r').readlines()
    assert len(text) // 4 == 8000
    for cursor in range(len(text) // 4): # each 4 lines is a datum
        if cursor < validation_index:
            train['x'].append(convert_raw_x(text[cursor*4]))
#             print(cursor, convert_raw_x(text[cursor*4]))
            train['y'].append(convert_raw_y(text[cursor*4 + 1]))
            # ignore comments and blanks (+2, +3)
        else:
            valid['x'].append(convert_raw_x(text[cursor*4]))
            valid['y'].append(convert_raw_y(text[cursor*4 + 1]))
            
    assert len(train['y']) == 7109 and len(valid['y']) == 891
    assert sorted(label2int.values()) == range(19) # 2 for each 9 asymmetric relations and 1 other
    
    return train, valid

def sentence_to_indices(sentence, vocab2int_dict):
    """Convert ONE spacy sentences to list of indices in the vocab"""
    return [ vocab2int_dict[token.text] for token in sentence ]

def sentences_to_indices(sentences, vocab2int_dict):
    """Convert list of spacy sentences to list of indices in the vocab"""
    data = []
    for sentence in sentences:
        data.append(sentence_to_indices(sentence, vocab2int_dict))
    return data

def create_vocab_from_data(sentences, vocab_limit=5000, dep=False, filter_oov=True, print_oov=False):
    """Create a vocab index, inverse index, and multinomial distribution over tokens from a list of spacy sentences
    
    if `dep`=True, return the dependencies instead of the tokens"""
    counts = collections.Counter()
    for sentence in sentences:
        for token in sentence:
            if dep:
                counts[token.dep_] += 1
            else:
                if filter_oov and not token.is_oov and token.text not in [u' ', u'\n\n']:
                    counts[token.text] += 1
                elif not filter_oov and token.text not in [u' ', u'\n\n']:
                    counts[token.text] += 1
                elif print_oov:
                    print("Token %r is oov" % token.text)
                
    # make sure we didn't aim too high
    if vocab_limit > len(counts):
        vocab_limit = len(counts)
    # create the vocab in most common order
    vocab = [ x[0] for x in counts.most_common() ][:vocab_limit]
    counts_ordered = [ x[1] for x in counts.most_common() ][:vocab_limit]
    # calculate the empirical distribution
    unigram_distribution = list(np.array(counts_ordered) / np.sum(counts_ordered, dtype=np.float32))
    # create index and inverted index
    vocab2int = { token:i for (i, token) in enumerate(vocab) }
    int2vocab = { i:token for (token, i) in vocab2int.items() }

    return vocab, vocab2int, int2vocab, unigram_distribution

def dependency_path_to_root(token):
    """Traverse up the dependency tree. Include the token we are tracing"""
    dep_path = [token]
    while token.head is not token:
        dep_path.append(token.head)
        token = token.head
    # dep_path.append(token.head) # add the root node
    return dep_path

def find_common_ancestor(e1_path, e2_path):
    """Loop through both dep paths and return common ancestor"""
    for t1 in e1_path:
        for t2 in e2_path:
            if t1.idx ==  t2.idx:
                return t1
    return None

def convert_semeval_to_sdps(data, labels, vocab2int, dep2int, int2label, label2int, int2vocab,
                            include_deps=False, include_reverse=True, print_check=False):
    """Conver list of (spacy, e1 index, e2 index) into list of lists of shortest dependency path sequences
    
    if `include_deps`, each datum is a tuple of tokens and their dependencies
    """
    sdps = []
    new_labels = []
    for i, (sentence, e1_idx, e2_idx) in enumerate(data):
        e1 = sentence[e1_idx]
        e2 = sentence[e2_idx]
        e1_path = dependency_path_to_root(e1)
        e2_path = dependency_path_to_root(e2)       
        # find common ancestor for both e1 and e2
        # just loop over both, checking if inner is in outer
        # the first token meeting this is the least common ancestor
        common_ancestor = find_common_ancestor(e1_path, e2_path)
        if common_ancestor is  None:
            print("ERROR: This sentence has no common dependency ancestor.  It was probably parsed incorrectly. SKIPPING")
            print(sentence)
            print(list(sentence))
            print(e1_idx, e2_idx, e1_path, e2_path)
            continue
        # assert common_ancestor is not None, "Didn't even find the common root node?"

        fsdp = []
        for token in e1_path:
            if include_deps:
                fsdp.append((vocab2int[token.text], dep2int[token.dep_]))
            else:
                fsdp.append(vocab2int[token.text])
            if token.idx == common_ancestor.idx:
                break
        bsdp = []
        for token in e2_path:
            # this time go up to BUT not including common acestor
            if token.idx == common_ancestor.idx:
                break
            if include_deps:
                bsdp.append((vocab2int[token.text], dep2int[token.dep_]))
            else:
                bsdp.append(vocab2int[token.text])
        
        sdp = fsdp + list(reversed(bsdp)) # reverse the order since we traversed right to left for e2
        sdps.append(sdp)
        new_labels.append(labels[i])
        if include_reverse:
            sdps.append(list(reversed(sdp)))
            new_labels.append(lookup_inverse_relation(labels[i], int2label, label2int))
        if print_check:
            print(sentence)
            print("%r (%i, %i)" % (list(sentence), e1_idx, e2_idx)) 
            print("%r" % [int2vocab[idx] for idx in sdp])
            print("%r" % sdp)
            print("%r" % common_ancestor)
            print("%r, %r" % (fsdp, [int2vocab[e] for e in fsdp]))
            print("%r, %r" % (bsdp, [int2vocab[e] for e in bsdp]))
            print("%r, %r" % (e1_path, [e.idx for e in e1_path]))
            print("%r, %r" % (e2_path, [e.idx for e in e2_path]))
            if include_reverse:
                print("%r" % [int2vocab[idx] for idx in reversed(sdp)])
                print("%r\n" % list(reversed(sdp)))
            print('\n')
    return sdps, new_labels

    # a helper function for negative sampling
def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    """
    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
        s += distribution[i]
        if s >= r:
            return i
    return len(distribution) - 1

def distribution_to_power(distribution, power):
    """Return a distribution, scaled to some power"""
    dist = [ pow(d, power) for d in distribution ]
    dist /= np.sum(dist)
    return dist

""" Calls """
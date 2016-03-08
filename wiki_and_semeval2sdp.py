"""
Data writer script.  Takes Mohamed's tokenized wiki sentences and creates nice data output in three files:

(1) a json dump file of all of the shortest dependency paths and targets
(2) a vocab and unigram distribution file for the tokens
(3) a vocab and unigram distribution file for the dependency labels

"""
from __future__ import print_function
import numpy as np
import os
import getopt
import math
import collections
import json
import click
from time import time

import semeval_data_helper as sdh

from spacy.en import English
nlp = English()

def noun_chunk_to_head_noun(chunk):
    """Given a chunk, find the noun who's head is outside the chunk. This is the head noun"""
    chunk_set = set(list(chunk))
    for token in chunk:
        if token.head not in chunk_set:
            return token
    print("No head noun found in chunk... %r" % chunk.text.lower())
    return None

def sentence_to_chunk_pairs(sentence):
    """Iterate over  sentence generating n choose 2 noun phrase heads"""
    chunk_pairs = []
    noun_chunks = list(sentence.noun_chunks)
    for i, chunk1 in enumerate(noun_chunks[:-1]):
        head1 = noun_chunk_to_head_noun(chunk1)
        if not head1:
            continue # don't let bad noun chunks in
        for chunk2 in noun_chunks[i+1:]:
            head2 = noun_chunk_to_head_noun(chunk2)
            if not head2:
                continue # don't let bad noun chunks in
            chunk_pairs.append((head1, head2))
    return chunk_pairs

def smart_token_to_text(token, lower=True):
    """Convet spacy token to lowercase text and simplify numbers and punctuation"""
    text = token.text.lower() if lower else token.text
    if token.is_punct:
        text = u'<PUNCT>'
    if token.like_num:
        text = u'<NUM>'
    return text

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

def sentence_to_sdps(sentence, min_len=1, max_len=7, verbose=False):
    """Takes sentence and returns all shortest dependency paths (SDP) between pairs of noun phrase heads in a sentence
    
    Args:
        sentence: a spacy Sentence
        min_len (opt): the minimum number of words along the path (not including endpoints)
    
    Returns:
        sdps: a dict with `path` and `target` fields
                where `path` is a list of (word, dep) tuples (where dep is the dependency that word is the head of)
                and `target` is the pair of head nouns at the endpoints of the path
                
    Notes:
        There are three cases of SDPs: 
        (1) there is not dependency path between X and Y. We obviously skip these
        (2) one nominal lies on the path to root of the other (wlog X <- ... <- Y <- ...)
            In this case we will lose one dependency, the one where X is the tail.
                                                                            | 
                                                                            v
        (3) the nominals have common ancestor in the tree (wlog X <- ... <- Z -> ... -> Y)
            In this case we will lose two dependencies, those involving X and Y.
    """
    noun_pairs = sentence_to_chunk_pairs(sentence)
    for X, Y in noun_pairs:
        ### INCLUDE X AND Y AT ENDS WITH PLACEHOLDERS ###
        X_path = dependency_path_to_root(X)
        Y_path = dependency_path_to_root(Y)
        common = find_common_ancestor(X_path, Y_path)
    #     # now we don't want nouns for assembly
    #     X_path = X_path[1:]
    #     Y_path = Y_path[1:]
        # CASE (1)
        if not common:
            if verbose:
                print("Bad SDP for sentence '%r' :: skipping" % sentence)
            continue
        # CASE (2)
        elif X is common:
            sdp = []
            for token in Y_path:        # looks like (Y <- ... <- X <-) ...
                sdp.append((smart_token_to_text(token), token.dep_))
                if token is common:     # stop after X
                    break
            sdp = list(reversed(sdp))   # flip to get ... (-> X -> ... -> Y)
        elif Y is common:
            sdp = []
            for token in X_path:        # looks like (X <- ... <- Y <- ) ...
                sdp.append((smart_token_to_text(token), token.dep_))
                if token is common:     # stop after Y
                      break
        # CASE (3)
        else:
            sdp = []
            for token in (X_path):      # looks like (X <- ... <- Z <-) ...
                sdp.append((smart_token_to_text(token), token.dep_))
                if token is common:     # keep Z this time
                    break
            ysdp = []                   # need to keep track of seperate, then will reverse and extend later
            for token in Y_path:        # looks like (Y <- ... <-) Z <- ... 
                if token is common:     # don't keep Z from this side
                    break
                ysdp.append((smart_token_to_text(token), token.dep_))
            sdp.extend(list(reversed(ysdp))) # looks like (X <- ... <- Z -> ... ) -> Y)
        # convert endpoints of the paths to placeholder X and Y tokens
        sdp[0] = (u'<X>', sdp[0][1])
        sdp[-1] = (u'<Y>', sdp[-1][1])


        ### CASE WHERE WE DONT INCLUDE X AND Y IN PATH.  THIS CAN LEAD TO EMPTY PATHS ###
        # X_path = dependency_path_to_root(X)
        # Y_path = dependency_path_to_root(Y)
        # common = find_common_ancestor(X_path, Y_path) # need nouns to find case (2)
        # # now we don't want nouns for assembly
        # X_path = X_path[1:]
        # Y_path = Y_path[1:]
        # # CASE (1)
        # if not common:
        #     if verbose:
        #         print("Bad SDP for sentence '%r' :: skipping" % sentence)
        #     continue
            
        # # CASE (2)
        # elif X is common:
        #     sdp = []
        #     for token in Y_path:        # looks like Y <- (...) <- X <- ...
        #         if token is common:     # stop before X
        #             break
        #         sdp.append((token.text.lower(), token.dep_))
        #     sdp = list(reversed(sdp))   # flip to get -> X -> (...) -> Y
        # elif Y is common:
        #     sdp = []
        #     for token in X_path:        # looks like X <- ... <- Y 
        #         if token is common:     # stop before Y
        #               break
        #         sdp.append((token.text.lower(), token.dep_))
    
        # # CASE (3)
        # else:
        #     sdp = []
        #     for token in (X_path):      # looks like X <- (... <- Z <-) ...
        #         sdp.append((token.text.lower(), token.dep_))
        #         if token is common:     # keep Z this time
        #             break
        #     ysdp = []                   # need to keep track of seperate, then will reverse and extend later
        #     for token in Y_path:        # looks like (Y <- ... <-) Z <- ... 
        #         if token is common:
        #             break
        #         ysdp.append((token.text.lower(), token.dep_))
        #     sdp.extend(list(reversed(ysdp))) # looks like X <- (... <- Z -> ... ) -> Y
            
        if len(sdp) < min_len or len(sdp) > max_len:
            continue                    # skip ones that are too short
        yield {'path': sdp, 'target':(X.text.lower(), Y.text.lower())}

def create_vocab_from_data(sentences, important_sentences=[], vocab_limit=None, 
                           min_count=None, dep=False, 
                           filter_oov=False, print_oov=False,
                           oov_count=1):
    """Create a vocab index, inverse index, and unigram distribution over tokens from a list of spacy sentences
    
    if `dep`=True, return the dependencies instead of the tokens"""
    counts = collections.Counter()
    for sentence in sentences:
        for token in sentence:
            if dep:
                counts[token.dep_] += 1
            else:
                if filter_oov and not token.is_oov and token.text not in [u' ', u'\n\n']:
                    counts[token.text.lower()] += 1
                elif not filter_oov and token.text not in [u' ', u'\n\n']:
                    counts[token.text.lower()] += 1
                elif print_oov:
                    print("Token %r is oov" % token.text.lower())
     # if we specify important sentences, vocab limits and min frequencies don't apply
    # that way we have total vocab coverage over these sentences
    if important_sentences:
        important_vocab = set()
        for sentence in important_sentences:
            for token in sentence:
                if dep:
                    counts[token.dep_] += 1
                    important_vocab.add(token.dep_)
                else:
                    if filter_oov and not token.is_oov and token.text not in [u' ', u'\n\n']:
                        counts[token.text.lower()] += 1
                        important_vocab.add(token.text.lower())
                    elif not filter_oov and token.text not in [u' ', u'\n\n']:
                        counts[token.text.lower()] += 1
                        important_vocab.add(token.text.lower())
                    elif print_oov:
                        print("Token %r is oov" % token.text.lower())

    counts = counts.most_common()
    if not (vocab_limit or min_count):
        vocab_limit = len(counts)
    elif vocab_limit > len(counts):
        print("Your vocab limit %i was bigger than the number of token types, now it's %i" 
              % (vocab_limit, len(counts)))
        vocab_limit = len(counts)
    elif min_count:
        # get first index of an element that doesn't meet the requency constraint
        vocab_limit = len(counts) # never found something too small
        for i, count in enumerate(map(lambda x:x[1], counts)):
            if count < min_count:
                vocab_limit = i
                break
    # now if we have important sentences
    # we need to add the missing vocabs back to the vocab and increase the size
    if important_sentences:
        missing_important = [count for count in counts[vocab_limit:] if count[0] in important_vocab]
        counts = counts[:vocab_limit] + missing_important
        vocab_limit = len(counts)
        print("Kept %i missing important words" % len(missing_important) )

    # create the vocab in most common order
    # include an <OOV> token and make it's count the sum of all elements that didn't make the cut
    vocab = [ x[0] for x in counts][:vocab_limit] + [u'<OOV>', u'<X>', u'<Y>', u'<NUM>', u'<PUNCT>']
    if not oov_count and vocab_limit < len(vocab): # if we didn't specify a psuedocount, take the real one... probably a bad idea
        oov_count = sum(map(lambda x:x[1], counts[vocab_limit:]))
    freqs = [ x[1] for x in counts ][:vocab_limit] + [oov_count]*5
    # calculate the empirical distribution
    unigram_distribution = list(np.array(freqs) / np.sum(freqs, dtype=np.float32))
    # create index and inverted index
    vocab2int = { token:i for (i, token) in enumerate(vocab) }
    int2vocab = { i:token for (token, i) in vocab2int.items() }

    return vocab, vocab2int, int2vocab, unigram_distribution

def post_process_sdp(sdp):
    """ Filter out unwanted sdps structure """
    bad_tokens = set([u'<PUNCT>']) #set([',', '.', '-', '(', ')', '&', '*', '_', '%', '!', '?', '/', '<', '>', '\\', '[', ']', '{', '}', '"', "'"])
    sdp['path'] = [x for x in sdp['path'] if x[0] not in bad_tokens]
    return sdp

def is_ok_sdp(sdp, int2vocab, oov_percent=75):
    """ Helper function to mak sure SDP isn't a poor example.

    Filters used to identify bas data:
    1. Neither targets may be oov
    2. The relation itself must be less than `oov_percent` percent number of relations
    """
    oov = int2vocab.keys()[-1]
    # print(oov, sdp['target'])
    if sdp['target'][0] == oov or sdp['target'][1] == oov:
        return False
    oov_count = len([ t for t in sdp['path'] if t[0] == oov])
    too_many = int((oov_percent/100.0)*len(sdp['path']))
    if oov_count > too_many:
        return False
    if not sdp['path'] or not sdp['target']:
        return False
    return True

def vocab2idx(token, vocab2int):
    """ Convert a vocab item to it's index accounting for OOV, 
    which is assumed to be the last element of the vocab
    """
    if token in vocab2int:
        return vocab2int[token]
    else:
        return vocab2int[u'<OOV>'] # OOV conversion

def sec_to_hms(seconds):
    """Return triple of (hour,minutes,seconds) from seconds"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

@click.command()
@click.option('-n', '--num_sentences', default=10000, help="Number of sentences to use")
@click.option('-m', '--min_count', default=5, help="Minimum count of a vocab to keep")
@click.option('-v', '--vocab_limit', default=None, help='Number of most common token types to keep. Trumps min_count')
@click.option('-i', '--infile', default='data/shuffled.en.tok.txt', help='Name of Mohammeds parsed wikidump sentences file')
@click.option('-o', '--outfile', default='data/semeval_wiki_sdp_', help='Outfile prefix')
@click.option('--minlen', default=1, help="Minimum length of the dependency path not including nominals")
@click.option('--maxlen', default=7, help="Maximum length of the dependency path not including nominals")
def main(num_sentences, min_count, vocab_limit, infile, outfile, minlen, maxlen):
    FLAGS = {
        'num_sentences': num_sentences, # max is 31661479
        'min_count':min_count,        
        'vocab_limit':vocab_limit,
        'sentence_file':infile,
        'out_prefix':outfile
    }
    
    start = time()
    print("="*80)
    print("Analyzing %i sentences + the Semeval training sentences" % num_sentences)
    print("="*80)

    print("(%i:%i:%i) Reading Data..." % sec_to_hms(time()-start))
    wiki_sentences = []
    for i, line in enumerate(open(FLAGS['sentence_file'], 'r')):
        if i > FLAGS['num_sentences']:
            break
        wiki_sentences.append(nlp(unicode(line.strip())))
        
    train, valid, test, label2int, int2label = sdh.load_semeval_data()
    sem_sentences = [ sent[0] for sent in train['sents']+valid['sents']+test['sents'] ]

    print("(%i:%i:%i) Creating vocab..." % sec_to_hms(time()-start))
    vocab, vocab2int, int2vocab, vocab_dist = create_vocab_from_data(wiki_sentences,
                                                                 important_sentences=sem_sentences,
                                                                 vocab_limit=FLAGS['vocab_limit'],
                                                                 min_count=FLAGS['min_count'],
                                                                 dep=False,
                                                                 oov_count=1)
    dep_vocab, dep2int, int2dep, dep_dist = create_vocab_from_data(wiki_sentences,
                                                                 important_sentences=sem_sentences,                                                                 vocab_limit=None,
                                                                 min_count=0,
                                                                 dep=True,
                                                                 oov_count=1)

    sem_data = [{'path':sdp, 'target':target} for (sdp, target)
                in zip(train['sdps']+valid['sdps'], train['targets']+valid['targets'])]
    # write out the data
    print("(%i:%i:%i) Writing data..." % sec_to_hms(time()-start))
    sdp_count = 0
    with open(FLAGS['out_prefix'] + str(FLAGS['num_sentences']), 'w') as outfile:
        # semeval
        for sdp in sem_data: # doesn't include semeval test sentences
            # convert from tokens to indices
            post_process_sdp(sdp)
            sdp['path'] = [ (vocab2idx(x[0], vocab2int), vocab2idx(x[1], dep2int)) for x in sdp['path'] ]
            sdp['target'] = [ vocab2idx(sdp['target'][0], vocab2int), vocab2idx(sdp['target'][1], vocab2int) ]
            if is_ok_sdp(sdp, int2vocab):
                sdp_count += 1
                # write out the dict as json line
                outfile.write(json.dumps(sdp) + '\n')
        # wiki
        for sentence in wiki_sentences:
            for sdp in sentence_to_sdps(sentence, min_len=minlen, max_len=maxlen):
                # convert from tokens to indices
                post_process_sdp(sdp)
                sdp['path'] = [ (vocab2idx(x[0], vocab2int), vocab2idx(x[1], dep2int)) for x in sdp['path'] ]
                sdp['target'] = [ vocab2idx(sdp['target'][0], vocab2int), vocab2idx(sdp['target'][1], vocab2int) ]
                if is_ok_sdp(sdp, int2vocab):
                    sdp_count += 1
                    # write out the dict as json line
                    outfile.write(json.dumps(sdp) + '\n')


    # write out the vocab file
    print("(%i:%i:%i) Writing vocab..." % sec_to_hms(time()-start))
    with open(FLAGS['out_prefix'] + str(FLAGS['num_sentences'])+'_vocab', 'w') as outfile:
        for term in zip(vocab, vocab_dist):
            outfile.write(json.dumps(term)+'\n')

    with open(FLAGS['out_prefix'] + str(FLAGS['num_sentences'])+'_dep', 'w') as outfile:
        for term in zip(dep_vocab, dep_dist):
            outfile.write(json.dumps(term)+'\n')

    print("="*80)
    print("DONE: Created %i SDPs from %i sentences with a total vocab size of %i" % (sdp_count, num_sentences, len(vocab)))
    print("Took a total of %i:%i:%i hours" % sec_to_hms(time()-start))
    print("="*80)

if __name__ == '__main__':
    main()


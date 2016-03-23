"""
Semeval Data handler
"""
import random
from spacy.en import English
nlp = English()

def convert_raw_x(line, verbose=False):
    """Convert raw line of semeval data into a useable form
    
    Convert to a triple of (spacy sentence, e1_token, e2_token)
    """
    if isinstance(line, str):
        line = unicode(line)
    s = line.strip()
    s = s[s.index(u'"')+1: -(s[::-1].index(u'"')+1)] # get s between first " and last "
    # we will assume that the first token follow the <e1> , <e2> tags are the entity words.  
    # note this is a big assumption and hopefully phrases will be in subtrees or in heads of the parse trees
    # TODO: this can be addressed by making it a 5-tuple with the endpoints also encoded
    
    # sometimes the tags are missing spaces in front or behind.
    # check out those cases separately so we don't add exrta whitespace and mess up parsing
    # Proper whitespaceing case
    s = s.replace(u' <e1>', u' e1>') # make sure there's spacing so it's recognized as seperate token
    s = s.replace(u'</e1> ', u' ')    # drop right tag
    s = s.replace(u' <e2>', u' e2>')
    s = s.replace(u'</e2> ', u' ')
    # if there wasn't proper whitespacing, the previous code didn't run
    # so fill in the gaps with these corner cases where we add in extra whitespace
    s = s.replace(u'<e1>', u' e1>') # make sure there's spacing so it's recognized as seperate token
    s = s.replace(u'</e1>', u' ')    # drop right tag
    s = s.replace(u'<e2>', u' e2>')
    s = s.replace(u'</e2>', u' ')
    
    s = nlp(s)
    tokenized_s = [token.text for token in s]
    for i, token in enumerate(tokenized_s):
        if u'e1>' == token[:3]:
            tokenized_s[i] = token[3:]
            e1_index = i
        elif u'e2>' == token[:3]:
            tokenized_s[i] = token[3:]
            e2_index = i
    s = u' '.join(tokenized_s)
    s = nlp(s)
    e1 = s[e1_index]
    e2 = s[e2_index]
    return (s, e1, e2)
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

def find_common_ancestor(e1_path, e2_path, verbose=False):
    """Loop through both dep paths and return common ancestor"""
    for t1 in e1_path:
        for t2 in e2_path:
            if verbose:
                print(t1, t2)
            if t1.idx ==  t2.idx:
                if verbose:
                    print("Common found!")
                return t1
    return None

def convert_nominals_to_sdp(X, Y, include_ends=False, verbose=False):
    X_path = dependency_path_to_root(X)
    Y_path = dependency_path_to_root(Y)
    if verbose:
        print(X.text, X.dep_)
        print(X_path)
        print(Y.text, Y.dep_)
        print(Y_path)
    common = find_common_ancestor(X_path, Y_path, verbose=verbose)
#     # now we don't want nouns for assembly
#     X_path = X_path[1:]
#     Y_path = Y_path[1:]
    # CASE (1)
    if not common:
        print("Didn't find common ancestor")
        return None
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
    if not include_ends:
        sdp[0] = (u'<X>', sdp[0][1])
        sdp[-1] = (u'<Y>', sdp[-1][1])
#     if len(sdp) < min_len or len(sdp) > max_len:
#         continue                    # skip ones that are too short or long
    return {'path': sdp, 'target':(X.text.lower(), Y.text.lower())}

def convert_nominals_to_sentence(X, Y, sent, include_ends=False, verbose=False):
    sdp = []
    started = False
    # loop through sentence, start recording when we see x, then stop after we see Y
    for i, token in enumerate(sent):
        if token is X:
            started = True
            sdp.append((smart_token_to_text(token), token.dep_))
        elif token is Y:
            sdp.append((smart_token_to_text(token), token.dep_))
            started = False
            break
        if started:
            sdp.append((smart_token_to_text(token), token.dep_))

    # convert endpoints of the paths to placeholder X and Y tokens
    if not include_ends:
        sdp[0] = (u'<X>', sdp[0][1])
        sdp[-1] = (u'<Y>', sdp[-1][1])
#     if len(sdp) < min_len or len(sdp) > max_len:
#         continue                    # skip ones that are too short or long
    return {'path': sdp, 'target':(X.text.lower(), Y.text.lower())}

def post_process_sdp(sdp):
    """ Filter out unwanted sdps structure """
    if not sdp:
        return sdp
    bad_tokens = set([u'<PUNCT>']) #set([',', '.', '-', '(', ')', '&', '*', '_', '%', '!', '?', '/', '<', '>', '\\', '[', ']', '{', '}', '"', "'"])
    sdp['path'] = [x for x in sdp['path'] if x[0] not in bad_tokens]
    return sdp

def is_ok_sdp(sdp):#, int2vocab, oov_percent=75):
    """ Helper function to mak sure SDP isn't a poor example.

    Filters used to identify bas data:
    1. Neither targets may be oov
    2. The relation itself must be less than `oov_percent` percent number of relations
    """
#     oov = int2vocab.keys()[-1]
#     # print(oov, sdp['target'])
#     if sdp['target'][0] == oov or sdp['target'][1] == oov:
#         return False
#     oov_count = len([ t for t in sdp['path'] if t[0] == oov])
#     too_many = int((oov_percent/100.0)*len(sdp['path']))
#     if oov_count > too_many:
#         return False
    if not sdp or not sdp['path'] or not sdp['target']:
        return False
    return True

def line_to_data(raw_line, include_ends=False, verbose=False, sentence=False):
    sent = convert_raw_x(raw_line)
    e1 = sent[1]
    e2 = sent[2]
    if sentence:
        sdp = convert_nominals_to_sentence(e1, e2, sent[0], include_ends=include_ends, verbose=verbose)
    else:
        sdp = convert_nominals_to_sdp(e1, e2, include_ends=include_ends, verbose=verbose)
    if not sdp:
        print(raw_line)
        print(sent)
#     post_process_sdp(sdp)
    if is_ok_sdp(sdp):
        return sent, sdp['path'], sdp['target']
    else:
        print("Bad sentence: %r" % raw_line )
        print(sent, sdp)
        return None, None, None

def create_label2int():
    labels = [
            'Cause-Effect(e1,e2)',
            'Cause-Effect(e2,e1)',
            'Product-Producer(e1,e2)',
            'Product-Producer(e2,e1)',
            'Entity-Origin(e1,e2)',
            'Entity-Origin(e2,e1)',
            'Instrument-Agency(e1,e2)',
            'Instrument-Agency(e2,e1)',
            'Component-Whole(e1,e2)',
            'Component-Whole(e2,e1)',
            'Content-Container(e1,e2)',
            'Content-Container(e2,e1)',
            'Entity-Destination(e1,e2)',
            'Entity-Destination(e2,e1)',
            'Member-Collection(e1,e2)', 
            'Member-Collection(e2,e1)', 
            'Message-Topic(e1,e2)',
            'Message-Topic(e2,e1)',
            'Other' ]
    return {label:i for (i, label) in enumerate(labels)}

def line_to_label(raw_label_line, label2int):
    """Convert raw line of semeval labels into a useable form (ints)"""
    # define the list ourselves so the ordering is nice
    line = raw_label_line.strip()
    # if line in label2int:
    #     return label2int[line]
    # else:
    #     label2int[line] = len(label2int.keys())
    #     return label2int[line]
    return label2int[line]

def load_semeval_data(shuffle_seed=42, include_ends=False, sentence=False):
    """Load in SemEval 2010 Task 8 Training file and return lists of tuples:
    
    Tuple form =  (spacy(stripped sentence), index of e1, index of e2)"""
    ### TRAINING AND VALIDATION DATA ###
    training_txt_file = 'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    validation_index = 8000 - 800 # len data - len valid - 1 since we start at 0
    validation_size = 800
    all_ = {'raws':[], 'sents':[], 'sdps':[], 'targets':[], 'labels':[], 'comments':[]}
    train = {'raws':[], 'sents':[], 'sdps':[], 'targets':[], 'labels':[], 'comments':[]}
    valid = {'raws':[], 'sents':[], 'sdps':[], 'targets':[], 'labels':[], 'comments':[]}
    text = open(training_txt_file, 'r').readlines()
    label2int = create_label2int() # keep running dictionary of labels
    assert len(text) // 4 == 8000
    for cursor in range(len(text) // 4): # each 4 lines is a datum
            text_line = text[4*cursor]
            label_line = text[4*cursor +1]
            comment = text[4*cursor + 2]
            sent, sdp, target = line_to_data(text_line, include_ends=include_ends, sentence=sentence)
            label = line_to_label(label_line, label2int)
#             print(sent, sdp, target, label)
            if not (sent and sdp and target):
                print("Skipping this one... %r" % text_line)
                print(sent, sdp, target, label)
                continue
            all_['raws'].append(text_line)
            all_['sents'].append(sent)
            all_['sdps'].append(sdp)
            all_['targets'].append(target)
            all_['labels'].append(label)
            all_['comments'].append(comment)
            if cursor < validation_index:
                train['raws'].append(text_line)
                train['sents'].append(sent)
                train['sdps'].append(sdp)
                train['targets'].append(target)
                train['labels'].append(label)
                train['comments'].append(comment)
            else:
                valid['raws'].append(text_line)
                valid['sents'].append(sent)
                valid['sdps'].append(sdp)
                valid['targets'].append(target)
                valid['labels'].append(label)
            #     valid['comments'].append(comment)
    # shuffle all and take the last validation_size as validation, rest as test
    if shuffle_seed:
        random.seed(shuffle_seed)
        zip_all = zip(all_['raws'], all_['sents'], all_['sdps'], all_['targets'], all_['labels'], all_['comments'])
        random.shuffle(zip_all)
        raws, sents, sdps, targets, labels, comments = zip(*zip_all)
        train['raws'], valid['raws'] = raws[:-validation_size], raws[-validation_size:]
        train['sents'], valid['sents'] = sents[:-validation_size], sents[-validation_size:]
        train['sdps'], valid['sdps'] = sdps[:-validation_size], sdps[-validation_size:]
        train['targets'], valid['targets'] = targets[:-validation_size], targets[-validation_size:]
        train['labels'], valid['labels'] = labels[:-validation_size], labels[-validation_size:]
        train['comments'], valid['comments'] = comments[:-validation_size], comments[-validation_size:]

    int2label = {i:label for (label, i) in label2int.items()}
    print("Num training: %i" % len(train['labels']))
    print("Num valididation: %i" % len(valid['labels']))
    assert sorted(label2int.values()) == range(19) # 2 for each 9 asymmetric relations and 1 other
    
    ### TEST DATA ### (has no labels)
    test_txt_file = "SemEval2010_task8_all_data/SemEval2010_task8_testing/TEST_FILE.txt"
    test = {'raws':[], 'sents':[], 'sdps':[], 'targets':[]}
    text = open(test_txt_file, 'r').readlines()
    for line in text:
        sent, sdp, target = line_to_data(line, include_ends=include_ends, sentence=sentence)
        if not (sent and sdp and target):
            print("Skipping this one... %r" % text_line)
            print(sent, sdp, target, label)
            sent = [nlp(u'<OOV>')]*3
            sdp = [[u'<OOV>',u'<OOV>']]
            target= [u'<OOV>',u'<OOV>']
        test['raws'].append(line)
        test['sents'].append(sent)
        test['sdps'].append(sdp)
        test['targets'].append(target)
    
    print("Num testing: %i" % len(test['targets']))

    return train, valid, test, label2int, int2label

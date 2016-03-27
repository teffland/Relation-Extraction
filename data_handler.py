"""
A DataHandler class to read in our data, and generate batches for us
"""
from __future__ import print_function
import json
import numpy as np
import random

class DataHandler(object):
    """Handler to read in data and generate data and batches for model training and evaluation"""
    def __init__(self, data_prefix, valid_percent=10, max_sequence_len=None, shuffle_seed=42):
        self._data_prefix = data_prefix
        self._valid_percent = valid_percent / 100.0
        self.read_data(shuffle_seed=shuffle_seed)
        if max_sequence_len:
            assert max_sequence_len >= self._max_seq_len, "Cannot for sequence length shorter than the data yields"
            self._max_seq_len = max_sequence_len

        print("%i total examples :: %i training : %i valid (%i:%i split)" 
              % (len(self._paths) + len(self._valid_paths), 
                len(self._paths), len(self._valid_paths),
                100-self._valid_percent*100, self._valid_percent*100))
        print("Vocab size: %i Dep size: %i" % (self._vocab_size, self._dep_size))

    def read_data(self, shuffle_seed=42):
        print("Creating Data objects...")
        # read in sdp data
        data = []
        with open(self._data_prefix, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        if shuffle_seed:
            random.seed(shuffle_seed)
            random.shuffle(data) # start off in random order before we do validation split 
        self._paths = [ datum['path'] for datum in data ]
        self._max_seq_len = max([ len(path) for path in self._paths ])
        self._targets = [ datum['target'] for datum in data] # targets get doubly wrapped in lists
        
        #make sure all of the paths have same depth and all the targets have depth 2
        assert len(set(len(p) for path in self._paths for p in path)) == 1, "Not all path tuples have same len"
        assert set(len(target) for target in self._targets) == set([2]), "All target tuples must be pairs"

        # now chop off a validation set. Make it 
        self._valid_split_idx = int((1-self._valid_percent)*len(self._paths))
        self._valid_paths = self._paths[self._valid_split_idx:]
        self._valid_targets = self._targets[self._valid_split_idx:]
        self._paths = self._paths[:self._valid_split_idx]
        self._targets = self._targets[:self._valid_split_idx]
        # print(self._paths)
        # read in vocab and distribution
        vocab_and_dist = []
        with open(self._data_prefix+"_vocab", 'r') as f:
            for line in f:
                vocab_and_dist.append(json.loads(line))
        self._vocab = [x[0] for x in vocab_and_dist]
        self._true_vocab_dist = [x[1] for x in vocab_and_dist]
        self._vocab_dist = self._true_vocab_dist
        self._vocab2int = {v:i for (i,v) in enumerate(self._vocab)}
        self._int2vocab = {i:v for (v,i) in self._vocab2int.items()}
        self._vocab_size = len(self._vocab)
        
        # read in dependency vocab and distribution
        dep_and_dist = []
        with open(self._data_prefix+"_dep", 'r') as f:
            for line in f:
                dep_and_dist.append(json.loads(line))
        self._dep_vocab = [x[0] for x in dep_and_dist]
        self._true_dep_dist = [x[1] for x in dep_and_dist]
        self._dep_dist = self._true_dep_dist
        self._dep2int = {v:i for (i,v) in enumerate(self._dep_vocab)}
        self._int2dep = {i:v for (v,i) in self._dep2int.items()}
        self._dep_size = len(self._dep_vocab)
        print("Done creating Data objects")

    def shuffle_data(self):
        """ Shuffle shit around to help SGD convergence"""
        paths_and_targets = zip(self._paths, self._targets)
        random.shuffle(paths_and_targets)
        self._paths = [d[0] for d in paths_and_targets]
        self._targets = [d[1] for d in paths_and_targets]
    
    def _sequences_to_tensor(self, list_of_lists):
        """ Convert list of lists of either single elements or tuples into matrix of appropriate dim"""
        lengths = np.array([len(list_) for list_ in list_of_lists]).reshape([-1, 1])
        
        #matrix case
        if isinstance(list_of_lists[0][0], (int, float)):
            matrix = np.zeros([len(list_of_lists), self._max_seq_len])
            for i, list_ in enumerate(list_of_lists):
                matrix[i, :len(list_)] = list_
            return matrix, lengths
        
        #tensor case
        if isinstance(list_of_lists[0][0], (tuple, list)):
            k = len(list_of_lists[0][0]) # we asserted before that all of them were the same len
            tensor = np.zeros([len(list_of_lists), self._max_seq_len, k])
            for i, list_ in enumerate(list_of_lists):
                for j in range(k):
                    tensor[i, :len(list_), j] = [ x[j] for x in list_ ]
            return tensor, lengths
    
    def _generate_batch(self, offset, batch_size, inputs, targets, target_neg=False, neg_per=None, neg_level=1):
        """Expects the data as list of lists of indices

        Converts them to matrices of indices, lang model labels, and lengths"""
        assert neg_level > 0, "Cannot have negative examples with no corruption"
        start = offset*batch_size
        end = start + batch_size
        if end > len(inputs):
            end = len(inputs)
#             print("Not full batch")
        inputs = inputs[start:end]
        targets = np.array(targets[start:end])
        labels = np.ones(targets.shape[0]).reshape((-1, 1))
        input_mat, len_vec = self._sequences_to_tensor(inputs)
        # generate the negative samples
        # randomly choose one index for each negative sample 
        # TODO: option to replace more than one phrase element
        # and replace that with a random word drawn from the scaled unigram distribution
        if neg_per:
            # neg level can't be higher than 2 if neg_target:
            if target_neg:
                neg_level = min(2, neg_level)
            negatives = []
            neg_targets = []
            for i, seq in enumerate(inputs): # for each true sequence
                for neg in range(neg_per): # create neg_per extra negative examples
                    neg_seq = seq[:]
                    neg_target = targets[i]
                    if target_neg:
                        # just one, pick a random target to flip
                        if neg_level == 1:
                            neg_idx = int(random.uniform(0,2)) # random 0,1 w/ 50% each
                            neg_target[neg_idx] = np.random.choice(range(len(self._vocab)), 
                                                   size=1, p=self._vocab_dist)[0]
                        if neg_level == 2:
                            neg_target = list(np.random.choice(range(len(self._vocab)), 
                                                   size=2, p=self._vocab_dist))
                    else: # otherwise we're corrupting the sequences, not the targets
                        # break sticks to split up noise into vocab and deps
                        # pick an int that's less that seq - <X> - <Y>
                        # but make sure that's not more than we asked for
                        num_v = min(int(random.uniform(0, max(len(seq)-2, 1))), neg_level)
                        # print(num_v, len(seq))
                        num_d = neg_level - num_v
                        if num_v: # choice breaks if zero
                            v_rand_idx = np.random.choice(range(1, len(seq)-1), size=num_v)
                            v_noise = np.random.choice(range(len(self._vocab)), 
                                                       size=num_v, p=self._vocab_dist)
                            for j, v in zip(v_rand_idx, v_noise):
                                neg_seq[j][0] = v
                        if num_d:
                            d_rand_idx = np.random.choice(range(0, len(seq)), size=num_d)
                            d_noise = np.random.choice(range(len(self._dep_vocab)), 
                                                       size=num_d, p=self._dep_dist)
                            # do the replacements
                            for j, d in zip(d_rand_idx, d_noise):
                                neg_seq[j][1] = d
                    negatives.append(neg_seq)
                    neg_targets.append(neg_target)
            neg_mat, neg_len = self._sequences_to_tensor(negatives)
            neg_labels = np.zeros_like(neg_len)
            all_inputs = np.vstack((input_mat, neg_mat)).astype(np.int32)
            all_targets = np.vstack((targets, np.array(neg_targets))).astype(np.int32)
            all_labels = np.vstack((labels, neg_labels)).astype(np.int32)
            all_lengths = np.vstack((len_vec, neg_len)).astype(np.int32)
        else:
            all_inputs = input_mat.astype(np.int32)
            all_targets = targets.astype(np.int32)
            all_labels = labels.astype(np.int32)
            all_lengths = len_vec.astype(np.int32)
        return all_inputs, all_targets, all_labels, all_lengths

    def _generate_class_batch(self, offset, batch_size, inputs, targets, labels):
        """Expects the data as list of lists of indices

        Converts them to matrices of indices, lang model labels, and lengths"""
        start = offset*batch_size
        end = start + batch_size
        if end > len(inputs):
            end = len(inputs)
#             print("Not full batch")
        inputs = inputs[start:end]
        targets = np.array(targets[start:end]).astype(np.int32)
        labels = np.array(labels[start:end]).reshape([-1, 1]).astype(np.int64)
        inputs, lens = self._sequences_to_tensor(inputs)
        inputs = inputs.astype(np.int32)
        lens = lens.astype(np.int32)
        return inputs, targets, labels, lens
    
    def batches(self, batch_size, target_neg=False, neg_per=5, neg_level=1, offset=0):
        num_steps = len(self._paths) // batch_size
        if num_steps == 0:
            num_steps = 1
        for step in range(offset, num_steps):
            yield self._generate_batch(step, batch_size, 
                                       self._paths, self._targets, 
                                       target_neg=target_neg,
                                       neg_per=neg_per, neg_level=neg_level)

    def validation_batch(self):
        valid_inputs, valid_targets, valid_labels, valid_lens = self._generate_batch(0,    
                                                              len(self._valid_targets), 
                                                              self._valid_paths, 
                                                              self._valid_targets, 
                                                              neg_per=0)
        return valid_inputs, valid_targets, valid_labels, valid_lens

    def classification_batch(self, batch_size, inputs, targets, labels, offset=0, shuffle=False):
        if shuffle:
            data = zip(inputs, targets, labels)
            random.shuffle(data)
            inputs, targets, labels = zip(*data)
        return self._generate_class_batch(offset, batch_size, inputs, targets, labels)
    
    def scale_vocab_dist(self, power):
        self._vocab_dist = self._distribution_to_power(self._true_vocab_dist, power)
        
    def scale_dep_dist(self, power):
        self._dep_dist = self._distribution_to_power(self._true_dep_dist, power)


    def _int_to_vocab(self, index, int2vocab):
        """ handle index conversion with fault tolerance """
        if index in int2vocab:
            return int2vocab[index]
        else:
            return int2vocab.values()[-1] # <OOV>

    def _vocab_to_int(self, vocab, vocab2int):
        """ handle index conversion with fault tolerance """
        if vocab in vocab2int:
            return vocab2int[vocab]
        else:
            # print(vocab, vocab2int.keys()[-1], vocab2int['<OOV>'])
            return vocab2int['<OOV>'] # <OOV>

    def sequence_to_sentence(self, sequence, len_=10e5, show_dep=False, delim=" "):
        # does the sequence contain the dependencies also?
        if isinstance(sequence[0], int): # this is just a sinlg elist not list of lists
            return delim.join([ self._int_to_vocab(x, self._int2vocab) 
                                   for (i, x) in enumerate(sequence) 
                                   if i < len_ ] )

        elif set([len(d) for d in sequence]) == set([2]): # list of lists of pairs of ints
            if show_dep:
                return delim.join([ "("+self._int_to_vocab(x[0], self._int2vocab)
                                    +", "+self._int_to_vocab(x[1], self._int2dep)+")"
                                     for i, x in enumerate(sequence) 
                                     if i < len_ ] )
            else:   
                return delim.join([ self._int_to_vocab(x[0], self._int2vocab) 
                                   for (i, x) in enumerate(sequence) 
                                   if i < len_ ] )


        elif set([len(d) for d in sequence]) == set([1]): # list of list of ints
            return delim.join([ self._int_to_vocab(x, self._int2vocab) 
                             for i, x in enumerate(sequence) 
                             if i < len_ ])
        else:
            print("Not sure what to make of sequence %r" % sequence)

    def sequences_to_sentences(self, sequences, lens=None,
                               show_dep=False, 
                               delim=" "):
        # is expecting a list of lists of lists eg, a list of paths, 
        # where a path is a list of lists of tokens and deps
        if lens:
            return [ self.sequence_to_sentence(sequence, len_, show_dep=show_dep, delim=delim) 
                    for (sequence, len_) in zip(sequences, lens) ]
        else:
            return [ self.sequence_to_sentence(sequence, show_dep=show_dep, delim=delim) 
                    for sequence in sequences ]

    def sentence_to_sequence(self, sentence, len_=10e5, show_dep=False, delim=" "):
        try:
            if isinstance(sentence[0], (unicode, str)): # this is just a sinlg elist not list of lists
                return [ self._vocab_to_int(x, self._vocab2int) 
                                       for (i, x) in enumerate(sentence) 
                                       if i < len_ ]

            elif set([len(d) for d in sentence]) == set([2]): # list of lists of pairs of ints
                return [ [self._vocab_to_int(x[0], self._vocab2int),
                        self._vocab_to_int(x[1], self._dep2int)]
                        for i, x in enumerate(sentence) ]

            elif set([len(d) for d in sentence]) == set([1]): # list of list of ints
                return [ self._vocab_to_int(x, self._vocab2int) 
                                 for i, x in enumerate(sentence) 
                                 if i < len_ ]
            else:
                print("Not sure what to make of sentence %r" % sentence)
        except:
            print("excepted sentence: ", sentence)

    def sentences_to_sequences(self, sentences, lens=None):
        # is expecting a list of lists of lists eg, a list of paths, 
        # where a path is a list of lists of tokens and deps
        if lens:
            return [ self.sentence_to_sequence(sentence, len_) 
                    for (sentence, len_) in zip(sentences, lens) ]
        else:
            return [ self.sentence_to_sequence(sentence) 
                    for sentence in sentences ]

    def readable_data(self, valid=False, show_dep=False):
        if valid:
            paths = self.sequences_to_sentences(self._valid_paths, show_dep=show_dep)
            targets = self.sequences_to_sentences(self._valid_targets, delim=", ")
        else:
            # print(self._paths)
            paths = self.sequences_to_sentences(self._paths, show_dep=show_dep)
            targets = self.sequences_to_sentences(self._targets, delim=", ") 
        return paths, targets
        
    def _distribution_to_power(self, distribution, power):
        """Return a distribution, scaled to some power"""
        dist = [ pow(d, power) for d in distribution ]
        dist /= np.sum(dist)
        return dist
    
    def _sample_distribution(self, distribution):
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
    
    def num_steps(self, batch_size):
        return len(self._paths) // batch_size

    def index_at_vocab(self, vocab):
        return self._vocab_to_int(vocab, self._vocab2int)

    def vocab_at(self, index):
        return self._int_to_vocab(index, self._int2vocab)

    def dep_at(self, index):
        return self._int_to_vocab(index, self._int2vocab)

    @property
    def data_prefix(self):
        return self._data_prefix
    
    @property
    def vocab(self):
        return self._vocab
    
    @property
    def dep_vocab(self):
        return self._dep_vocab

    @property
    def max_seq_len(self):
        return self._max_seq_len

    @max_seq_len.setter
    def max_seq_len(self, value):
        self._max_seq_len = max(self._max_seq_len, value)

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def dep_size(self):
        return self._dep_size

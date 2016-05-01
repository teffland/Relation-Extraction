"""
Relation Embed model
"""
from __future__ import print_function
import numpy as np
import tensorflow as tf

def batch_triple_inner(W, x, y, z):
    """ Computes the inner product of 3 vectors and a tensor

    Args:
        W: a 3D tensor with shape[x_len, y_len, z_len]
        x: a 2D tensor with shape[batch_size, x_len]
        y: a 2D tensor with shape[batch_size, y_len]
        y: a 2D tensor with shape[batch_size, z_len]

    NOTE: Literally as naive as is possible"""
    val = tf.zeros(tf.pack([tf.shape(x)[0], 1])) # get zeros to work with unknown size
    for i in xrange(x.get_shape()[1]):
        for j in xrange(y.get_shape()[1]):
            for k in xrange(z.get_shape()[1]):
                val += W[i,j,k]*x[:,i]*y[:,j]*z[:,k]
    return val

class RelEmbed(object):
    """ Encapsulation of the dependency RNN lang model
    
    TODO:
    Add configuration to classification styles:
    - Softmax loss
    - ranking loss
        - hinge vs softplus
            - margin size
            - piecewise like in dos Santos?
        - model 'Other' or not
        - matmul inner product 
        vs element_wise inner product (equivalent to diagonal matmul)
        vs single matrix (not class-wise inner product, but a column per class)
        
    TODO: Add unit tests
    - Classification scores

    TODO: Add functions for pulling out weight matrices as np arrays

    TODO: Add configuration for composition styles
    - RNN vs GRU vs LSTM
    - Forward or BiDirectional
    - Number of layers

    TODO: Add configuration for regularization
    Unsupervised:
    - L1 vs L2
    - \lambda
    Supervised:
    - L1 vs L2
    - \lambda

    """
    def __init__(self, config):
        self.config = config
        self.max_num_steps = config['max_num_steps']
        self.word_embed_size = config['word_embed_size']
        self.dep_embed_size = config['dep_embed_size']
        self.pos_embed_size = config['pos_embed_size']
        self.hidden_layer_size = config['hidden_layer_size']
        self.input_size = self.word_embed_size + self.dep_embed_size + self.pos_embed_size
        self.bidirectional = config['bidirectional']
        self.num_clusters = config['num_clusters']
        self.hidden_size = self.word_embed_size #config['hidden_size']
        self.pretrained_word_embeddings = config['pretrained_word_embeddings'] # None if we don't provide them
        if np.any(self.pretrained_word_embeddings):
            assert self.word_embed_size == self.pretrained_word_embeddings.shape[1]
        self.num_classes = config['num_predict_classes']
        self.max_grad_norm = config['max_grad_norm']

        self.predict_style = 'END' # could be 'ALL' or 'AVG' also
        
        self.vocab_size = config['vocab_size']
        self.dep_vocab_size = config['dep_vocab_size']
        self.pos_vocab_size = config['pos_vocab_size']
        self.name = config['model_name']
        self.checkpoint_prefix = config['checkpoint_prefix'] + self.name
        self.summary_prefix = config['summary_prefix'] + self.name
        
        self.initializer = tf.random_uniform_initializer(-.1, .1)
        self.word_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1./(self.word_embed_size))
        self.dep_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1./(self.dep_embed_size))
        self.pos_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1./(self.pos_embed_size))
        self.hidden_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1./(self.hidden_size))
        with tf.name_scope(self.name):
            with tf.name_scope("Forward"):
                self._build_forward_graph()
            with tf.name_scope("Classifier"):
                if config['supervised']:
                    self._build_classification_graph()
            with tf.name_scope("Backward"):
                self._build_train_graph()
                if config['supervised']:
                    self._build_class_train_graph()
            with tf.name_scope("Nearby"):
                self._build_similarity_graph()

        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=config['max_to_keep'])
            
        if config['interactive']:
            self.session = tf.InteractiveSession()
        else:
            self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())        
        self.summary_writer = tf.train.SummaryWriter(self.summary_prefix, self.session.graph_def)
        
    def save_validation_accuracy(self, new_score):
        assign_op = self._valid_accuracy.assign(new_score)
        _, summary = self.session.run([assign_op, self._valid_acc_summary])
        self.summary_writer.add_summary(summary)
        
    def _build_forward_graph(self):
        # input tensor of zero padded indices to get to max_num_steps
        # None allows for variable batch sizes
        with tf.name_scope("Inputs"):
            self._input_phrases = tf.placeholder(tf.int32, [None, self.max_num_steps, 3]) # [batch_size, w_{1:N}, 2]
            self._input_targets = tf.placeholder(tf.int32, [None, 1]) # [batch_size, w_x]
            self._input_labels = tf.placeholder(tf.int32, [None, 1]) # [batch_size, from true data?] \in {0,1}
            self._input_lengths = tf.placeholder(tf.int32, [None, 1]) # [batch_size, N] (len of each sequence)
            self._input_predict_x = tf.placeholder(tf.int32, [None, 1]) # whether to use the backwards RNN or not
            batch_size = tf.shape(self._input_lengths)[0]
            self._keep_prob = tf.placeholder(tf.float32)
        
        with tf.name_scope("Embeddings"):
            if np.any(self.pretrained_word_embeddings):
                self._word_embeddings = tf.Variable(self.pretrained_word_embeddings,name="word_embeddings")
                self._target_embeddings = tf.Variable(self.pretrained_word_embeddings, name="left_target_embeddings")
                self._right_target_embeddings = tf.Variable(self.pretrained_word_embeddings, name="right_target_embeddings")
            else:
                self._word_embeddings = tf.get_variable("word_embeddings", 
                                                        [self.vocab_size, self.word_embed_size], 
                                                    initializer=self.word_initializer,
                                                        dtype=tf.float32)
                self._target_embeddings = tf.get_variable("target_embeddings", 
                                                        [self.vocab_size, self.word_embed_size], 
                                                    initializer=self.word_initializer,
                                                    trainable=False,
                                                        dtype=tf.float32)
            
            self._dependency_embeddings = tf.get_variable("dependency_embeddings", 
                                                    [self.dep_vocab_size, self.dep_embed_size], 
                                                    initializer=self.dep_initializer,
                                                    dtype=tf.float32)
            self._pos_embeddings = tf.get_variable("pos_embeddings", 
                                                    [self.pos_vocab_size, self.pos_embed_size], 
                                                    initializer=self.pos_initializer,
                                                    dtype=tf.float32)
            # renormalize every turn
            self._target_embeddings - tf.nn.l2_normalize(self._target_embeddings, 1)

            # perform a task-specific transform of the rnn
            rnn_w = tf.get_variable("rnn_w", [self.hidden_size, self.hidden_size],
                                    dtype=tf.float32)
            
            input_embeds = tf.nn.dropout(tf.nn.embedding_lookup(self._word_embeddings, 
                                                  tf.slice(self._input_phrases, [0,0,0], [-1, -1, 1])),
                                         keep_prob=self._keep_prob)
            dep_embeds = tf.nn.dropout(tf.nn.embedding_lookup(self._dependency_embeddings,
                                                tf.slice(self._input_phrases, [0,0,1], [-1, -1, 1])),
                                       keep_prob=self._keep_prob)
            pos_embeds = tf.nn.dropout(tf.nn.embedding_lookup(self._pos_embeddings,
                                                tf.slice(self._input_phrases, [0,0,2], [-1, -1, 1])),
                                       keep_prob=self._keep_prob)
            ### SEPARATE TARGET EMBEDDING MATRIX ###
            # left_target_embeds = tf.nn.dropout(tf.nn.embedding_lookup(self._left_target_embeddings, 
            #                                             tf.slice(self._input_targets, [0,0], [-1, 1])),
            #                                     keep_prob=self._keep_prob)
            # right_target_embeds = tf.nn.dropout(tf.nn.embedding_lookup(self._right_target_embeddings, 
            #                                             tf.slice(self._input_targets, [0,1], [-1, 1])),
            #                                      keep_prob=self._keep_prob)
            # no delay dropout so we can tanh it first
            # left_target_embeds = tf.nn.embedding_lookup(self._target_embeddings, 
                                                        # tf.slice(self._input_targets, [0,0], [-1, 1]))
            # right_target_embeds = tf.nn.embedding_lookup(self._right_target_embeddings, 
            #                                             tf.slice(self._input_targets, [0,1], [-1, 1]))
            ### ALL SAME EMBEDDING MATRIX ###
            self._target_embeds = tf.nn.embedding_lookup(self._target_embeddings, 
                                                        tf.slice(self._input_targets, [0,0], [-1, 1]))

#             print(tf.slice(self._input_phrases, [0,0,1], [-1, -1, 1]).get_shape(), dep_embeds.get_shape())
#             print(left_target_embeds.get_shape(), right_target_embeds.get_shape())
            # self._target_embeds = tf.squeeze(tf.concat(2, [left_target_embeds, right_target_embeds]), [1])
            # self._target_embeds = tf.nn.dropout(tf.nn.l2_normalize(self._target_embeds, 1 ), keep_prob=self._keep_prob)
            self._target_embeds = tf.nn.dropout(tf.squeeze(self._target_embeds, [1]), keep_prob=self._keep_prob)
            # self._target_embeds = tf.squeeze(self._target_embeds, [1])

            print(self._target_embeds.get_shape())
            # TODO: Add dropout to embeddings
        
        with tf.name_scope("RNN"):
            
            # TODO: Make it multilevel
#             self._initial_state = self.cell.zero_state(batch_size, tf.float32)
#             print(self._initial_state.get_shape())
            input_words = [ tf.squeeze(input_, [1, 2]) for input_ in tf.split(1, self.max_num_steps, input_embeds)]
            input_deps = [ tf.squeeze(input_, [1, 2]) for input_ in tf.split(1, self.max_num_steps, dep_embeds)]
            input_pos = [ tf.squeeze(input_, [1, 2]) for input_ in tf.split(1, self.max_num_steps, pos_embeds)]
            inputs = [ tf.concat(1, [input_word, input_dep, input_pos_]) 
                       for (input_word, input_dep, input_pos_) in zip(input_words, input_deps, input_pos)]

            # inputs = input_words # just use words
            # start off with a basic configuration
            if self.bidirectional:
                self.fwcell = tf.nn.rnn_cell.GRUCell(self.hidden_size, 
                                                input_size=self.input_size)
                self.bwcell = tf.nn.rnn_cell.GRUCell(self.hidden_size, 
                                                input_size=self.input_size)
                with tf.variable_scope("FW") as scope:
                    # predicting y if going forward
                    y_outs, y_state = tf.nn.rnn(self.fwcell, inputs, 
                                         sequence_length=tf.squeeze(self._input_lengths, [1]),
                                         dtype=tf.float32, scope=scope)
                with tf.variable_scope("BW") as scope:
                    x_outs, x_state = tf.nn.rnn(self.bwcell, inputs, 
                                     sequence_length=tf.squeeze(self._input_lengths, [1]),
                                     dtype=tf.float32, scope=scope)
                # print(len(x_outs), self.max_num_steps)
                # final out only
                #TODO: Make sure squeeez isn't fucking it up (pretty sure it's not)
                choose_x_y = tf.select(tf.cast(tf.squeeze(self._input_predict_x, [1]), tf.bool), x_state, y_state)
                # choose_x_y = tf.matmul(choose_x_y, rnn_w)
                self._final_states = tf.nn.dropout(choose_x_y, keep_prob=self._keep_prob)
                # self._final_states = choose_x_y
                # # all outs
                # choose_x_y = tf.concat(0, [tf.select(tf.cast(tf.squeeze(self._input_predict_x, [1]), tf.bool), x, y)
                #               for (x,y) in zip(x_outs, y_outs)])
                # self._final_states = tf.nn.dropout(choose_x_y, keep_prob=self._keep_prob) 
                # self._final_state = tf.nn.dropout(tf.concat(1, [x_state, y_state]), keep_prob=self._keep_prob)
                # print(self._final_state.get_shape())
            else:
                self.cell = tf.nn.rnn_cell.GRUCell(self.hidden_size, 
                                                input_size=self.input_size)
                _, state = tf.nn.rnn(self.cell, inputs, 
                                     sequence_length=tf.squeeze(self._input_lengths, [1]),
                                     dtype=tf.float32)
#                                  initial_state=self._initial_state)
            # self._final_state = tf.nn.dropout(tf.nn.l2_normalize(state, 1), keep_prob= self._keep_prob)
                self._final_state = tf.nn.dropout(state, keep_prob=self._keep_prob)

            # get references to the RNN vars
            # with tf.variable_scope('RNN', reuse=True):
            #     self._gate_matrix = tf.get_variable('GRUCell/Gates/Linear/Matrix')
            #     self._gate_bias = tf.get_variable('GRUCell/Gates/Linear/Bias')
            #     self._cand_matrix = tf.get_variable('GRUCell/Candidate/Linear/Matrix')
            #     self._cand_bias = tf.get_variable('GRUCell/Candidate/Linear/Bias')

        
        # self._lambda2 = tf.Variable(10e-6, trainable=False, name="L2_Lambda2")
        self._lambda = tf.Variable(10e-7, trainable=False, name="L2_Lambda")
        with tf.name_scope("Loss"):
            ### CLUSTERED ###
            # self._cluster_input = self._final_states
            # self._clusters_w =  tf.get_variable("clusters_w", [self._cluster_input.get_shape()[1], self.num_clusters**2])
            # self._clusters_b = tf.Variable(tf.zeros([self.num_clusters**2], dtype=tf.float32), name="clusters_b")   

            # logits = tf.matmul(self._cluster_input, self._clusters_w) + self._clusters_b

            # self._xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, 
            #                                      tf.to_int64(tf.squeeze(self._input_labels, [1]))))        
            
            # self._l2_penalty = self._lambda*(tf.nn.l2_loss(self._clusters_w)
            #                                + tf.nn.l2_loss(self._clusters_b))
            # self._loss = self._xent + self._l2_penalty 
            ### NEGATIVE SAMPLED INNER PRODUCT ###
            # final state only
            flat_states = tf.reshape(self._final_states, [-1])
            flat_target_embeds = tf.reshape(self._target_embeds, [-1])
            flat_logits = tf.mul(flat_states, flat_target_embeds)
            logits = tf.reduce_sum(tf.reshape(flat_logits, tf.pack([batch_size, -1])), 1)

            # # all states
            # # flat_states = tf.reshape(self._final_states, [-1])
            # # flat_target_embeds = tf.reshape(tf.tile(self._target_embeds, [self.max_num_steps, 1]), 
            # #                                 [-1])
            # # flat_logits = tf.mul(flat_states, flat_target_embeds)
            # # logits = tf.reduce_sum(tf.reshape(flat_logits, 
            # #                                    tf.pack([batch_size*self.max_num_steps, -1])), 
            # #                         1)
           
            
            self._l2_penalty = self._lambda*(tf.nn.l2_loss(self._target_embeds)
                                            +tf.nn.l2_loss(rnn_w))
            #tf.nn.l2_loss(self._gate_matrix)
            #                     #           + tf.nn.l2_loss(self._gate_bias)
            #                     #           + tf.nn.l2_loss(self._cand_matrix)
            #                     #           + tf.nn.l2_loss(self._cand_bias))
            #                                # + tf.nn.l2_loss(self._word_embeddings))
            #                                         #+tf.nn.l2_loss(self._dependency_embeddings)
            #                                         # tf.nn.l2_loss(self._left_target_embeddings)
            #                                         # +tf.nn.l2_loss(self._right_target_embeddings))
            # # tile_labels = tf.tile(self._input_labels, [self.max_num_steps, 1])
            self._xent = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, 
                                                                    tf.to_float(self._input_labels))
                                        ,name="neg_sample_loss")
            self._loss = self._xent + self._l2_penalty 
            
        with tf.name_scope("Summaries"):
            logit_mag = tf.histogram_summary("Logit_magnitudes", logits)
            l2 = tf.scalar_summary("L2_penalty", self._l2_penalty)
            ### find the l2 squared losses of each vector
            xent = tf.scalar_summary("Sigmoid_xent", self._xent)
            target_embed_mag = tf.histogram_summary("Target_Embed_L2", tf.reduce_sum(self._target_embeds**2, 1)/2.)
            state_mag = tf.histogram_summary("RNN_final_state_L2", tf.reduce_sum(self._final_states**2, 1)/2.)
            self._penalty_summary = tf.merge_summary([xent, target_embed_mag, state_mag, logit_mag, l2])
            self._train_cost_summary = tf.merge_summary([tf.scalar_summary("Train_NEG_Loss", self._loss)])
            self._valid_cost_summary = tf.merge_summary([tf.scalar_summary("Validation_NEG_Loss", self._loss)])
        
    def _build_classification_graph(self):
        # tf.get_variable_scope().reuse_variables()
        with tf.name_scope("Inputs"):
            # the naming x/y means we are trying to PREDICT x or y
            # so in_x_phrase is the one with y in the phrase, to predict x
            self._input_x_phrases = tf.placeholder(tf.int32, [None, self.max_num_steps, 3]) # [batch_size, w_{N:1}, 3]
            self._input_y_phrases = tf.placeholder(tf.int32, [None, self.max_num_steps, 3]) # [batch_size, w_{1:N}, 3]
            self._input_x_targets = tf.placeholder(tf.int32, [None, 1]) # [batch_size, w_x]
            self._input_y_targets = tf.placeholder(tf.int32, [None, 1]) # [batch_size, w_y]
            # labels, lengths, and keep prob are specified in `_forward_graph`
            batch_size = tf.shape(self._input_lengths)[0]

        with tf.name_scope("Embeddings"):
            x_input_embeds = tf.nn.dropout(tf.nn.embedding_lookup(self._word_embeddings, 
                                                  tf.slice(self._input_x_phrases, [0,0,0], [-1, -1, 1])),
                                         keep_prob=self._keep_prob)
            x_dep_embeds = tf.nn.dropout(tf.nn.embedding_lookup(self._dependency_embeddings,
                                                tf.slice(self._input_x_phrases, [0,0,1], [-1, -1, 1])),
                                       keep_prob=self._keep_prob)
            x_pos_embeds = tf.nn.dropout(tf.nn.embedding_lookup(self._pos_embeddings,
                                                tf.slice(self._input_x_phrases, [0,0,2], [-1, -1, 1])),
                                       keep_prob=self._keep_prob)

            y_input_embeds = tf.nn.dropout(tf.nn.embedding_lookup(self._word_embeddings, 
                                                  tf.slice(self._input_y_phrases, [0,0,0], [-1, -1, 1])),
                                         keep_prob=self._keep_prob)
            y_dep_embeds = tf.nn.dropout(tf.nn.embedding_lookup(self._dependency_embeddings,
                                                tf.slice(self._input_y_phrases, [0,0,1], [-1, -1, 1])),
                                       keep_prob=self._keep_prob)
            y_pos_embeds = tf.nn.dropout(tf.nn.embedding_lookup(self._pos_embeddings,
                                                tf.slice(self._input_y_phrases, [0,0,2], [-1, -1, 1])),
                                       keep_prob=self._keep_prob)

            x_target_embeds =  tf.nn.embedding_lookup(self._word_embeddings, 
                                                        tf.slice(self._input_x_targets, [0,0], [-1, 1]))
            y_target_embeds =  tf.nn.embedding_lookup(self._word_embeddings, 
                                                        tf.slice(self._input_y_targets, [0,0], [-1, 1]))
            self._x_target_embeds = tf.nn.dropout(tf.squeeze(x_target_embeds, [1]), keep_prob=self._keep_prob)
            self._y_target_embeds = tf.nn.dropout(tf.squeeze(y_target_embeds, [1]), keep_prob=self._keep_prob)

        with tf.name_scope("RNN"):
            x_input_words = [ tf.squeeze(input_, [1, 2]) for input_ in tf.split(1, self.max_num_steps, x_input_embeds)]
            x_input_deps = [ tf.squeeze(input_, [1, 2]) for input_ in tf.split(1, self.max_num_steps, x_dep_embeds)]
            x_input_pos = [ tf.squeeze(input_, [1, 2]) for input_ in tf.split(1, self.max_num_steps, x_pos_embeds)]
            x_inputs = [ tf.concat(1, [input_word, input_dep, input_pos_]) 
                       for (input_word, input_dep, input_pos_) in zip(x_input_words, x_input_deps, x_input_pos)]

            y_input_words = [ tf.squeeze(input_, [1, 2]) for input_ in tf.split(1, self.max_num_steps, y_input_embeds)]
            y_input_deps = [ tf.squeeze(input_, [1, 2]) for input_ in tf.split(1, self.max_num_steps, y_dep_embeds)]
            y_input_pos = [ tf.squeeze(input_, [1, 2]) for input_ in tf.split(1, self.max_num_steps, y_pos_embeds)]
            y_inputs = [ tf.concat(1, [input_word, input_dep, input_pos_]) 
                       for (input_word, input_dep, input_pos_) in zip(y_input_words, y_input_deps, y_input_pos)]

            if self.bidirectional:
                with tf.variable_scope("FW", reuse=True) as scope:
                    x_outs, x_state = tf.nn.rnn(self.fwcell, x_inputs, 
                                              sequence_length=tf.squeeze(self._input_lengths, [1]), 
                                              dtype=tf.float32, scope=scope)
                with tf.variable_scope("BW", reuse=True) as scope:
                    y_outs, y_state = tf.nn.rnn(self.bwcell, y_inputs, 
                                              sequence_length=tf.squeeze(self._input_lengths, [1]), 
                                              dtype=tf.float32, scope=scope)
            else:
                with tf.variable_scope("RNN", reuse=True) as scope:
                    x_outs, x_state = tf.nn.rnn(self.cell, x_inputs, 
                                                  sequence_length=tf.squeeze(self._input_lengths, [1]), 
                                                  dtype=tf.float32, scope=scope)
                    y_outs, y_state = tf.nn.rnn(self.cell, y_inputs, 
                                                  sequence_length=tf.squeeze(self._input_lengths, [1]), 
                                                  dtype=tf.float32, scope=scope)

            self._x_final_state = tf.nn.dropout(x_outs[-1], keep_prob=self._keep_prob)
            self._y_final_state = tf.nn.dropout(y_outs[-1], keep_prob=self._keep_prob)
 
        with tf.name_scope("Classifier"):
            self._class_lambda = tf.Variable(10e-3, trainable=False, name="Class_L2_Lambda")
            self._class_final_states = tf.concat(1, [self._x_final_state, self._y_final_state])
            self._class_target_embeds = tf.concat(1, [self._x_target_embeds, self._y_target_embeds])
            self._softmax_input = tf.concat(1, [self._class_final_states, self._class_target_embeds], 
                                            name="concat_input")

            ### REGULAR SOFTMAX ###
            # self._softmax_input = self._fin al_state # only predict using endpoints

            ### with a hidden layer
            # self._hidden_w = tf.get_variable("hidden_w", [self._softmax_input.get_shape()[1], self.hidden_layer_size])
            # self._hidden_b = tf.Variable(tf.zeros([self.hidden_layer_size], dtype=tf.float32), name="hidden_b")
            # self._scoring_w = tf.get_variable("scoring_w", [self.hidden_layer_size, self.num_classes])
            # self._scoring_b = tf.Variable(tf.zeros([self.num_classes], dtype=tf.float32), name="scoring_b")

            # hidden_logits = tf.nn.dropout(tf.nn.tanh(tf.nn.xw_plus_b(self._softmax_input, 
            #                                                          self._hidden_w, 
            #                                                          self._hidden_b)), 
            #                               keep_prob=self._keep_prob)
            # class_logits = tf.nn.xw_plus_b(hidden_logits, self._scoring_w,  self._scoring_b)
            # self._predictions = tf.argmax(class_logits, 1, name="predict")
            # self._predict_probs = tf.nn.softmax(class_logits, name="predict_probabilities")

            ### just softmax
            softmax_shape = [2*self.hidden_size + 2*self.word_embed_size, self.num_classes]
            self.score_w = tf.Variable(tf.random_uniform(softmax_shape, minval=-1.0, maxval=1.0), 
                                       name="score_w")
            self.score_bias = tf.Variable(tf.zeros([self.num_classes], dtype=tf.float32), name="score_bias")

            scores = tf.matmul(self._softmax_input, self.score_w) + self.score_bias
            self._predictions = tf.argmax(scores, 1, name="predict")
            self._predict_probs = tf.nn.softmax(scores, name="predict_probabilities")

            ### WORKING DIAG TENSOR INNER PRODUCT 2.0 ###
            # score(class c) = h^T * W_c * <w_x, w_y>
            # self.ws = [ tf.get_variable("score_w_"+str(i), [self.hidden_size])
            #             for i in range(self.num_classes) ]
            # scores = tf.concat(1,
            #                    [ tf.reduce_sum(tf.mul(self._final_state, tf.mul(w, self._target_embeds)), 
            #                                    1, keep_dims=True)
            #                      for w in self.ws ] )
            # print(scores.get_shape())
            # self._predictions = tf.argmax(scores, 1, name="predict")
            # self._predict_probs = tf.nn.softmax(scores, name="predict_probabilities")

            ## WORKING FULL TENSOR BILINEAR PRODUCT W/ LINEAR COMPONENT AND BIAS ###
            # self.ws = [ tf.get_variable("score_w_"+str(i), [self.hidden_size, self.hidden_size])
            #             for i in range(self.num_classes) ]
            # self.score_w = tf.get_variable("score_w", [self._softmax_input.get_shape()[1], self.num_classes])   
            # self.score_bias = tf.Variable(tf.zeros([self.num_classes], dtype=tf.float32), name="score_b")
            # scores = (tf.concat(1,
            #                    [ tf.squeeze( tf.batch_matmul( # [ batch x 1 x 1] -> [ batch x 1 ]
            #                                     tf.expand_dims(self._final_state, [1]), # [ batch x 1 x hidden ]
            #                                     tf.expand_dims( # [ batch x hidden x 1 ]
            #                                         tf.matmul(self._target_embeds, w), # [ batch x hidden ]
            #                                                   [2])),
            #                                     [2])
            #                      for w in self.ws ])
            #          + tf.matmul(self._softmax_input, self.score_w)
            #          + self.score_bias)
            # ## DO A TRANSFORM ON h ALSO ##
            # self.hs = [ tf.get_variable("score_h_"+str(i), [self.hidden_size, self.hidden_size])
            #             for i in range(self.num_classes) ]
            # scores = tf.concat(1,
            #                    [ tf.squeeze( tf.batch_matmul( # [ batch x 1 x 1] -> [ batch x 1 ]
            #                                     tf.expand_dims( # [ batch x hidden x 1 ]
            #                                         tf.matmul(self._final_state, h), # [ batch x hidden ]
            #                                                   [1]),
            #                                     tf.expand_dims( # [ batch x hidden x 1 ]
            #                                         tf.matmul(self._target_embeds, w), # [ batch x hidden ]
            #                                                   [2])),
            #                                 [2])
            #                      for h, w in zip(self.hs, self.ws) ])
            # print(scores.get_shape())
            # self._predictions = tf.argmax(scores, 1, name="predict")
            # self._predict_probs = tf.nn.softmax(scores, name="predict_probabilities")

            ### TENSOR TRIPLE PRODUCT ###
            # left_target, right_target = tf.split(1, 2, self._target_embeds)
            # self.ws = [ tf.get_variable("score_w_"+str(i), [self.hidden_size, self.word_embed_size, self.word_embed_size])
            #             for i in range(self.num_classes) ]
            # # # self.hs = [ tf.get_variable("score_h_"+str(i), [self.hidden_size, self.hidden_size])
            # # #             for i in range(self.num_classes) ]    
            # self.score_bias = tf.Variable(tf.zeros([self.num_classes], dtype=tf.float32), name="score_b")
            # # # scores = tf.concat(1,
            # # #                    [ tf.squeeze( tf.batch_matmul( # [ batch x 1 x 1] -> [ batch x 1 ]
            # # #                                     tf.expand_dims(self._final_state, [1]), # [ batch x 1 x hidden ]
            # # #                                     tf.expand_dims( # [ batch x hidden x 1 ]
            # # #                                         tf.matmul(self._target_embeds, w), # [ batch x hidden ]
            # # #                                                   [2])),
            # # #                                     [2])
            # # #                      for w in self.ws ])

            # scores = (tf.concat(1,
            #                    [ tf.expand_dims(batch_triple_inner(w, 
            #                                                        self._final_state, 
            #                                                        left_target, 
            #                                                        right_target),
            #                                     [1])

            #                    # tf.squeeze( tf.batch_matmul( # [ batch x 1 x 1] -> [ batch x 1 ]
            #                    #                  tf.expand_dims( # [ batch x hidden x 1 ]
            #                    #                      tf.matmul(self._final_state, h), # [ batch x hidden ]
            #                    #                                [1]),
            #                    #                  tf.expand_dims( # [ batch x hidden x 1 ]
            #                    #                      tf.matmul(self._target_embeds, w), # [ batch x hidden ]
            #                    #                                [2])),
            #                    #              [2])
            #                      for w in self.ws ])
            #          )#+ self.score_bias)
            # # scores += self.score_bias
            # # print(scores.get_shape())
            # self._predictions = tf.argmax(scores, 1, name="predict")
            # self._predict_probs = tf.nn.softmax(scores, name="predict_probabilities")
        
        with tf.name_scope("Loss"):
            self._class_labels = tf.placeholder(tf.int64, [None, 1])
            # self._class_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(class_logits, 
            #                                                                   tf.squeeze(self._class_labels, [1]))

            ### SOFTMAX CROSS ENTROPY ###
            self._class_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(scores, 
                                                                              tf.squeeze(self._class_labels, [1]))
            self._avg_class_loss = tf.reduce_mean(self._class_xent)

            ### MARGIN RANKING BASED ###

            self._class_l2 = self._class_lambda*(tf.nn.l2_loss(self.score_w)
                                                + tf.nn.l2_loss(self.score_bias))

            # self._class_l2 = self._class_lambda*(tf.nn.l2_loss(self._scoring_w)
            #                                     + tf.nn.l2_loss(self._scoring_b)
            #                                     + tf.nn.l2_loss(self._hidden_w)
            #                                     + tf.nn.l2_loss(self._hidden_b))

            # self._class_l2 = self._class_lambda*( tf.add_n([tf.nn.l2_loss(w) for w in self.ws])
            #                                     + tf.nn.l2_loss(self.score_w)
            #                                     # + tf.add_n([tf.nn.l2_loss(h) for h in self.hs])
            #                                     + tf.nn.l2_loss(self.score_bias))

            self._class_loss = self._avg_class_loss + self._class_l2

        with tf.name_scope("Summaries"):
            class_l2 = tf.scalar_summary("Classify_L2_penalty", self._class_l2)
            class_xent = tf.scalar_summary("Avg_Xent_Loss", self._avg_class_loss)
            target_embed_mag = tf.histogram_summary("Class_Target_Embed_L2", tf.nn.l2_loss(self._class_target_embeds))
            state_mag = tf.histogram_summary("Class_RNN_final_state_L2", tf.nn.l2_loss(self._class_final_states))
            self._class_penalty_summary = tf.merge_summary([class_l2, class_xent, target_embed_mag, state_mag])
            self._train_class_loss_summary = tf.merge_summary([tf.scalar_summary("Train_Avg_Class_Xent", self._avg_class_loss)])
            self._valid_class_loss_summary = tf.merge_summary([tf.scalar_summary("Valid_Avg_Class_Xent", self._avg_class_loss)])

    def _build_train_graph(self):
        with tf.name_scope("Unsupervised_Trainer"):
            self._global_step = tf.Variable(0, name="global_step", trainable=False)
#             self._lr = tf.Variable(1.0, trainable=False)
            self._optimizer = tf.train.AdamOptimizer(.001)
            
            # clip and apply gradients
            grads_and_vars = self._optimizer.compute_gradients(self._loss)
#             for gv in grads_and_vars:
#                 print(gv, gv[1] is self._cost)
            clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) 
                                      for gv in grads_and_vars if gv[0] is not None] # clip_by_norm doesn't like None
            
            with tf.name_scope("Summaries"):
                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                self._grad_summaries = tf.merge_summary(grad_summaries)
            self._train_op = self._optimizer.apply_gradients(clipped_grads_and_vars, global_step=self._global_step)
            
    def _build_class_train_graph(self):
        with tf.name_scope("Classification_Trainer"):
            self._class_global_step = tf.Variable(0, name="class_global_step", trainable=False)
#             self._lr = tf.Variable(1.0, trainable=False)
            self._class_optimizer = tf.train.AdamOptimizer(.001)
            
            # clip and apply gradients
            grads_and_vars = self._class_optimizer.compute_gradients(self._class_loss)
#             for gv in grads_and_vars:
#                 print(gv, gv[1] is self._cost)
            clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) 
                                      for gv in grads_and_vars if gv[0] is not None] # clip_by_norm doesn't like None
            
            with tf.name_scope("Summaries"):
                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.histogram_summary("class_{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.scalar_summary("class_{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                self._class_grad_summaries = tf.merge_summary(grad_summaries)
            self._class_train_op = self._class_optimizer.apply_gradients(clipped_grads_and_vars, 
                                                                         global_step=self._class_global_step)
            
    def _build_similarity_graph(self):
        # tf.get_variable_scope().reuse_variables()
        with tf.name_scope("Inputs"):
            # word or phrase we want similarities for
#             self._query_word = tf.placeholder(tf.int32, [1], name="q_word")
            self._query_phrase = tf.placeholder(tf.int32, [self.max_num_steps, 3], name="q_phrase")
            self._query_length = tf.placeholder(tf.int32, [1], name="q_len") # lengths for RNN
            self._query_target = tf.placeholder(tf.int32, [1,1], name="q_target")
            # words and phrases to compute similarities over
#             self._sim_words = tf.placeholder(tf.int32, [None, 1])
            self._sim_phrases = tf.placeholder(tf.int32, [None, self.max_num_steps, 3])
            self._sim_lengths = tf.placeholder(tf.int32, [None, 1]) # lengths for RNN
            self._sim_targets = tf.placeholder(tf.int32, [None, 1])
            sim_size = tf.shape(self._sim_lengths)[0]
        
        with tf.name_scope("Embeddings"):
            query_phrase_embed = tf.nn.embedding_lookup(self._word_embeddings, 
                                                  tf.slice(self._query_phrase, [0,0], [-1, 1]))
            query_dep_embed = tf.nn.embedding_lookup(self._dependency_embeddings,
                                                tf.slice(self._query_phrase, [0,1], [-1, 1]))
            query_pos_embed = tf.nn.embedding_lookup(self._pos_embeddings,
                                                tf.slice(self._query_phrase, [0,2], [-1, 1]))
            q_target_embed = tf.nn.embedding_lookup(self._target_embeddings, 
                                                        tf.slice(self._query_target, [0,0], [-1, 1]))
            q_target_embed = tf.squeeze(q_target_embed, [1])
#             query_word_embed = tf.nn.embedding_lookup(self._word_embeddings, self._query_word)
#             query_phrase_embed = tf.nn.embedding_lookup(self._word_embeddings, self._query_phrase)
#             sim_word_embed = tf.nn.embedding_lookup(self._word_embeddings, tf.squeeze(self._sim_words, [1]))
            sim_phrase_embed = tf.nn.embedding_lookup(self._word_embeddings, 
                                                  tf.slice(self._sim_phrases, [0, 0, 0], [-1, -1, 1]))
            sim_dep_embed = tf.nn.embedding_lookup(self._dependency_embeddings, 
                                                  tf.slice(self._sim_phrases, [0, 0, 1], [-1, -1, 1]))
            sim_pos_embed = tf.nn.embedding_lookup(self._pos_embeddings, 
                                                  tf.slice(self._sim_phrases, [0, 0, 2], [-1, -1, 1]))
            sim_target_embeds = tf.nn.embedding_lookup(self._target_embeddings, 
                                                        tf.slice(self._sim_targets, [0,0], [-1, 1]))
            sim_target_embeds = tf.squeeze(sim_target_embeds, [1])
        
        with tf.name_scope("RNN"):
            # compute rep of a query phrase
            query_phrase = [tf.squeeze(qw, [1]) for qw in tf.split(0, self.max_num_steps, query_phrase_embed)]
            query_dep = [tf.squeeze(qd, [1]) for qd in tf.split(0, self.max_num_steps, query_dep_embed)]
            query_pos = [tf.squeeze(qd, [1]) for qd in tf.split(0, self.max_num_steps, query_pos_embed)]

#             print(query_phrase[0].get_shape(), query_dep[0].get_shape())
            query_input = [ tf.concat(1, [qw, qd, qp]) for (qw, qd, qp) in zip(query_phrase, query_dep, query_pos)]

            # just words
            # query_input = query_phrase
            if self.bidirectional:
            #     outs = tf.nn.bidirectional_rnn(self.fwcell, self.bwcell, query_input, 
            #                             sequence_length=tf.to_int64(self._query_length),
            #                             dtype=tf.float32)
            #     # splice out the final forward and backward hidden states since apparently the documentation lies
            #     fw_state = tf.split(1, 2, outs[-1])[0]
            #     bw_state = tf.split(1, 2, outs[0])[1]
            #     query_phrase_state = tf.concat(1, [fw_state, bw_state])
                with tf.variable_scope("FW", reuse=True) as scope:
                    _, query_phrase_state = tf.nn.rnn(self.fwcell, query_input, 
                                              sequence_length=tf.to_int64(self._query_length), 
                                              dtype=tf.float32, scope=scope)
            else:
                with tf.variable_scope("RNN", reuse=True) as scope:
                    _, query_phrase_state = tf.nn.rnn(self.cell, query_input, 
                                                  sequence_length=tf.to_int64(self._query_length), 
                                                  dtype=tf.float32, scope=scope)

            # compute reps of similarity phrases
            sim_phrases = [tf.squeeze(qw, [1,2]) for qw in tf.split(1, self.max_num_steps, sim_phrase_embed)]
            sim_deps = [tf.squeeze(qd, [1,2]) for qd in tf.split(1, self.max_num_steps, sim_dep_embed)]
            sim_pos = [tf.squeeze(qp, [1,2]) for qp in tf.split(1, self.max_num_steps, sim_pos_embed)]

            sim_input = [ tf.concat(1, [qw, qd, qp]) for (qw, qd, qp) in zip(sim_phrases, sim_deps, sim_pos)]

            #jsut words
            # sim_input = sim_phrases
            if self.bidirectional:
                with tf.variable_scope("FW", reuse=True) as scope:
                    _, sim_phrase_states = tf.nn.rnn(self.fwcell, sim_input, 
                                                 sequence_length=tf.to_int64(tf.squeeze(self._sim_lengths, [1])), 
                                                 dtype=tf.float32, scope=scope)
                # outs = tf.nn.bidirectional_rnn(self.fwcell, self.bwcell, sim_input, 
                #                         sequence_length=tf.to_int64(tf.squeeze(self._sim_lengths, [1])),
                #                         dtype=tf.float32)
                # # splice out the final forward and backward hidden states since apparently the documentation lies
                # fw_state = tf.split(1, 2, outs[-1])[0]
                # bw_state = tf.split(1, 2, outs[0])[1]
                # sim_phrase_states = tf.concat(1, [fw_state, bw_state])
            else:
                with tf.variable_scope("RNN", reuse=True) as scope:
                    _, sim_phrase_states = tf.nn.rnn(self.cell, sim_input, 
                                                 sequence_length=tf.to_int64(tf.squeeze(self._sim_lengths, [1])), 
                                                 dtype=tf.float32, scope=scope)
            
        with tf.name_scope("Similarities"):
            with tf.name_scope("Normalize"):

                # query_phrase = tf.nn.l2_normalize(tf.concat(1, [query_phrase_state, q_target_embed]), 1)
                query_phrase = tf.nn.l2_normalize(query_phrase_state, 1)
#                 query_word = tf.nn.l2_normalize(query_word_embed, 1)
                # sim_phrases = tf.nn.l2_normalize(tf.concat(1, [sim_phrase_states, sim_target_embeds]), 1)
                sim_phrases = tf.nn.l2_normalize(sim_phrase_states, 1)
#                 sim_word = tf.nn.l2_normalize(sim_word_embed, 1)                  

            with tf.name_scope("Calc_distances"):
                # do for words
#                 print(q)
#                 query_word_nearby_dist = tf.matmul(query_word, sim_word, transpose_b=True)
#                 qw_nearby_val, qw_nearby_idx = tf.nn.top_k(query_word_nearby_dist, min(1000, self.vocab_size))
#                 self.qw_nearby_val = tf.squeeze(qw_nearby_val)
#                 self.qw_nearby_idx = tf.squeeze(qw_nearby_idx)
#                 self.qw_nearby_words = tf.squeeze(tf.gather(self._sim_words, qw_nearby_idx))

                # do for phrases
                query_phrase_nearby_dist = tf.matmul(query_phrase, sim_phrases, transpose_b=True)
                qp_nearby_val, qp_nearby_idx = tf.nn.top_k(query_phrase_nearby_dist, min(1000, sim_size))
#                 self.sanity_check = tf.squeeze(tf.matmul(query_phrase, query_phrase, transpose_b=True))
                self.qp_nearby_val = tf.squeeze(qp_nearby_val)
                self.qp_nearby_idx = tf.squeeze(qp_nearby_idx)
#                 self.qp_nearby_lens = tf.squeeze(tf.gather(self._sim_lengths, qp_nearby_idx))
            
    def partial_class_fit(self, x_input_phrases, y_input_phrases,
                                x_input_targets, y_input_targets,
                                class_labels, input_lengths, keep_prob=.5):
        """Fit a mini-batch
        
        Expects a batch_x: [self.batch_size, self.max_num_steps]
                  batch_y: the same
                  batch_seq_lens: [self.batch_size]
                  
        Returns average batch perplexity
        """
        loss, xent, _, g_summaries, c_summary, p_summary = self.session.run([self._class_loss, self._avg_class_loss,
                                                            self._class_train_op, 
                                                            self._class_grad_summaries,
                                                            self._train_class_loss_summary,
                                                            self._class_penalty_summary],
                                                           {self._input_x_phrases:x_input_phrases,
                                                            self._input_y_phrases:y_input_phrases,
                                                            self._input_x_targets:x_input_targets,
                                                            self._input_y_targets:y_input_targets,
                                                            self._class_labels:class_labels,
                                                            self._input_lengths:input_lengths,
                                                            self._keep_prob:keep_prob})
        self.summary_writer.add_summary(g_summaries)
        self.summary_writer.add_summary(c_summary)
        self.summary_writer.add_summary(p_summary)
        return loss, xent
    
    def partial_unsup_fit(self, input_phrases, input_targets, 
                         input_labels, input_lengths, input_predict_x,
                         keep_prob=.5):
        """Fit a mini-batch
        
        Expects a batch_x: [self.batch_size, self.max_num_steps]
                  batch_y: the same
                  batch_seq_lens: [self.batch_size]
                  
        Returns average batch perplexity
        """
        loss, xent, _, g_summaries, c_summary, p_summary = self.session.run([self._loss, self._xent, self._train_op, 
                                                            self._grad_summaries,
                                                            self._train_cost_summary,
                                                            self._penalty_summary],
                                                           {self._input_phrases:input_phrases,
                                                            self._input_targets:input_targets,
                                                            self._input_labels:input_labels,
                                                            self._input_lengths:input_lengths,
                                                            self._input_predict_x:input_predict_x,
                                                            self._keep_prob:keep_prob})
        self.summary_writer.add_summary(g_summaries)
        self.summary_writer.add_summary(c_summary)
        self.summary_writer.add_summary(p_summary)
        return loss, xent
    
    def validation_loss(self, valid_phrases, valid_targets,     
                        valid_labels, valid_lengths, valid_predict_x):
        """Calculate loss on validation inputs, but don't run trainer"""
        loss, v_summary = self.session.run([self._loss, self._valid_cost_summary],
                                           {self._input_phrases:valid_phrases,
                                            self._input_targets:valid_targets,
                                            self._input_labels:valid_labels,
                                            self._input_lengths:valid_lengths,
                                            self._input_predict_x:valid_predict_x,
                                            self._keep_prob:1.0})
        self.summary_writer.add_summary(v_summary)
        return loss
    
    def validation_class_loss(self, valid_x_phrases, valid_y_phrases, 
                              valid_x_targets, valid_y_targets, 
                              valid_labels, valid_lengths):
        """Calculate loss on validation inputs, but don't run trainer"""
        loss, xent, v_summary = self.session.run([self._class_loss, self._avg_class_loss, self._valid_class_loss_summary],
                                           {self._input_x_phrases:valid_x_phrases,
                                            self._input_y_phrases:valid_y_phrases,
                                            self._input_x_targets:valid_x_targets,
                                            self._input_y_targets:valid_y_targets,
                                            self._class_labels:valid_labels,
                                            self._input_lengths:valid_lengths,
                                                            self._keep_prob:1.0})
        self.summary_writer.add_summary(v_summary)
        return loss, xent
    
    def validation_phrase_nearby(self, q_phrase, q_phrase_len, q_target, sim_phrases, sim_phrase_lens, sim_targets):
        """Return nearby phrases from the similarity set
        """
        # TODO: Input predict_x to decide which RNN to use
        nearby_vals, nearby_idx = self.session.run([self.qp_nearby_val, self.qp_nearby_idx],
                                                           {self._query_phrase:q_phrase, 
                                                            self._query_length:q_phrase_len,
                                                            self._query_target:q_target,
                                                            self._sim_phrases:sim_phrases,
                                                            self._sim_lengths:sim_phrase_lens,
                                                            self._sim_targets:sim_targets,
                                                            self._keep_prob:1.0})
#         print("Sanity check: %r" % sanity)
        return nearby_vals, nearby_idx
    
    def embed_phrases_and_targets(self, phrases, targets, lengths):
        phrase_reps, target_reps = self.session.run([self._final_state, self._target_embeds],
                                                    { self._input_phrases:phrases,
                                                      self._input_targets:targets,
                                                      self._input_lengths:lengths,
                                                            self._keep_prob:1.0})
        return phrase_reps, target_reps
    
#     def validation_word_nearby(self, q_word, sim_words):
#         """Return nearby phrases from the similarity set
#         """
#         nearby_vals, nearby_idx = self.session.run([self.qw_nearby_val, 
#                                                       self.qw_nearby_idx],
#                                                        {self._query_word:q_word, 
#                                                         self._sim_words:sim_words})
#         return nearby_vals, nearby_idx
        
    def predict(self, x_input_phrases, y_input_phrases,
                      x_input_targets, y_input_targets,
                      input_lengths, return_probs=False):
        if return_probs:
            predictions, distributions = self.session.run([self._predictions, self._predict_probs],
                                                          {self._input_x_phrases:x_input_phrases,
                                                           self._input_y_phrases:y_input_phrases,
                                                           self._input_x_targets:x_input_targets,
                                                           self._input_y_targets:y_input_targets,
                                                           self._input_lengths:input_lengths,
                                                           self._keep_prob:1.0})
            distributions = distributions.reshape([path_lens.shape[0], -1])
            #predictions are 2d array w/ one col
            return list(predictions), list(distributions) 
        
        else:
            predictions = self.session.run(self._predictions,
                                           {self._input_x_phrases:x_input_phrases,
                                           self._input_y_phrases:y_input_phrases,
                                           self._input_x_targets:x_input_targets,
                                           self._input_y_targets:y_input_targets,
                                           self._input_lengths:input_lengths,
                                           self._keep_prob:1.0})
            return list(predictions)
            
    def checkpoint(self):
        if not self.config['supervised']:
            save_name = (self.checkpoint_prefix + '.ckpt-'+str(self._global_step.eval()))
        else:
            save_name = (self.checkpoint_prefix + '.ckpt-'+str(self._global_step.eval())+'-'+str(self._class_global_step.eval()))

        print("Saving model to file: %s" %  save_name)
        self.saver.save(self.session, save_name)
        return save_name

    def restore(self, model_ckpt_path):
        self.saver.restore(self.session, model_ckpt_path)

    def restore_unsupervised(self, model_ckpt_path):
        """ Restore the unsupervised components from another RNN"""
        # TODO: run all of the ssign statements in the session should make it work
        # create a new one with the same configuration
        name = model_ckpt_path.split('/')[1].split('-')[0].split('.')[0]
        config = self.config
        print('name: ', name)
        config['model_name'] = name
        config['interactive'] = False
        config['supervised'] = False

        # get the outer RNN vars
        # with tf.variable_scope('RNN/GRUCell/Gates/Linear', reuse=True):
        #     gate_matrix = tf.get_variable('Matrix')
        #     gate_bias = tf.get_variable('Bias')
        # with tf.variable_scope('RNN/GRUCell/Candidate/Linear', reuse=True):
        #     cand_matrix = tf.get_variable('Matrix')
        #     cand_bias = tf.get_variable('Bias')
        # use a new graph
        g = tf.Graph()
        with g.as_default():
            unsup = RelEmbed(config)
            unsup.restore(model_ckpt_path)
            # for op in g.get_operations():
            #     print(op.name)
        self._word_embeddings.assign(unsup.session.run(unsup._word_embeddings))
        self._dependency_embeddings.assign(unsup.session.run(unsup._dependency_embeddings))
        self._pos_embeddings.assign(unsup.session.run(unsup._pos_embeddings))
        self._left_target_embeddings.assign(unsup.session.run(unsup._left_target_embeddings))
        self._right_target_embeddings.assign(unsup.session.run(unsup._right_target_embeddings))
        # do the RNN linear vars
        # tf.get_variable_scope().reuse_variables()
        self._gate_matrix.assign(unsup.session.run(unsup._gate_matrix))
        self._gate_bias.assign(unsup.session.run(unsup._gate_bias))
        self._cand_matrix.assign(unsup.session.run(unsup._cand_matrix))
        self._cand_bias.assign(unsup.session.run(unsup._cand_bias))
        unsup.session.close()
        del unsup

    def random_restart_score_weights(self):
        random_w = np.random.uniform(low=-.5, high=.5, size=(2*self.hidden_size + 2*self.word_embed_size, self.num_classes))
        zero_bias = np.zeros(self.num_classes)
        self.session.run([self.score_w.assign(random_w),
                          self.score_bias.assign(zero_bias)])
        
    def __repr__(self):
        return ("<DPNN: W:%i, D:%i, P:%i H:%i, V:%i>" 
                % (self.word_embed_size, self.dep_embed_size, self.pos_embed_size,
                    self.hidden_size, self.vocab_size))

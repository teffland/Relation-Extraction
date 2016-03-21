"""
Relation Embed model
"""
from __future__ import print_function
import numpy as np
import tensorflow as tf

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
#         self.batch_size = config['batch_size']
        self.max_num_steps = config['max_num_steps']
        self.word_embed_size = config['word_embed_size']
        self.dep_embed_size = config['dep_embed_size']
        self.input_size = self.word_embed_size + self.dep_embed_size
        self.hidden_size = 2 * self.word_embed_size #config['hidden_size']
        self.pretrained_word_embeddings = config['pretrained_word_embeddings'] # None if we don't provide them
        if np.any(self.pretrained_word_embeddings):
            assert self.word_embed_size == self.pretrained_word_embeddings.shape[1]
        self.num_classes = config['num_predict_classes']
        self.max_grad_norm = config['max_grad_norm']
        
        self.vocab_size = config['vocab_size']
        self.dep_vocab_size = config['dep_vocab_size']
        self.name = config['model_name']
        self.checkpoint_prefix = config['checkpoint_prefix'] + self.name
        self.summary_prefix = config['summary_prefix'] + self.name
        
        self.initializer = tf.random_uniform_initializer(-1., 1.)
        with tf.name_scope(self.name):
            with tf.name_scope("Forward"):
                self._build_forward_graph()
                self._build_classification_graph()
            with tf.name_scope("Backward"):
                self._build_train_graph()
                self._build_class_train_graph()
            with tf.name_scope("Nearby"):
                self._build_similarity_graph()
        
        self._valid_accuracy = tf.Variable(0.0, trainable=False)
        self._valid_acc_summary = tf.merge_summary([tf.scalar_summary("Valid_accuracy", self._valid_accuracy)])

        self.saver = tf.train.Saver(tf.all_variables())
            
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())        
        self.summary_writer = tf.train.SummaryWriter(self.summary_prefix, self.session.graph_def)
        
    def save_validation_accuracy(self, new_score):
        assign_op = self._valid_accuracy.assign(new_score)
        _, summary = self.session.run([assign_op, self._valid_acc_summary])
        self.summary_writer.add_summary(summary)
        
    def _build_forward_graph(self):
        # input tensor of zero padded indices to get to max_num_steps
        # None allows for variable batch sizes
        self._lambda = tf.Variable(.001, trainable=False, name="L2_Lambda")
        with tf.name_scope("Inputs"):
            self._input_phrases = tf.placeholder(tf.int32, [None, self.max_num_steps, 2]) # [batch_size, w_{1:N}, 2]
            self._input_targets = tf.placeholder(tf.int32, [None, 2]) # [batch_size, w_x]
            self._input_labels = tf.placeholder(tf.int32, [None, 1]) # [batch_size, from true data?] \in {0,1}
            self._input_lengths = tf.placeholder(tf.int32, [None, 1]) # [batch_size, N] (len of each sequence)
            batch_size = tf.shape(self._input_lengths)[0]
            self._keep_prob = tf.placeholder(tf.float32)
        
        with tf.name_scope("Embeddings"):
            if np.any(self.pretrained_word_embeddings):
                self._word_embeddings = tf.Variable(self.pretrained_word_embeddings,name="word_embeddings")
                self._left_target_embeddings = tf.Variable(self.pretrained_word_embeddings, name="left_target_embeddings")
                self._right_target_embeddings = tf.Variable(self.pretrained_word_embeddings, name="right_target_embeddings")
            else:
                self._word_embeddings = tf.get_variable("word_embeddings", 
                                                        [self.vocab_size, self.word_embed_size],
                                                        dtype=tf.float32)
                self._left_target_embeddings = tf.get_variable("left_target_embeddings", 
                                                        [self.vocab_size, self.word_embed_size],
                                                        dtype=tf.float32)
                self._right_target_embeddings = tf.get_variable("right_target_embeddings", 
                                                        [self.vocab_size, self.word_embed_size],
                                                        dtype=tf.float32)
            
            self._dependency_embeddings = tf.get_variable("dependency_embeddings", 
                                                    [self.dep_vocab_size, self.dep_embed_size],
                                                    dtype=tf.float32)
            # TODO: Add POS embeddings
            
            input_embeds = tf.nn.dropout(tf.nn.embedding_lookup(self._word_embeddings, 
                                                  tf.slice(self._input_phrases, [0,0,0], [-1, -1, 1])),
                                         keep_prob=self._keep_prob)
            dep_embeds = tf.nn.dropout(tf.nn.embedding_lookup(self._dependency_embeddings,
                                                tf.slice(self._input_phrases, [0,0,1], [-1, -1, 1])),
                                       keep_prob=self._keep_prob)
            left_target_embeds = tf.nn.dropout(tf.nn.embedding_lookup(self._left_target_embeddings, 
                                                        tf.slice(self._input_targets, [0,0], [-1, 1])),
                                                keep_prob=self._keep_prob)
            right_target_embeds = tf.nn.dropout(tf.nn.embedding_lookup(self._right_target_embeddings, 
                                                        tf.slice(self._input_targets, [0,1], [-1, 1])),
                                                 keep_prob=self._keep_prob)
#             print(tf.slice(self._input_phrases, [0,0,1], [-1, -1, 1]).get_shape(), dep_embeds.get_shape())
#             print(left_target_embeds.get_shape(), right_target_embeds.get_shape())
            self._target_embeds = tf.squeeze(tf.concat(2, [left_target_embeds, right_target_embeds]), [1])
#             print(target_embeds.get_shape())
            # TODO: Add dropout to embeddings
        
        with tf.name_scope("RNN"):
            # start off with a basic configuration
            self.cell = tf.nn.rnn_cell.GRUCell(self.hidden_size, 
                                                input_size=self.input_size)
            # TODO: Make it multilevel
#             self._initial_state = self.cell.zero_state(batch_size, tf.float32)
#             print(self._initial_state.get_shape())
            input_words = [ tf.squeeze(input_, [1, 2]) for input_ in tf.split(1, self.max_num_steps, input_embeds)]
            input_deps = [ tf.squeeze(input_, [1, 2]) for input_ in tf.split(1, self.max_num_steps, dep_embeds)]
            inputs = [ tf.concat(1, [input_word, input_dep]) 
                      for (input_word, input_dep) in zip(input_words, input_deps)]

            _, state = tf.nn.rnn(self.cell, inputs, 
                                 sequence_length=tf.squeeze(self._input_lengths, [1]),
                                 dtype=tf.float32)
#                                  initial_state=self._initial_state)
            self._final_state = tf.nn.dropout(state, keep_prob= self._keep_prob)
            
        with tf.name_scope("Loss"):
            flat_states = tf.reshape(state, [-1])
            flat_target_embeds = tf.reshape(self._target_embeds, [-1])
#             assert self.hidden_size == (self.word_embed_size), "Hidden state must equal concated inputs" 
            flat_logits = tf.mul(flat_states, flat_target_embeds)
            logits = tf.reduce_sum(tf.reshape(flat_logits, tf.pack([batch_size, -1])), 1)
            self._l2_penalty = self._lambda*(tf.nn.l2_loss(self._word_embeddings)
                                            +tf.nn.l2_loss(self._dependency_embeddings)
                                            +tf.nn.l2_loss(self._left_target_embeddings)
                                            +tf.nn.l2_loss(self._right_target_embeddings))
            self._xent = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, 
                                                                    tf.to_float(self._input_labels)),
                                        name="neg_sample_loss")
            self._loss = self._xent + self._l2_penalty 
            
        with tf.name_scope("Summaries"):
            logit_mag = tf.histogram_summary("Logit_magnitudes", logits)
            l2 = tf.scalar_summary("L2_penalty", self._l2_penalty)
            xent = tf.scalar_summary("Sigmoid_xent", self._xent)
            target_embed_mag = tf.histogram_summary("Target_Embed_L2", tf.nn.l2_loss(self._target_embeds))
            state_mag = tf.histogram_summary("RNN_final_state_L2", tf.nn.l2_loss(self._final_state))
            self._penalty_summary = tf.merge_summary([logit_mag, l2, xent, target_embed_mag, state_mag])
            self._train_cost_summary = tf.merge_summary([tf.scalar_summary("Train_NEG_Loss", self._loss)])
            self._valid_cost_summary = tf.merge_summary([tf.scalar_summary("Validation_NEG_Loss", self._loss)])
        
    def _build_classification_graph(self):
        with tf.name_scope("Classifier"):
            self._class_lambda = tf.Variable(0.001, trainable=False, name="Class_L2_Lambda")
            self._softmax_input = tf.concat(1, [self._final_state, self._target_embeds], name="concat_input")
            self._softmax_w = tf.get_variable("softmax_w", [self._softmax_input.get_shape()[1], self.num_classes])
            self._softmax_b = tf.Variable(tf.zeros([self.num_classes], dtype=tf.float32), name="softmax_b")

            class_logits = tf.matmul(self._softmax_input, self._softmax_w) + self._softmax_b
            self._predictions = tf.argmax(class_logits, 1, name="predict")
            self._predict_probs = tf.nn.softmax(class_logits, name="predict_probabilities")
        
        with tf.name_scope("Loss"):
            self._class_labels = tf.placeholder(tf.int64, [None, 1])
            self._class_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(class_logits, 
                                                                              tf.squeeze(self._class_labels, [1]))
            self._avg_class_loss = tf.reduce_mean(self._class_xent)
            self._class_l2 = self._class_lambda*(tf.nn.l2_loss(self._softmax_w)
                                                + tf.nn.l2_loss(self._softmax_b))

            self._class_loss = self._avg_class_loss + self._class_l2

        with tf.name_scope("Summaries"):
            class_l2 = tf.scalar_summary("Classify_L2_penalty", self._class_l2)
            class_xent = tf.scalar_summary("Avg_Xent_Loss", self._avg_class_loss)
            self._class_penalty_summary = tf.merge_summary([class_l2, class_xent])
            self._train_class_loss_summary = tf.merge_summary([tf.scalar_summary("Train_Avg_Class_Xent", self._avg_class_loss)])
            self._valid_class_loss_summary = tf.merge_summary([tf.scalar_summary("Valid_Avg_Class_Xent", self._avg_class_loss)])

    def _build_train_graph(self):
        with tf.name_scope("Unsupervised_Trainer"):
            self._global_step = tf.Variable(0, name="global_step", trainable=False)
#             self._lr = tf.Variable(1.0, trainable=False)
            self._optimizer = tf.train.AdagradOptimizer(1.0)
            
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
            self._class_optimizer = tf.train.AdagradOptimizer(1.0)
            
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
        tf.get_variable_scope().reuse_variables()
        with tf.name_scope("Inputs"):
            # word or phrase we want similarities for
#             self._query_word = tf.placeholder(tf.int32, [1], name="q_word")
            self._query_phrase = tf.placeholder(tf.int32, [self.max_num_steps, 2], name="q_phrase")
            self._query_length = tf.placeholder(tf.int32, [1], name="q_len") # lengths for RNN
            # words and phrases to compute similarities over
#             self._sim_words = tf.placeholder(tf.int32, [None, 1])
            self._sim_phrases = tf.placeholder(tf.int32, [None, self.max_num_steps, 2])
            self._sim_lengths = tf.placeholder(tf.int32, [None, 1]) # lengths for RNN
            sim_size = tf.shape(self._sim_lengths)[0]
        
        with tf.name_scope("Embeddings"):
            query_phrase_embed = tf.nn.embedding_lookup(self._word_embeddings, 
                                                  tf.slice(self._query_phrase, [0,0], [-1, 1]))
            query_dep_embed = tf.nn.embedding_lookup(self._dependency_embeddings,
                                                tf.slice(self._query_phrase, [0,1], [-1, 1]))
#             query_word_embed = tf.nn.embedding_lookup(self._word_embeddings, self._query_word)
#             query_phrase_embed = tf.nn.embedding_lookup(self._word_embeddings, self._query_phrase)
#             sim_word_embed = tf.nn.embedding_lookup(self._word_embeddings, tf.squeeze(self._sim_words, [1]))
            sim_phrase_embed = tf.nn.embedding_lookup(self._word_embeddings, 
                                                  tf.slice(self._sim_phrases, [0, 0, 0], [-1, -1, 1]))
            sim_dep_embed = tf.nn.embedding_lookup(self._dependency_embeddings, 
                                                  tf.slice(self._sim_phrases, [0, 0, 1], [-1, -1, 1]))
        
        with tf.name_scope("RNN"):
            # compute rep of a query phrase
            query_phrase = [tf.squeeze(qw, [1]) for qw in tf.split(0, self.max_num_steps, query_phrase_embed)]
            query_dep = [tf.squeeze(qd, [1]) for qd in tf.split(0, self.max_num_steps, query_dep_embed)]
#             print(query_phrase[0].get_shape(), query_dep[0].get_shape())
            query_input = [ tf.concat(1, [qw, qd]) for (qw, qd) in zip(query_phrase, query_dep)]
            _, query_phrase_state = tf.nn.rnn(self.cell, query_input, 
                                              sequence_length=self._query_length, 
                                              dtype=tf.float32)
            # compute reps of similarity phrases
            sim_phrases = [tf.squeeze(qw, [1,2]) for qw in tf.split(1, self.max_num_steps, sim_phrase_embed)]
            sim_deps = [tf.squeeze(qd, [1,2]) for qd in tf.split(1, self.max_num_steps, sim_dep_embed)]
            sim_input = [ tf.concat(1, [qw, qd]) for (qw, qd) in zip(sim_phrases, sim_deps)]
            _, sim_phrase_states = tf.nn.rnn(self.cell, sim_input, 
                                             sequence_length=tf.squeeze(self._sim_lengths, [1]), 
                                             dtype=tf.float32)
            
        with tf.name_scope("Similarities"):
            with tf.name_scope("Normalize"):
#                 print(query_phrase.get_shape())
                query_phrase = tf.nn.l2_normalize(query_phrase_state, 1)
#                 query_word = tf.nn.l2_normalize(query_word_embed, 1)
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
            
    def partial_class_fit(self, input_phrases, input_targets, class_labels, input_lengths):
        """Fit a mini-batch
        
        Expects a batch_x: [self.batch_size, self.max_num_steps]
                  batch_y: the same
                  batch_seq_lens: [self.batch_size]
                  
        Returns average batch perplexity
        """
        loss, _, g_summaries, c_summary, p_summary = self.session.run([self._class_loss, self._class_train_op, 
                                                            self._class_grad_summaries,
                                                            self._train_class_loss_summary,
                                                            self._class_penalty_summary],
                                                           {self._input_phrases:input_phrases,
                                                            self._input_targets:input_targets,
                                                            self._class_labels:class_labels,
                                                            self._input_lengths:input_lengths,
                                                            self._keep_prob:.5})
        self.summary_writer.add_summary(g_summaries)
        self.summary_writer.add_summary(c_summary)
        self.summary_writer.add_summary(p_summary)
        return loss
    
    def partial_unsup_fit(self, input_phrases, input_targets, input_labels, input_lengths):
        """Fit a mini-batch
        
        Expects a batch_x: [self.batch_size, self.max_num_steps]
                  batch_y: the same
                  batch_seq_lens: [self.batch_size]
                  
        Returns average batch perplexity
        """
        loss, _, g_summaries, c_summary, p_summary = self.session.run([self._loss, self._train_op, 
                                                            self._grad_summaries,
                                                            self._train_cost_summary,
                                                            self._penalty_summary],
                                                           {self._input_phrases:input_phrases,
                                                            self._input_targets:input_targets,
                                                            self._input_labels:input_labels,
                                                            self._input_lengths:input_lengths,
                                                            self._keep_prob:.5})
        self.summary_writer.add_summary(g_summaries)
        self.summary_writer.add_summary(c_summary)
        self.summary_writer.add_summary(p_summary)
        return loss
    
    def validation_loss(self, valid_phrases, valid_targets, valid_labels, valid_lengths):
        """Calculate loss on validation inputs, but don't run trainer"""
        loss, v_summary = self.session.run([self._loss, self._valid_cost_summary],
                                           {self._input_phrases:valid_phrases,
                                            self._input_targets:valid_targets,
                                            self._input_labels:valid_labels,
                                            self._input_lengths:valid_lengths,
                                                            self._keep_prob:1.0})
        self.summary_writer.add_summary(v_summary)
        return loss
    
    def validation_class_loss(self, valid_phrases, valid_targets, valid_labels, valid_lengths):
        """Calculate loss on validation inputs, but don't run trainer"""
        loss, v_summary = self.session.run([self._avg_class_loss, self._valid_class_loss_summary],
                                           {self._input_phrases:valid_phrases,
                                            self._input_targets:valid_targets,
                                            self._class_labels:valid_labels,
                                            self._input_lengths:valid_lengths,
                                                            self._keep_prob:1.0})
        self.summary_writer.add_summary(v_summary)
        return loss
    
    def validation_phrase_nearby(self, q_phrase, q_phrase_len, sim_phrases, sim_phrase_lens):
        """Return nearby phrases from the similarity set
        """
        nearby_vals, nearby_idx = self.session.run([self.qp_nearby_val, self.qp_nearby_idx],
                                                                   {self._query_phrase:q_phrase, 
                                                                    self._query_length:q_phrase_len,
                                                                    self._sim_phrases:sim_phrases,
                                                                    self._sim_lengths:sim_phrase_lens,
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
        
    def predict(self, paths, targets, path_lens, return_probs=False):
        if return_probs:
            predictions, distributions = self.session.run([self._predictions, self._predict_probs],
                                                          {self._input_phrases:paths,
                                                           self._input_targets:targets,
                                                           self._input_lengths:path_lens,
                                                            self._keep_prob:1.0})
            distributions = distributions.reshape([path_lens.shape[0], -1])
            #predictions are 2d array w/ one col
            print("pred shape:", predictions.shape)
            return list(predictions), list(distributions) 
        
        else:
            predictions = self.session.run(self._predictions,
                                           {self._input_phrases:paths,
                                            self._input_targets:targets,
                                            self._input_lengths:path_lens,
                                            self._keep_prob:1.0})
            print("pred shape:", predictions.shape)
            return list(predictions)
            
    def checkpoint(self):
        save_name = (self.checkpoint_prefix + '.ckpt-'+str(self._global_step.eval())+'-'+str(self._class_global_step.eval()))
        print("Saving model to file: %s" %  save_name)
        self.saver.save(self.session, save_name)
        return save_name

    def restore(self, model_ckpt_path):
        self.saver.restore(self.session, model_ckpt_path)
        
    def __repr__(self):
        return ("<DPNN: W:%i, D:%i, H:%i, V:%i>" 
                % (self.word_embed_size, self.dep_embed_size, self.hidden_size, self.vocab_size))

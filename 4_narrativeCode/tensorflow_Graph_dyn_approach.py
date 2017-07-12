import tensorflow as tf 
import numpy as np
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope

def seq2seq(batch_size=100,
	        enc_input_dimension=512,
            enc_timesteps_max=10,
            dec_timesteps_max=50,
			hidden_units=128,
            hidden_layers=2,
            input_embedding_size=300,
            vocab_size=30000):

	# Inputs / Outputs / Cells
	with variable_scope.variable_scope("IO"):
		# Placeholders
		encoder_inputs = tf.placeholder(dtypes.float32, shape=[batch_size, enc_timesteps_max, enc_input_dimension], name="encoder_inputs")
		encoder_lengths = tf.placeholder(dtypes.int32, shape=[batch_size], name="encoder_lengths")
		decoder_inputs = tf.placeholder(dtypes.int64, shape=[batch_size, dec_timesteps_max], name="decoder_inputs")
		decoder_lengths = tf.placeholder(dtypes.int32, shape=[batch_size], name="decoder_lengths")
		decoder_outputs = tf.placeholder(dtypes.int64, shape=[batch_size, dec_timesteps_max], name="decoder_outputs")
		masking = tf.placeholder(dtypes.float32, shape=[batch_size, dec_timesteps_max], name="loss_masking")

		# Cells
		encoder_cell = single_cell_enc = tf.contrib.rnn.LSTMCell(hidden_units)
		if hidden_layers > 1:
			encoder_cell = tf.contrib.rnn.MultiRNNCell([single_cell_enc] * hidden_layers)
		decoder_cell = single_cell_dec = tf.contrib.rnn.LSTMCell(hidden_units)
		if hidden_layers > 1:
			decoder_cell = tf.contrib.rnn.MultiRNNCell([single_cell_dec] * hidden_layers)

	# Seq2Seq
	with variable_scope.variable_scope("seq2seq_Model"):
		
		# Encoder
		_, initial_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs, sequence_length=encoder_lengths, dtype=tf.float32, scope = "rnn_encoder")

		# Decoder
		with variable_scope.variable_scope("rnn_decoder"):
			outputs = []
			embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
			decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)
			lstm_output,state = tf.nn.dynamic_rnn(decoder_cell, decoder_inputs_embedded, sequence_length=decoder_lengths, initial_state=initial_state)
			transp = tf.transpose(lstm_output, [1, 0, 2])
			lstm_output_unpacked = tf.unstack(transp)
			for item in lstm_output_unpacked:
				logits = tf.layers.dense(inputs=item, units=vocab_size)
				outputs.append(logits)
				tensor_output = tf.stack(values=outputs, axis=0)
				forward = tf.transpose(tensor_output, [1, 0, 2])

	with variable_scope.variable_scope("Backpropergation"):
		loss = tf.contrib.seq2seq.sequence_loss(targets=decoder_outputs, logits=forward, weights=masking)
		updates = tf.train.AdamOptimizer().minimize(loss)

	return (forward,updates,loss, encoder_inputs, decoder_inputs, decoder_outputs, masking, encoder_lengths, decoder_lengths)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


	
network, updates, loss, enc_in, dec_in, dec_out, mask, enc_len, dec_len = seq2seq(batch_size=2, enc_input_dimension=1, enc_timesteps_max=5, dec_timesteps_max=5, hidden_units=128, hidden_layers=2, input_embedding_size=20, vocab_size=10)

feed = {}
feed["encoder_inputs"] = [[[1],[1],[1],[0],[0]],[[3],[3],[3],[0],[0]]]
feed["encoder_length"] = [3,3]
feed["decoder_inputs"] = [[1,1,1,0,0],[1,1,1,0,0]]
feed["decoder_length"] = [3,3]
feed["decoder_outputs"] = [[1,1,1,0,0],[2,2,2,0,0]]	
feed["mask"] = [[1.,1.,1.,0.,0.],[1.,1.,1.,0.,0.]]

init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
	session.run(init)
	writer = tf.summary.FileWriter(".", graph=tf.get_default_graph())
	
	for _ in range(40):
		training_output = session.run([updates, loss], feed_dict={enc_in:feed["encoder_inputs"], dec_in:feed["decoder_inputs"], dec_out: feed["decoder_outputs"], mask: feed["mask"], enc_len: feed["encoder_length"], dec_len: feed["decoder_length"]})

	result_raw = session.run(network, feed_dict={enc_in:feed["encoder_inputs"], dec_in:feed["decoder_inputs"], enc_len: feed["encoder_length"], dec_len: feed["decoder_length"]})
	print result_raw

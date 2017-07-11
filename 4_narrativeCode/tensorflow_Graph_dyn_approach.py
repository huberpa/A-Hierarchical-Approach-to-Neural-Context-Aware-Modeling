import tensorflow as tf 
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope

def seq2seq(encoder_inputs,
            decoder_inputs,
            enc_cell,
            dec_cell,
            input_embedding_size=300,
            vocab_size=30000,
            dtype=dtypes.float32):

	with variable_scope.variable_scope("seq2seq_Model"):

		# No Embeddings in Encoder, but Embeddings in Decoder
		_, initial_state = tf.nn.dynamic_rnn(enc_cell, encoder_inputs, dtype=dtype, scope = "rnn_encoder")

		with variable_scope.variable_scope("rnn_decoder"):
			outputs = []
			embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
			encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)
			lstm_output,state = tf.nn.dynamic_rnn(dec_cell, encoder_inputs_embedded, initial_state=initial_state, dtype=dtype)
			transp = tf.transpose(lstm_output, [1, 0, 2])
			lstm_output_unpacked = tf.unstack(transp)
			for item in lstm_output_unpacked:
				output = tf.contrib.layers.fully_connected(inputs=item, num_outputs=vocab_size, activation_fn=tf.nn.softmax)
				outputs.append(output)
		return outputs




batch_size = 2
max_input_length_enc = 5
max_input_length_dec = 5
input_size = 1
vocab_size = 5
input_embedding_size = 10
decoder_inputs = []

encoder_inputs = (tf.placeholder(dtypes.float32, shape=[batch_size, max_input_length_enc, input_size], name="enc_inputs"))
decoder_inputs = (tf.placeholder(dtypes.int64, shape=[batch_size, max_input_length_dec], name="dec_inputs"))
decoder_outputs = (tf.placeholder(dtypes.int64, shape=[batch_size, max_input_length_dec], name="dec_outputs"))

hidden_nb_enc = 5
hidden_nb_dec = 5

hidden_size_enc = 256
hidden_size_dec = 256

encoder_cell = single_cell_enc = tf.contrib.rnn.LSTMCell(hidden_size_enc)
if hidden_nb_enc > 1:
	encoder_cell = tf.contrib.rnn.MultiRNNCell([single_cell_enc] * hidden_nb_enc)

decoder_cell = single_cell_dec = tf.contrib.rnn.LSTMCell(hidden_size_dec)
if hidden_nb_dec > 1:
	decoder_cell = tf.contrib.rnn.MultiRNNCell([single_cell_dec] * hidden_nb_dec)	

outputs = seq2seq(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs, enc_cell=encoder_cell, dec_cell=decoder_cell, input_embedding_size=input_embedding_size, vocab_size=vocab_size)

#cost = tf.reduce.mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs, logits=outputs))
#updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

print len(outputs)
print outputs
# Initializing the variables
init = tf.global_variables_initializer()
# Launch the graph
with tf.Session() as session:
	session.run(init)
	writer = tf.summary.FileWriter(".", graph=tf.get_default_graph())


	feed = {}
	feed["encoder_inputs"] = [[[1],[1],[1],[0],[0]],[[2],[2],[2],[0],[0]]]
	feed["decoder_inputs"] = [[1,2,3,0,0],[3,2,1,0,0]]
	#feed["decoder_outputs"] = [[2,3,4,0,0],[4,3,2,0,0]]	
	#outputs = session.run(updates, feed_dict=feed)
	test = session.run(outputs, feed_dict={encoder_inputs:feed["encoder_inputs"], decoder_inputs:feed["decoder_inputs"]})

	print test


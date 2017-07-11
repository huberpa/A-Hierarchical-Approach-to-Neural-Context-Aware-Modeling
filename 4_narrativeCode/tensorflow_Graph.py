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
		_, initial_state = tf.contrib.rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype, scope = "rnn_encoder")

		with variable_scope.variable_scope("rnn_decoder"):
			state = initial_state
			outputs = []
			embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
			for i, inp in enumerate(decoder_inputs):
				if i > 0:
					variable_scope.get_variable_scope().reuse_variables()
				encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, inp)
				lstm_output, state = dec_cell(encoder_inputs_embedded, state)
				output = tf.contrib.layers.fully_connected(inputs=lstm_output, num_outputs=vocab_size, activation_fn=tf.nn.softmax)
				outputs.append(output)
		return outputs, state

batch_size = 5
input_length_enc = 3
input_length_dec = 4
input_size = 128
vocab_size = 30000
input_embedding_size = 300
encoder_inputs = []
decoder_inputs = []

for _ in range(input_length_enc):
	encoder_inputs.append(tf.placeholder(dtypes.float32, shape=[batch_size, input_size]))

for _ in range(input_length_dec):
	decoder_inputs.append(tf.placeholder(dtypes.int64, shape=[batch_size]))

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

outputs, state = seq2seq(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs, enc_cell=encoder_cell, dec_cell=decoder_cell, input_embedding_size=input_embedding_size, vocab_size=vocab_size)

print len(outputs)
print outputs[0].shape
print state[0]

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
	sess.run(init)
	writer = tf.summary.FileWriter(".", graph=tf.get_default_graph())



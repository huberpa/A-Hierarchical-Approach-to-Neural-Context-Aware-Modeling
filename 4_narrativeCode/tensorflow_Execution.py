import tensorflow as tf 
import numpy as np

feed = {}
feed["encoder_inputs"] = [[[1],[1],[1],[0],[0]],[[3],[3],[3],[0],[0]]]
feed["encoder_length"] = [3,3]
feed["decoder_inputs"] = [[1,1,1,0,0],[1,1,1,0,0]]
feed["decoder_length"] = [3,3]
feed["decoder_outputs"] = [[1,1,1,0,0],[2,2,2,0,0]]	
feed["mask"] = [[1.,1.,1.,0.,0.],[1.,1.,1.,0.,0.]]

with tf.Session() as session:
	tf.train.import_meta_graph(tf.train.latest_checkpoint("./models")+".meta").restore(session, tf.train.latest_checkpoint("./models"))
	variables = tf.get_collection('variables_to_store')

	result_raw = session.run(variables[0], feed_dict={variables[3]:feed["encoder_inputs"], variables[4]:feed["decoder_inputs"], variables[7]: feed["encoder_length"], variables[8]: feed["decoder_length"]})
	print result_raw

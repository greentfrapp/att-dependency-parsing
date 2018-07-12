
import tensorflow as tf
import numpy as np

# Options
# - Use word embeddings
# - Use siblibing information
# - Inside-out left-right
# - Character embeddings
# - DONE - Use hidden state of child instead of word index
# - Optimize log likelihood of correct prediction instead of crossentropy

class AttentionModel(object):

	def __init__(self, sess, vocab_size, pos_size=18, pos_2_size=42, max_len=10, hidden=512, name="DepParse", pos_enc=True, enc_layers=8, dec_layers=8, heads=8):
		super(AttentionModel, self).__init__()
		self.sess = sess
		self.max_len = max_len
		self.vocab_size = vocab_size
		self.pos_size = pos_size
		self.pos_2_size = pos_2_size
		self.hidden = hidden
		self.name = name
		self.pos_enc = pos_enc
		self.enc_layers = enc_layers
		self.dec_layers = dec_layers
		self.heads = heads
		with tf.variable_scope(self.name):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=2)

	def build_model(self):

		self.inputs = tf.placeholder(
			shape=(None, self.max_len + 2, self.vocab_size),
			dtype=tf.float32,
			name="encoder_input",
		)

		self.pos_inputs = tf.placeholder(
			shape=(None, self.max_len + 2, self.pos_size),
			dtype=tf.float32,
			name="encoder_pos_input",
		)

		self.pos_2_inputs = tf.placeholder(
			shape=(None, self.max_len + 2, self.pos_2_size),
			dtype=tf.float32,
			name="encoder_pos_2_input",
		)

		self.y_in_parents = tf.placeholder(
			shape=(None, self.max_len, self.max_len + 2),
			dtype=tf.float32,
			name="y_in_parents",
		)

		self.y_in_children = tf.placeholder(
			shape=(None, self.max_len, self.max_len + 2),
			dtype=tf.float32,
			name="y_in_children",
		)

		self.y_out_parents = tf.placeholder(
			shape=(None, self.max_len, self.max_len + 2),
			dtype=tf.float32,
			name="y_out_parents",
		)

		self.y_out_children = tf.placeholder(
			shape=(None, self.max_len, self.max_len + 2),
			dtype=tf.float32,
			name="y_out_children",
		)

		self.is_training = tf.placeholder(
			shape=(None),
			dtype=tf.bool,
			name="is_training",
		)

		posit_enc = tf.Variable(
			initial_value=tf.zeros((1, self.max_len + 2, self.hidden)),
			trainable=True,
			dtype=tf.float32,
			name="enc_positional_coding"
		)

		posit_dec = tf.Variable(
			initial_value=tf.zeros((1, self.max_len, self.hidden)),
			trainable=True,
			dtype=tf.float32,
			name="dec_positional_coding"
		)

		# Embed inputs to hidden dimension
		input_emb = tf.layers.dense(
			inputs=self.inputs,
			units=self.hidden,
			activation=None,
			name="input_embedding",
		)

		# Embed POS inputs to hidden dimension
		pos_input_emb = tf.layers.dense(
			inputs=self.pos_inputs,
			units=self.hidden,
			activation=None,
			name="pos_input_embedding"
		)

		# # Embed POS2 inputs to hidden dimension
		# pos_2_input_emb = tf.layers.dense(
		# 	inputs=self.pos_2_inputs,
		# 	units=self.hidden,
		# 	activation=None,
		# 	name="pos_2_input_embedding"
		# )

		encoding = input_emb + posit_enc + pos_input_emb #+ pos_2_input_emb

		# Embed inputs to hidden dimension
		y_in_parents_emb = tf.matmul(self.y_in_parents, encoding)
		y_in_children_emb = tf.matmul(self.y_in_children, encoding)

		dec_input_emb = tf.layers.dense(
			# inputs=tf.concat([self.y_in_parents, self.y_in_children], axis=2),
			inputs=tf.concat([y_in_parents_emb, y_in_children_emb], axis=2),
			units=self.hidden,
			activation=None,
			name="dec_input_embedding",
		)

		# dec_input_emb_parents = tf.layers.dense(
		# 	# inputs=tf.concat([self.y_in_parents, self.y_in_children], axis=2),
		# 	inputs=self.y_in_parents,
		# 	units=self.hidden,
		# 	activation=None,
		# 	name="dec_input_embedding_parents",
		# )

		# dec_input_emb_children = tf.layers.dense(
		# 	# inputs=tf.concat([self.y_in_parents, self.y_in_children], axis=2),
		# 	inputs=self.y_in_children,
		# 	units=self.hidden,
		# 	activation=None,
		# 	name="dec_input_embedding_children",
		# )

		# Add positional encodings

		decoding = dec_input_emb + posit_dec
		# decoding = dec_input_emb_children + dec_input_emb_parents + posit_dec

		for i in np.arange(self.enc_layers):
			encoding, _ = self.multihead_attention(
				query=encoding,
				key=encoding,
				value=encoding,
				h=self.heads,
			)
			# Encoder Dense
			dense = tf.layers.dense(
				inputs=encoding,
				units=self.hidden * 2,
				activation=tf.nn.relu,
				name="encoder_layer{}_dense1".format(i + 1)
			)
			encoding += tf.layers.dense(
				inputs=dense,
				units=self.hidden,
				activation=None,
				name="encoder_layer{}_dense2".format(i + 1)
			)
			encoding = tf.contrib.layers.layer_norm(encoding, begin_norm_axis=2)

		for i in np.arange(self.dec_layers):
			decoding, _ = self.multihead_attention(
				query=decoding,
				key=decoding,
				value=decoding,
				h=self.heads,
				mask=True,
			)
			decoding, _ = self.multihead_attention(
				query=decoding,
				key=encoding,
				value=encoding,
				h=self.heads,
			)
			dense = tf.layers.dense(
				inputs=decoding,
				units=self.hidden * 2,
				activation=tf.nn.relu,
				name="decoder_layer{}_dense1".format(i + 1)
			)
			decoding += tf.layers.dense(
				inputs=dense,
				units=self.hidden,
				activation=None,
				name="decoder_layer{}_dense2".format(i + 1)
			)
			decoding = tf.contrib.layers.layer_norm(decoding, begin_norm_axis=2)

		# decoding = tf.layers.dense(
		# 	inputs=decoding,
		# 	units=self.hidden * 2,
		# 	activation=tf.nn.relu,
		# 	name="decoding_1",
		# )

		# decode parents
		decoded_parents = tf.layers.dense(
			inputs=decoding,
			# units=self.hidden * 2,
			units=self.max_len + 2,
			activation=tf.nn.relu,
			name="decoding_parents_pre",
		)
		# decoded_parents = tf.layers.dropout(
		# 	inputs=decoded_parents,
		# 	rate=0.5,
		# 	training=self.is_training,
		# )
		# decoded_parents = tf.layers.dense(
		# 	inputs=decoded_parents,
		# 	units=self.max_len + 2,
		# 	activation=None,
		# 	name="decoded_parents",
		# )

		# decode children
		decoded_children = tf.layers.dense(
			inputs=decoding,
			# units=self.hidden * 2,
			units=self.max_len + 2,
			activation=tf.nn.relu,
			name="decoding_children_pre",
		)
		# decoded_children = tf.layers.dropout(
		# 	inputs=decoded_children,
		# 	rate=0.5,
		# 	training=self.is_training,
		# )
		# decoded_children = tf.layers.dense(
		# 	inputs=decoded_children,
		# 	units=self.max_len + 2,
		# 	activation=None,
		# 	name="decoded_children",
		# )

		# self.logits = decoding
		self.softmax_parents = tf.nn.softmax(decoded_parents)
		self.softmax_children = tf.nn.softmax(decoded_children)
		# self.predictions = tf.argmax(self.logits, axis=2)
		self.loss_parents = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_out_parents, logits=decoded_parents))
		self.loss_children = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_out_children, logits=decoded_children))
		self.loss = self.loss_parents + self.loss_children
		self.optimize = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

	def multihead_attention(self, query, key, value, h=4, mask=False):
		W_query = tf.Variable(
			initial_value=tf.random_normal((self.hidden, self.hidden), stddev=1e-2),
			trainable=True,
			dtype=tf.float32,
		)
		W_key = tf.Variable(
			initial_value=tf.random_normal((self.hidden, self.hidden), stddev=1e-2),
			trainable=True,
			dtype=tf.float32,
		)
		W_value = tf.Variable(
			initial_value=tf.random_normal((self.hidden, self.hidden), stddev=1e-2),
			trainable=True,
			dtype=tf.float32,
		)
		W_output = tf.Variable(
			initial_value=tf.random_normal((self.hidden, self.hidden), stddev=1e-2),
			trainable=True,
			dtype=tf.float32,
		)
		multi_query = tf.concat(tf.unstack(tf.reshape(tf.matmul(tf.reshape(query, [-1, self.hidden]), W_query), [-1, 1, tf.shape(query)[1], h, int(self.hidden/h)]), axis=3), axis= 1)
		multi_key = tf.concat(tf.unstack(tf.reshape(tf.matmul(tf.reshape(key, [-1, self.hidden]), W_key), [-1, 1, tf.shape(key)[1], h, int(self.hidden/h)]), axis=3), axis= 1)
		multi_value = tf.concat(tf.unstack(tf.reshape(tf.matmul(tf.reshape(value, [-1, self.hidden]), W_value), [-1, 1, tf.shape(value)[1], h, int(self.hidden/h)]), axis=3), axis= 1)
		dotp = tf.matmul(multi_query, multi_key, transpose_b=True) / (tf.cast(tf.shape(multi_query)[-1], tf.float32) ** 0.5)
		attention_weights = tf.nn.softmax(dotp)

		if mask:
			attention_weights = tf.matrix_band_part(attention_weights, -1, 0)
			attention_weights /= tf.reduce_sum(attention_weights, axis=3, keep_dims=True)

		weighted_sum = tf.matmul(attention_weights, multi_value)
		weighted_sum = tf.concat(tf.unstack(weighted_sum, axis=1), axis=-1)
		
		multihead = tf.reshape(tf.matmul(tf.reshape(weighted_sum, [-1, self.hidden]), W_output), [-1, tf.shape(query)[1], self.hidden])
		output = multihead + query
		# output = tf.contrib.layers.layer_norm(output, begin_norm_axis=2)
		return output, attention_weights
		

	def save(self, savepath, global_step=None, prefix="ckpt", verbose=False):
		if savepath[-1] != '/':
			savepath += '/'
		self.saver.save(self.sess, savepath + prefix, global_step=global_step)
		if verbose:
			print("Model saved to {}.".format(savepath + prefix + '-' + str(global_step)))

	def load(self, savepath, verbose=False):
		if savepath[-1] != '/':
			savepath += '/'
		ckpt = tf.train.latest_checkpoint(savepath)
		self.saver.restore(self.sess, ckpt)
		if verbose:
			print("Model loaded from {}.".format(ckpt))
		return ckpt
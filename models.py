
import tensorflow as tf
import numpy as np

# Idea 1 : Using attention matrix to denote parents
# Problem : If we use dot product attention then the self-attention relation is somewhat symmetrical
# Is that true?
# Preprocessing
# - Add ROOT to front of every sequence
# - Add NULL to back of every sequence
# - For labels, point padding labels to NULL
# - Define loss as L2 loss between attention map and dependency map

class AttentionModel(object):

	def __init__(self, sess, vocab_size, pos_size=18, max_len=20, hidden=512, name="DepParse", pos_enc=True, enc_layers=3, dec_layers=3, heads=8):
		super(AttentionModel, self).__init__()
		self.sess = sess
		self.max_len = max_len
		self.vocab_size = vocab_size
		self.pos_size = pos_size
		self.hidden = hidden
		self.name = name
		self.pos_enc = pos_enc
		self.enc_layers = enc_layers
		self.dec_layers = dec_layers
		self.heads = heads
		with tf.variable_scope(self.name):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=1)

	def build_model(self):

		self.inputs = tf.placeholder(
			shape=(None, self.max_len + 2, self.vocab_size),
			dtype=tf.float32,
			name="encoder_input",
		)

		self.pos_inputs = tf.placeholder(
			shape=(None, self.max_len + 2, self.pos_size),
			dtype=tf.float32,
			name="encoder_input",
		)

		self.labels = tf.placeholder(
			shape=(None, self.max_len + 2, self.max_len + 2),
			dtype=tf.float32,
			name="labels",
		)

		decoder_input = tf.Variable(
			initial_value=tf.zeros((1, self.max_len + 2, self.hidden)),
			trainable=True,
			dtype=tf.float32,
			name="decoder_input",
		)

		posit_enc = tf.Variable(
			initial_value=tf.zeros((1, self.max_len + 2, self.hidden)),
			trainable=True,
			dtype=tf.float32,
			name="positional_coding"
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

		# Add positional encodings
		encoding = input_emb + posit_enc + pos_input_emb

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

		decoding, self.attention_weights = self.multihead_attention(
			query=tf.tile(decoder_input, multiples=tf.concat(([tf.shape(self.inputs)[0]], [1], [1]), axis=0)),
			key=encoding,
			value=encoding,
			h=self.heads,
		)

		for i in np.arange(self.dec_layers):
			decoding, _ = self.multihead_attention(
				query=decoding,
				key=decoding,
				value=decoding,
				h=self.heads,
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
			decoding = tf.contrib.layers.layer_norm(encoding, begin_norm_axis=2)

		decoding = tf.layers.dense(
			inputs=decoding,
			units=self.max_len + 2,
			activation=None,
			name="decoding",
		)

		self.logits = decoding
		self.predictions = tf.argmax(self.logits, axis=2)
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
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
		output = tf.contrib.layers.layer_norm(output, begin_norm_axis=2)
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
import tensorflow as tf

a = tf.placeholder(
	shape=(2, 2, 3),
	dtype=tf.float32,
)

b = tf.placeholder(
	shape=(None),
	dtype=tf.float32,
)

c = tf.matmul(b, a)

sess = tf.Session()

print(sess.run(c, {a: [[[1, 2, 3],[4, 5, 6]], [[7, 8, 9],[10, 11, 12]]], b: [[[1, 0], [1, 0]], [[0, 1], [1, 0]]]}))
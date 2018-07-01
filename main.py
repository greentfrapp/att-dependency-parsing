
import numpy as np
import tensorflow as tf
from absl import flags
from absl import app

from utils import DependencyTask
from models import AttentionModel

FLAGS = flags.FLAGS

# Commands
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("test", False, "Test")
flags.DEFINE_integer("sample", 0, "Sample to test")

# Train parameters
flags.DEFINE_string("savepath", "models/", "Savepath to load/save models")

# Task parameters
flags.DEFINE_integer("max_len", 20, "Max sequence length")

def main(unused_args):
	if FLAGS.train:
		task = DependencyTask("data/en_ewt-ud-train.conllu")
		dev_task = DependencyTask("data/en_ewt-ud-dev.conllu", dictpath="data/en_ewt-ud-train.conllu.dict")
		sess = tf.Session()
		model = AttentionModel(sess, len(task.dict), max_len=FLAGS.max_len)

		sess.run(tf.global_variables_initializer())

		n_steps = 5000

		for step in np.arange(n_steps):
			x_tokens, x_pos, x_pos_2, y_in, y_out = task.next_batch(64, FLAGS.max_len)

			feed_dict = {
				model.inputs: x_tokens,
				model.pos_inputs: x_pos,
				model.pos_2_inputs: x_pos_2,
				model.dec_inputs: y_in,
				model.labels: y_out,
				model.is_training: True,
				# model.mask: mask,
			}
			
			loss, _ = sess.run([model.loss, model.optimize], feed_dict)
			if (step + 1) % 10 == 0:
				model.save(FLAGS.savepath, global_step=step+1)
				x_tokens, x_pos, x_pos_2, y_in, y_out = dev_task.next_batch(64, FLAGS.max_len)

				feed_dict = {
					model.inputs: x_tokens,
					model.pos_inputs: x_pos,
					model.pos_2_inputs: x_pos_2,
					model.dec_inputs: y_in,
					model.labels: y_out,
					model.is_training: False,
					# model.mask: mask,
				}
				val_loss = sess.run(model.loss, feed_dict)
				# correct = 0.
				# wrong = 0.
				# for i, sample_pred in enumerate(val_pred):
				# 	for j, pred in enumerate(sample_pred):
				# 		if np.argmax(y_heads[i, j, :]) != FLAGS.max_len + 1:
				# 			if np.argmax(y_heads[i, j, :]) == pred:
				# 				correct += 1
				# 			else:
				# 				wrong += 1
				# for i, sample_pred in enumerate(val_pred):
				# 	if sample_pred[0] == np.argmax(y_heads[i, :]):
				# 		correct += 1
				# 	else:
				# 		wrong += 1
				print("Step #{} - Loss : {:.2f} - Val Loss : {:.2f} - Val Acc : {}".format(step + 1, loss, val_loss, None))
				
				

	if FLAGS.test:
		task = DependencyTask("data/en_ewt-ud-train.conllu")
		dev_task = DependencyTask("data/en_ewt-ud-dev.conllu", dictpath="data/en_ewt-ud-train.conllu.dict")
		sess = tf.Session()
		model = AttentionModel(sess, len(task.dict), max_len=FLAGS.max_len)

		model.load(FLAGS.savepath)

		x_tokens, x_pos, x_pos_2, y_in, y_out = dev_task.next_batch(64, FLAGS.max_len)

		correct = 0.
		wrong = 0.

		for n_sample in np.arange(64):

			gold_parents = dict()
			parent = 0
			for token in y_out[n_sample]:
				idx = np.argmax(token)
				if idx == 0:
					break
				if idx not in gold_parents:
					gold_parents[idx] = parent
				parent = idx

			token_ids = list(np.argmax(x_tokens[n_sample], axis=1))
			if 0 in token_ids:
				size = token_ids.index(0) 
			else:
				size = FLAGS.max_len + 1

			mask = np.ones(FLAGS.max_len + 2)
			children = list(np.arange(1, size))
			dec_input = [0]
			n = 0
			parents = dict()
			parent = 0
			while True:
				temp = []
				for i in dec_input:
					temp.append(i)
				while len(temp) < FLAGS.max_len * 5:
					temp.append(-1)

				feed_dict = {
					model.inputs: [x_tokens[n_sample]],
					model.pos_inputs: [x_pos[n_sample]],
					model.pos_2_inputs: [x_pos_2[n_sample]],
					model.dec_inputs: [np.eye(FLAGS.max_len + 2)[np.array(temp)]],
					# model.labels: y_out,
					model.is_training: False,
					# model.mask: mask,
				}
				val_pred = sess.run(model.softmax_logits, feed_dict)
				new = np.argmax(val_pred[0][n] * mask)
				dec_input.append(new)
				if new == 0:
					break

				if new not in parents:
					parents[new] = parent
				parent = new

				if new in children:
					children.pop(children.index(new))
				mask = np.zeros(FLAGS.max_len + 2)
				for child in children:
					mask[child] = 1
				if parent != 0:
					mask[parents[parent]] = 1
				n += 1
				
				if n == FLAGS.max_len * 5:
					break
			
			for child, gold_parent in gold_parents.items():
				try:
					if gold_parent == parents[child]:
						correct += 1
					else:
						wrong += 1
				except:
					wrong += 1



			# sentence = dev_task.prettify(x_tokens[FLAGS.sample], clip_padding=True)
			# print(sentence)
			# print(dec_input[1:])
			# print(list(np.argmax(y_out[FLAGS.sample], axis=1)))
			# print(parents)
			# print(gold_parents)
			# quit()
		print(correct / (correct + wrong))


if __name__ == "__main__":
	app.run(main)


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
			x_tokens, x_pos, y_heads = task.next_batch(64, FLAGS.max_len)
			feed_dict = {
				model.inputs: x_tokens,
				model.pos_inputs: x_pos,
				model.labels: y_heads,
			}

			loss, _ = sess.run([model.loss, model.optimize], feed_dict)
			if (step + 1) % 10 == 0:
				x_tokens, x_pos, y_heads = dev_task.next_batch(64, FLAGS.max_len)

				feed_dict = {
					model.inputs: x_tokens,
					model.pos_inputs: x_pos,
					model.labels: y_heads,
				}
				val_loss, val_pred = sess.run([model.loss, model.predictions], feed_dict)
				correct = 0.
				wrong = 0.
				
				for i, sample_pred in enumerate(val_pred):
					for j, pred in enumerate(sample_pred):
						if np.argmax(y_heads[i, j, :]) != FLAGS.max_len + 1:
							if np.argmax(y_heads[i, j, :]) == pred:
								correct += 1
							else:
								wrong += 1
				print("Step #{} - Loss : {:.2f} - Val Loss : {:.2f} - Val Acc : {:.2f}".format(step + 1, loss, val_loss, correct / (correct + wrong)))
				
				model.save(FLAGS.savepath, global_step=step+1)

	if FLAGS.test:
		task = DependencyTask("data/en_ewt-ud-train.conllu")
		dev_task = DependencyTask("data/en_ewt-ud-dev.conllu", dictpath="data/en_ewt-ud-train.conllu.dict")
		sess = tf.Session()
		model = AttentionModel(sess, len(task.dict), max_len=FLAGS.max_len)

		model.load(FLAGS.savepath)

		x_tokens, x_pos, y_heads = dev_task.next_batch(64, FLAGS.max_len)
		feed_dict = {
			model.inputs: x_tokens,
			model.pos_inputs: x_pos,
			model.labels: y_heads,
		}
		val_loss, val_pred = sess.run([model.loss, model.predictions], feed_dict)

		sentence = dev_task.prettify(x_tokens[FLAGS.sample], clip_padding=True)
		prediction = val_pred[FLAGS.sample]
		while prediction[-1] == FLAGS.max_len + 1:
			prediction = prediction[:-1]
		labels = np.argmax(y_heads[FLAGS.sample], axis=1)
		while labels[-1] == FLAGS.max_len + 1:
			labels = labels[:-1]
		print(sentence)
		print(prediction)
		print(labels)


if __name__ == "__main__":
	app.run(main)

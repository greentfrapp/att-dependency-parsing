
import numpy as np
import tensorflow as tf
from absl import flags
from absl import app

from utils import DependencyTask
from models import AttentionModel

FLAGS = flags.FLAGS

# Commands
flags.DEFINE_bool("train", False, "Train")

def main(unused_args):
	if FLAGS.train:
		task = DependencyTask("data/en_ewt-ud-train.conllu")
		dev_task = DependencyTask("data/en_ewt-ud-dev.conllu", dictpath="data/en_ewt-ud-train.conllu.dict")
		sess = tf.Session()
		model = AttentionModel(sess, len(task.dict), max_len=50)

		sess.run(tf.global_variables_initializer())

		n_steps = 10000

		for step in np.arange(n_steps):
			x_tokens, x_pos, y_heads = task.next_batch(64, 50)
			feed_dict = {
				model.inputs: x_tokens,
				model.pos_inputs: x_pos,
				model.labels: y_heads,
			}
			loss, _ = sess.run([model.loss, model.optimize], feed_dict)

			if (step + 1) % 10 == 0:
				x_tokens, x_pos, y_heads = dev_task.next_batch(64, 50)
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
						if np.argmax(y_heads[i, j, :]) != 51:
							if np.argmax(y_heads[i, j, :]) == pred:
								correct += 1
							else:
								wrong += 1
				print("Step #{} - Loss : {:.2f} - Val Loss : {:.2f} - Val Acc : {:.2f}".format(step + 1, loss, val_loss, correct / (correct + wrong)))

if __name__ == "__main__":
	app.run(main)
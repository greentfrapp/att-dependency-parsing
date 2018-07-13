
from gensim.models import Word2Vec, KeyedVectors

from utils import DependencyTask

def main():

	task = DependencyTask("data/en_ewt-ud-train.conllu")

	model = Word2Vec(task.pos_tokens, size=100, window=5, min_count=1, workers=4)
	model.save("data/en_ewt-ud-train.conllu.pos.word2vec")
	model.wv.save("data/en_ewt-ud-train.pos.conllu.kv")

def test():
	wv = KeyedVectors.load("data/en_ewt-ud-train.conllu.kv", mmap='r')
	vector = wv["killed"]
	print(vector)

if __name__ == "__main__":
	main()
	# test()

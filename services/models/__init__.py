from gensim.models import KeyedVectors

thai2fit_model: KeyedVectors = KeyedVectors.load_word2vec_format("thai2fit/thai2vecNoSym.bin", binary=True)

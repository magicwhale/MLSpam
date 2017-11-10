import gensim
import os
import sys
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from matplotlib import pyplot
import re
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np

def build_model(directory_name):
	"""
	ham_directory = directory_name + "/ham"
	spam_directory = directory_name + "/spam"
	"""
	ham_directory1 = "enron1/ham"
	spam_directory1 = "enron1/spam"
	ham_directory2 = "enron2/ham"
	spam_directory2 = "enron2/spam"
	ham_directory3 = "enron3/ham"
	spam_directory3 = "enron3/spam"
	ham_directory4 = "enron4/ham"
	spam_directory4 = "enron4/spam"
	ham_directory5 = "enron5/ham"
	spam_directory5 = "enron5/spam"
	ham_directory6 = "enron6/ham"
	spam_directory6 = "enron6/spam"
	lines = get_lines(ham_directory1)
	lines += get_lines(spam_directory1)
	lines += get_lines(ham_directory2)
	lines += get_lines(spam_directory2)
	lines += get_lines(ham_directory3)
	lines += get_lines(spam_directory3)
	lines += get_lines(ham_directory4)
	lines += get_lines(spam_directory4)
	lines += get_lines(ham_directory5)
	lines += get_lines(spam_directory5)
	lines += get_lines(ham_directory6)
	lines += get_lines(spam_directory6)
	model = gensim.models.Word2Vec(lines)
	model.wv.save_word2vec_format('my_model_full_enron_unicode.bin')


"""
Output plot of embedded word vectors
"""
def visualize_model(model):
	word_vecs = model[model.wv.vocab]
	pca = PCA(n_components = 2)
	result = pca.fit_transform(word_vecs)
	# create a scatter plot of the projection
	pyplot.scatter(result[:, 0], result[:, 1])
	words = list(model.wv.vocab)
	for i, word in enumerate(words):
		try:
			pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
		except:
			print("this word couldnt be printed: " + word)
	pyplot.show()



"""
Input: a directory name containing our training data files
returns a list of every line in every training file
"""
def get_lines(directory_name):
	# A list of every line in the training set
	lines = []
	pattern = re.compile('[\W_]+')
	# Iterate over every file in our training set
	for filename in os.listdir(directory_name):
		# only look at text files
		if filename.endswith(".txt"):
			# get correct path to file from current working dir
			filename = directory_name + "/" + filename
			with open(filename, 'r') as email:
				for line in email:
					sentence = []
					line = line.strip()  # Get rid of newline character
					if line:  #Nonempty line
						words = line.split(" ")
						for word in words:
							word = word.decode('utf-8','ignore').encode("utf-8")
							pattern.sub('', word)
							if word != '':
								sentence.append(word)
					lines.append(sentence)

				# Add every line (split on \n character) to our list of lines
				#lines.append(email.read().splitlines())
	return lines



def embed_emails(directory_name, model, label):
	train_samples = []
	train_labels = []
	filenames = {}
	pattern = re.compile('[\W_]+')
	i = 1
	# Iterate over every file in our training set
	for filename in sorted_nicely(os.listdir(directory_name)):
		filenames[filename] = i
		i += 1
		embedding = np.zeros(100)
		flag = False
		# only look at text files
		if filename.endswith(".txt"):
			# get correct path to file from current working dir
			filename = directory_name + "/" + filename
			with open(filename, 'r') as email:
				num_words = 0
				for line in email:
					line = line.strip()  # Get rid of newline character
					if line:  #Nonempty line
						words = line.split(" ")
						for word in words:
							try:
								#strip word of punctuation
								pattern.sub('', word)
								if word == '':
									continue
								embedding += model[word]
								flag = True
							except:
								pass
							"""WHAT ABOUT WORDS THAT DONT APPEAR IN MODEL!!!!"""
							num_words += 1
				if flag:
					# one vector to represent entire email
					email_embedding = embedding/float(num_words)
				train_samples.append(email_embedding)
				train_labels.append([label])
	
	train_labels = np.array(train_labels)
	c, r = train_labels.shape
	print(train_labels.shape)
	train_labels = train_labels.reshape(c,)
	return np.stack(train_samples), train_labels


def classify_data(directory_name, model, classifier):
	test_samples, _ = embed_emails(directory_name, model, 0)
	return classifier.predict(test_samples)


def find_best_score_params(X_train, y_train, parameter_grid, clf_method):
	# Set our classifier to use the method specified checking over
	# the parameters specified with 5-fold cross validation
	clf = GridSearchCV(clf_method(), parameter_grid, cv = 5)

	# Try building a model from our training data using the 
	# classification method specified and with all the different params
	clf.fit(X_train, y_train)

	return clf.best_params_, clf.best_score_


"""
Copied from stack overflow
"""
def sorted_nicely(l): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

if __name__ == "__main__":
	#build_model("email_classification_data/train_data")
	
	#glove_input_file = "glove.6B.100d.txt"
	word2vec_output_file = 'glove.6B.100d.txt.word2vec'
	#gensim.scripts.glove2word2vec.glove2word2vec(glove_input_file, word2vec_output_file)
	#word2vec_output_file = "my_model_full_enron_unicode.bin"
	#word2vec_output_file = "my_model"


	model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
	#model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_output_file)


	print("model loaded\n")
	directory_name = "email_classification_data/train_data"
	ham_directory = directory_name + "/ham"
	ham_samples, ham_labels = embed_emails(ham_directory, model, 0)
	print(ham_samples.shape)
	print(ham_labels.shape)
	spam_directory = directory_name + "/spam"
	spam_samples, spam_labels = embed_emails(spam_directory, model, 1)
	print(spam_samples.shape)
	print(spam_labels.shape)
	train_samples = np.concatenate((ham_samples, spam_samples))
	train_labels = np.concatenate((ham_labels, spam_labels))
	

	# Split data into train and test sets using stratified sampling
	X_train, X_test, y_train, y_test = train_test_split(train_samples, train_labels, test_size = 0.2, stratify = train_labels)


	#rbf kernel 

	parameter_grid = [{'kernel' : ['rbf'], 'C' : [0.1, 1, 3], 'gamma' : [0.1, 0.5, 1, 3, 6, 10]}]

	params = find_best_score_params(X_train, y_train, parameter_grid, SVC)

	clf = SVC(C = params[0]['C'], kernel = 'rbf', gamma = params[0]['gamma'])

	# train our given classifier on our train set
	clf.fit(train_samples, train_labels)
	directory_name = "email_classification_data/test_data"
	predicted_labels = classify_data(directory_name, model, clf)

	out_file = open("predictions.txt", "w")
	out_file.write("email_id,labels\n")
	#writer = csv.writer(out_file)
	for index, element in enumerate(predicted_labels):
		out_file.write(str(index + 1) + "," + str(element))
		out_file.write("\n")












import sys
import os
import tempfile
import gensim
import nltk
from datetime import datetime, timedelta,date
from collections import defaultdict
import requests
import math
import json
import re
import pytz
from array import array
import nltk.data
from gensim import corpora, models, similarities
import os.path
import numpy as np
DATA_SERVER = 'xx.xx.xx.xx'
TEMP_FOLDER = tempfile.gettempdir()
from gensim.models import doc2vec, Doc2Vec, CoherenceModel
import operator
from gensim.utils import lemmatize
from nltk.corpus import stopwords

#print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))

def build_texts(documents):

    for document in documents:
        yield gensim.utils.simple_preprocess(document, deacc=True, min_len=3)

def process_texts(texts):

    texts = [[word for word in line if word not in stops] for line in texts]
    texts = [bigram[line] for line in texts]
    texts = [[word.split('/')[0] for word in lemmatize(' '.join(line), allowed_tags=re.compile('(NN)'), min_length=3)] for line in texts]
    return texts

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

class DocsLeeCorpus(object):
    def __init__(self, string_tags=False):
        self.string_tags = string_tags

    def _tag(self, i):
        return i if not self.string_tags else '_*%d' % i

    def __iter__(self):
        with open(datapath('lee_background.cor')) as f:
            for i, line in enumerate(f):
                yield doc2vec.TaggedDocument(utils.simple_preprocess(line), [self._tag(i)])

def get_reviews():
        current_date = datetime.now()
        d1 = datetime.now() - timedelta(days=current_date.day)
        d2 = datetime.now() - timedelta(days=270)
        d1 = d1.strftime('%d/%m/%Y')
        d2 = d2.strftime('%d/%m/%Y')
        #print ('d1: %s  d2: %s' %(d1, d2))
	if (str(sys.argv[1]) != 'ALL') :
		server_id = int(sys.argv[1])
		url = 'http://{0}/api/get_reviews/{1}?d1={2}&d2={3}'.format(DATA_SERVER,int(server_id), d2, d1)
		connect_timeout = 10.0
		response = requests.get(url=url, timeout=connect_timeout)
		#print response.status_code
		jresult = response.json()
		all_reviews_data = []
		documents = []

        ref_date = datetime.now() - timedelta(days=current_date.day)

        name_no_ext = '/tmp/'+str(sys.argv[1])+'/'+str(sys.argv[1])
        if ( (os.path.isfile(name_no_ext+'.docs')) & (os.path.isfile(name_no_ext+'.dict')) & (os.path.isfile(name_no_ext+'.tfidf')) & (os.path.isfile(name_no_ext+'.lda'))& (os.path.isfile(name_no_ext+'tfidf.index')) & (os.path.isfile(name_no_ext+'lda.index'))):
                if ( (datetime.fromtimestamp(os.path.getmtime(name_no_ext+'.docs')) > ref_date) &(datetime.fromtimestamp(os.path.getmtime(name_no_ext+'.dict')) > ref_date) & (datetime.fromtimestamp(os.path.getmtime(name_no_ext+'.tfidf')) > ref_date) & (datetime.fromtimestamp(os.path.getmtime(name_no_ext+'.lda'))>ref_date) & (datetime.fromtimestamp(os.path.getmtime(name_no_ext+'tfidf.index')) > ref_date) & (datetime.fromtimestamp(os.path.getmtime(name_no_ext+'lda.index'))>ref_date)):
			print 'came here'
			docs_file = open (os.path.join(TEMP_FOLDER, str(sys.argv[1]) +'/'+ str(sys.argv[1])+'.docs'),"r")
                        #print 'Dictionary, Corpus and Index files already exists for location id(server id) '+ str(server_id) +' till '+ d1+'. So reusing those files to find similarity'
                        dictionary = corpora.Dictionary.load(os.path.join(TEMP_FOLDER, str(sys.argv[1])+'/'+ str(sys.argv[1])+'.dict'))
                        tfidf = models.TfidfModel.load(os.path.join(TEMP_FOLDER, str(sys.argv[1])+'/'+ str(sys.argv[1])+'.tfidf'))
                        lda = models.LdaModel.load(os.path.join(TEMP_FOLDER, str(sys.argv[1])+'/'+ str(sys.argv[1])+'.lda'))
                        index_tfidf = similarities.Similarity.load(os.path.join(TEMP_FOLDER, str(sys.argv[1])+'/'+ str(sys.argv[1])+'tfidf.index'))
			documents = np.load(docs_file)
			sentences = [doc2vec.TaggedDocument(words, [i]) for i, words in enumerate(documents)]


			print (lda.show_topics(num_topics=10, num_words=10, log=False, formatted=True))
			model = Doc2Vec.load(os.path.join(TEMP_FOLDER, str(sys.argv[1])+'/'+ str(sys.argv[1])+'.doc2vec'))
			#model = Doc2Vec(sentences, size=400, window=8, min_count=5, workers=4, iter=20)
			"""for epoch in range(200):
                                if epoch % 20 == 0:
                                        print ('Now training epoch %s'%epoch)
                                model.train(sentences, total_examples=len(documents), epochs=1)
                                model.alpha -= 0.002  # decrease the learning rate
                                model.min_alpha = model.alpha  # fix the learning rate, no deca
                                model.dbow_words=1"""

			#model.save(os.path.join(TEMP_FOLDER, str(sys.argv[1])+'/'+ str(sys.argv[1])+'.doc2vec'))
			#model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
			#docvec = model.docvecs[1]
                        #print docvec
			a = np.array(documents)
			similar_doc = model.docvecs.most_similar(a.tolist().index(str(sys.argv[2])))

			#similar_doc = model.docvecs.most_similar(sentence)
			#print similar_do
			#new_docs =[sys.argv[2]]
			#sentence = [doc2vec.TaggedDocument(words, [i]) for i, words in enumerate(new_docs)]
			#tokens = str(re.sub('[.]', ' ', sys.argv[2])).split()
			#w = filter(lambda x: x in model.wv.vocab, tokens)
			#print model.most_similar(positive=w)
			#new_vector = model.infer_vector(tokens,alpha=0.025, min_alpha=0.025, steps=20)
			#similar_doc = model.docvecs.most_similar(positive = [new_vector],topn=10)
			print similar_doc
			#tokens = "great location".split()
			#new_vector = model.infer_vector(tokens)
			#docvec = model.docvecs.most_similar(new_vector)
			#print docvec
			#print sentences[25]
			print documents[26258]
			for sim in similar_doc[:10]:
                        	print ( documents[sim[0]] , sim[1])
                        #print all_reviews_data
                        #print documents
                        print('Number of sentences compared : %s (%s to %s)' % (len(documents) , d2, d1) )
                        sentence = re.sub('[.]', ' ', sys.argv[2])
                        doc = str(sentence)
                        #print doc
                        #tfidf = models.TfidfModel(corpus, normalize=True)
			# remove words that appear only once
                        vec_bow = dictionary.doc2bow(doc.lower().split())
                        vec_tfidf = tfidf[vec_bow] # convert the query to LSI spac
                        sims = index_tfidf[vec_tfidf]
                        sims = sorted(enumerate(sims), key=lambda item: -item[1])
                        sim_ids = []
                        for sim in sims[:10]:
                                if sim[0] not in sim_ids:
                                        sim_ids.append(sim[0])
                                        #print(sim[0], sim[1], documents[sim[0]])  # print sorted (document number, similarity score) 2-tuples
                                        print (documents[sim[0]], sim[1])
			index_lda = similarities.Similarity.load(os.path.join(TEMP_FOLDER, str(sys.argv[1])+'/'+ str(sys.argv[1])+'lda.index'))
                        vec_lda = lda[vec_bow] # convert the query to LSI space
                        sims2 = index_lda[vec_lda]
                        sims2 = sorted(enumerate(sims2), key=lambda item: -item[1])
                        sim_ids_2 = []
                        for sim in sims2[:10]:
                                #print(sim[0], sim[1], documents[sim[0]])  # print sorted (document number, similarity score) 2-tuples
                                if ((sim[0] not in sim_ids) & (sim[0] not in sim_ids_2)):
                                        sim_ids_2.append(sim[0])
                                        print (documents[sim[0]], sim[1])
                else:
                        #print 'Dictionary, Corpus and Index files needs to be updated for location id(server id) '+ str(server_id)
			docs_file = open (os.path.join(TEMP_FOLDER, str(sys.argv[1]) +'/'+ str(sys.argv[1])+'.docs'),"wb")
			if (str(sys.argv[1]) == 'ALL'):
				documents = []
				for x in range(1,175):
					url = 'http://{0}/api/get_reviews/{1}?d1={2}&d2={3}'.format(DATA_SERVER,x, d2, d1)
					connect_timeout = 10.0
					response = requests.get(url=url, timeout=connect_timeout)
					#print response.status_code
					jresult = response.json()
					all_reviews_data = []
					if 'all' in jresult:
						for d in jresult['all']:
							if 'identifier' in d and 'rating' in d and 'title' in d and 'comment' in d and 'review_date' in d and 'has_m' in d and 'm_response' in d:
								tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
								if d['comment']:
									doc_comments = tokenizer.tokenize(d['comment'])
									for sentence in doc_comments:
										documents.append( sentence  if re.match("^[a-zA-Z0-9_]*$", sentence[-1]) else sentence[:-1])
									all_reviews_data.append({'identifier': d['identifier'], 'rating': d['rating'], 'title': d['title'], 'comment': d['comment'], 'src': d['src'], 'review_date': datetime.strptime(d['review_date'], '%Y-%m-%d'), 'has_m': d['has_m'], 'm_response': d['m_response']})
				#docs_file = open (os.path.join(TEMP_FOLDER, str(sys.argv[1]) +'/'+ str(sys.argv[1])+'.docs'),"ab")
				np.save(docs_file,documents)
			else:
				if 'all' in jresult:
					for d in jresult['all']:
						if 'identifier' in d and 'rating' in d and 'title' in d and 'comment' in d and 'review_date' in d and 'has_m' in d and 'm_response' in d:
							tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
							if d['comment']:
                                                                doc_comments = tokenizer.tokenize(d['comment'])
                                                                for sentence in doc_comments:
                                                                        documents.append( sentence  if re.match("^[a-zA-Z0-9_]*$", sentence[-1]) else sentence[:-1])
                                                                all_reviews_data.append({'identifier': d['identifier'], 'rating': d['rating'], 'title': d['title'], 'comment': d['comment'], 'src': d['src'], 'review_date': datetime.strptime(d['review_date'], '%Y-%m-%d'), 'has_m': d['has_m'], 'm_response': d['m_response']})
				np.save(docs_file,documents)
                        #print documents
                        print('Number of sentences compared : %s (%s to %s)' % (len(documents) , d2, d1) )
                        #print('Number of sentences compared : %s' % len(documents))
                        stoplist = set('for a of the and to in we our'.split())
                        texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
                        # remove words that appear only once
                        frequency = defaultdict(int)
                        for text in texts:
                                for token in text:
                                        frequency[token] += 1
                        texts = [[token for token in text if frequency[token] > 1] for text in texts]
                        dictionary = corpora.Dictionary(texts)
                        dictionary.save(os.path.join(TEMP_FOLDER, str(sys.argv[1]) +'/'+ str(sys.argv[1])+'.dict'))  # store the dictionary, for future reference
                        #print(dictionary)
                        corpus = [dictionary.doc2bow(text) for text in texts]
                        tfidf = models.TfidfModel(corpus, normalize=True)
                        tfidf.save(os.path.join(TEMP_FOLDER, str(sys.argv[1]) +'/'+ str(sys.argv[1])+'.tfidf'))
                        #corpus_tfidf = tfidf[corpus]
                        #tfidf = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
                        sentence = re.sub('[.]', ' ', sys.argv[2])
                        doc = str(sentence)
                        #doc = str(sys.argv[2])
                        #doc = raw_input("Enter a sentence:")
                        #print doc
                        vec_bow = dictionary.doc2bow(doc.lower().split())
                        vec_tfidf = tfidf[vec_bow] # convert the query to LSI space
                        index = similarities.MatrixSimilarity(tfidf[corpus]) # transform corpus to LSI space and index it
                        index.save(os.path.join(TEMP_FOLDER, str(sys.argv[1])+'/'+ str(sys.argv[1])+'tfidf.index'))
                        sims = index[vec_tfidf]
                        sims = sorted(enumerate(sims), key=lambda item: -item[1])
                        sim_ids = []
                        for sim in sims[:10]:
                                if sim[0] not in sim_ids:
                                        sim_ids.append(sim[0])
                                        #print(sim[0], sim[1], documents[sim[0]])  # print sorted (document number, similarity score) 2-tuples
                                        print (documents[sim[0]], sim[1])
                        lda = models.LdaModel( tfidf[corpus], id2word=dictionary, num_topics=100)
                        lda.save(os.path.join(TEMP_FOLDER, str(sys.argv[1]) +'/'+ str(sys.argv[1])+'.lda'))
                        vec_lda = lda[vec_bow] # convert the query to LSI space
                        index = similarities.MatrixSimilarity(lda[corpus]) # transform corpus to LSI space and index it
                        index.save(os.path.join(TEMP_FOLDER, str(sys.argv[1])+'/'+ str(sys.argv[1])+'lda.index'))
                        sims2 = index_lda[vec_lda]
                        sims2 = sorted(enumerate(sims2), key=lambda item: -item[1])
                        sim_ids_2 = []
                        for sim in sims2[:10]:
                                #print(sim[0], sim[1], documents[sim[0]])  # print sorted (document number, similarity score) 2-tuples
                                if ((sim[0] not in sim_ids) & (sim[0] not in sim_ids_2)):
                                        sim_ids_2.append(sim[0])
                                        print documents[sim[0]]
        else:
                #print 'Dictionary, Corpus and Index files needs to be updated for location id(server id) '+ str(server_id)
                #os.mkdir(os.path.join(TEMP_FOLDER, str(sys.argv[1])+'/'))
		os.mkdir(os.path.join(TEMP_FOLDER, str(sys.argv[1])+'/'+'tfidf'+'/'))
		os.mkdir(os.path.join(TEMP_FOLDER, str(sys.argv[1])+'/'+'lda'+'/'))
		if (str(sys.argv[1]) == 'ALL'):
			documents = []
			name_no_ext = '/tmp/'+str(sys.argv[1])+'/'+str(sys.argv[1])
			print 'came here'
			if (os.path.isfile(name_no_ext+'.docs')):
				print 'reached here'
				docs_file = open (os.path.join(TEMP_FOLDER, str(sys.argv[1]) +'/'+ str(sys.argv[1])+'.docs'),"r")
				documents = np.load(docs_file)
			else :
				docs_file = open (os.path.join(TEMP_FOLDER, str(sys.argv[1]) +'/'+ str(sys.argv[1])+'.docs'),"wb")
				for x in range(1,175):
					print x
					url = 'http://{0}/api/get_reviews/{1}?d1={2}&d2={3}'.format(DATA_SERVER,x, d2, d1)
					connect_timeout = 10.0
					response = requests.get(url=url, timeout=connect_timeout)
					print response.status_code
					jresult = response.json()
					all_reviews_data = []

					if 'all' in jresult:
						for d in jresult['all']:
							if 'identifier' in d and 'rating' in d and 'title' in d and 'comment' in d and 'review_date' in d and 'has_m' in d and 'm_response' in d:
								tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
								#print d['comment']
								if d['comment']:
									doc_comments = tokenizer.tokenize(d['comment'])
									for sentence in doc_comments:
										documents.append( sentence  if re.match("^[a-zA-Z0-9_]*$", sentence[-1]) else sentence[:-1])
									all_reviews_data.append({'identifier': d['identifier'], 'rating': d['rating'], 'title': d['title'], 'comment': d['comment'], 'src': d['src'], 'review_date': datetime.strptime(d['review_date'], '%Y-%m-%d'), 'has_m': d['has_m'], 'm_response': d['m_response']})
					print 'came here2'
				#docs_file = open (os.path.join(TEMP_FOLDER, str(sys.argv[1]) +'/'+ str(sys.argv[1])+'.docs'),"ab")
				np.save(docs_file,documents)
		else:
			docs_file = open (os.path.join(TEMP_FOLDER, str(sys.argv[1]) +'/'+ str(sys.argv[1])+'.docs'),"wb")
			if 'all' in jresult:
				for d in jresult['all']:
					if 'identifier' in d and 'rating' in d and 'title' in d and 'comment' in d and 'review_date' in d and 'has_m' in d and 'm_response' in d:
						tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
						if d['comment']:
							doc_comments = tokenizer.tokenize(d['comment'])
							for sentence in doc_comments:
								documents.append( sentence  if re.match("^[a-zA-Z0-9_]*$", sentence[-1]) else sentence[:-1])
							all_reviews_data.append({'identifier': d['identifier'], 'rating': d['rating'], 'title': d['title'], 'comment': d['comment'], 'src': d['src'], 'review_date': datetime.strptime(d['review_date'], '%Y-%m-%d'), 'has_m': d['has_m'], 'm_response': d['m_response']})
			np.save(docs_file,documents)
		#documents.save(os.path.join(TEMP_FOLDER, str(sys.argv[1]) +'/'+ str(sys.argv[1])+'.docs'))
                #print all_reviews_data
                #print documents
                print('Number of sentences compared : %s (%s to %s)' % (len(documents) , d2, d1) )
                #print('Number of sentences compared : %s' % len(documents))
                stoplist = set('for a of the and to in we our'.split())
                texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
                # remove words that appear only once
                frequency = defaultdict(int)
                for text in texts:
                        for token in text:
                                frequency[token] += 1
                texts = [[token for token in text if frequency[token] > 1] for text in texts]
                dictionary = corpora.Dictionary(texts)
                dictionary.save(os.path.join(TEMP_FOLDER, str(sys.argv[1]) +'/'+ str(sys.argv[1])+'.dict'))  # store the dictionary, for future reference
                #print(dictionary)
                corpus = [dictionary.doc2bow(text) for text in texts]
                tfidf = models.TfidfModel(corpus, normalize=True)
                tfidf.save(os.path.join(TEMP_FOLDER, str(sys.argv[1]) +'/'+ str(sys.argv[1])+'.tfidf'))
                #corpus_tfidf = tfidf[corpus]
                #tfidf = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
                #doc = str(sys.argv[2])
                sentence = re.sub('[.]', ' ', sys.argv[2])
                doc = str(sentence)
                #doc = raw_input("Enter a sentence:")
                #print doc
                vec_bow = dictionary.doc2bow(doc.lower().split())
                vec_tfidf = tfidf[vec_bow] # convert the query to LSI space
                index = similarities.Similarity('/tmp/ALL/tfidf/',tfidf[corpus],len(dictionary)) # transform corpus to LSI space and index it
                index.save(os.path.join(TEMP_FOLDER, str(sys.argv[1])+'/'+ str(sys.argv[1])+'tfidf.index'))
                sims = index[vec_tfidf]
                sims = sorted(enumerate(sims), key=lambda item: -item[1])
                sim_ids = []
                for sim in sims[:10]:
                        if sim[0] not in sim_ids:
                                sim_ids.append(sim[0])
                                #print(sim[0], sim[1], documents[sim[0]])  # print sorted (document number, similarity score) 2-tuples
                                print documents[sim[0]]
                lda = models.ldamodel.LdaModel( tfidf[corpus], id2word=dictionary, num_topics=100,update_every=0, passes=20)
                lda.save(os.path.join(TEMP_FOLDER, str(sys.argv[1]) +'/'+ str(sys.argv[1])+'.lda'))
                vec_lda = lda[vec_bow] # convert the query to LSI space
                index = similarities.Similarity('/tmp/ALL/lda/',lda[corpus], len(dictionary)) # transform corpus to LSI space and index it
                index.save(os.path.join(TEMP_FOLDER, str(sys.argv[1])+'/'+ str(sys.argv[1])+'lda.index'))
                sims2 = index[vec_lda]
                sims2 = sorted(enumerate(sims2), key=lambda item: -item[1])
                sim_ids_2 = []
                for sim in sims2[:10]:
                        #print(sim[0], sim[1], documents[sim[0]])  # print sorted (document number, similarity score) 2-tuples
                        if ((sim[0] not in sim_ids) & (sim[0] not in sim_ids_2)):
                                sim_ids_2.append(sim[0])
                                print documents[sim[0]]
get_reviews()


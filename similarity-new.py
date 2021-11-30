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
from pymongo import MongoClient
c = MongoClient('localhost',27017)
db = c.primer
coll = db.reviewset
s_coll = db.sentiset
DATA_SERVER = 'xx.xx.xx.xx'
TEMP_FOLDER = tempfile.gettempdir()
#print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def get_reviews():
        current_date = datetime.now()
        d1 = datetime.now() - timedelta(days=current_date.day)
        d2 = datetime.now() - timedelta(days=300)
        d1 = d1.strftime('%d/%m/%Y')
        d2 = d2.strftime('%d/%m/%Y')
	review_ids =[]
        similar_response = []
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
			docs_file.close()
                        review_ids_file = open (os.path.join(TEMP_FOLDER, str(sys.argv[1]) +'/'+ str(sys.argv[1])+'.review_ids'),"r")
                        review_ids = np.load(review_ids_file)
                        review_ids_file.close()
                        #print all_reviews_data
                        #print documents
                        print('Number of sentences compared : %s (%s to %s)' % (len(documents) , d2, d1) )
                        sentence = re.sub('[.]', ' ', sys.argv[2])
                        doc = str(sentence)
                        print doc
                        #tfidf = models.TfidfModel(corpus, normalize=True)
                        vec_bow = dictionary.doc2bow(doc.lower().split())
                        vec_tfidf = tfidf[vec_bow] # convert the query to LSI spac
                        sims = index_tfidf[vec_tfidf]
                        sims = sorted(enumerate(sims), key=lambda item: -item[1])
                        sim_ids = []
			condition = True
			if (str(sys.argv[3]) == 'n'):
				condition = False
                        for sim in sims[:10]:
                                if sim[0] not in sim_ids:
                                        sim_ids.append(sim[0])
                                        #print(sim[0], sim[1], documents[sim[0]])  # print sorted (document number, similarity score) 2-tuples
					d = coll.find_one({'review_id': review_ids[sim[0]] })
					if d['rating']:
						#print (d['rating'],d['src'])
                                                if (d['src'] == 2):
							rating = int(str(d['rating'])[0])
                                                        #rating = int(d['rating'].lstrip().rstrip().split(' ')[0])
                                                if (d['src'] == 3):
                                                        rating = int(math.ceil(float(d['rating'].lstrip().rstrip().split(' ')[0])/2))
                                                if (d['src'] == 4):
                                                        rating =(d['rating'].lstrip().rstrip().split(' ')[0])
                                                if (d['src'] == 5):
                                                        rating = (['ONE','TWO','THREE','FOUR','FIVE'].index(d['rating']))+1
						if (d['src'] == 6):
                                                        rating = int(str(d['rating'])[0])
                                        else:
                                                rating =3
					if ((rating >3)== condition):
						print documents[sim[0]]
						print 'Title: ', str(d['title'])
                                                print 'Comment: ', str(d['comment'])
						print " "
						print 'Response: ', str(d['m_response'])
						print " "
						s = s_coll.find_one({'review_id': review_ids[sim[0]] })
						try:
							#if (any(d['sentence'] == documents[sim[0]] for d in s['aspects'])):
							for d in s['aspects']:
								if str(re.sub('[.]', '', d['sentence'])) == documents[sim[0]]:
									print d
							#print s['aspects']
						except:
							print 'no aspects'
			index_lda = similarities.Similarity.load(os.path.join(TEMP_FOLDER, str(sys.argv[1])+'/'+ str(sys.argv[1])+'lda.index'))
                        vec_lda = lda[vec_bow] # convert the query to LSI space
                        sims2 = index_lda[vec_lda]
                        sims2 = sorted(enumerate(sims2), key=lambda item: -item[1])
                        sim_ids_2 = []
                        for sim in sims2[:5]:
                                #print(sim[0], sim[1], documents[sim[0]])  # print sorted (document number, similarity score) 2-tuples
                                if ((sim[0] not in sim_ids) & (sim[0] not in sim_ids_2)):
                                        sim_ids_2.append(sim[0])
					d = coll.find_one({'review_id': review_ids[sim[0]] })
					if d['rating']:
						if (d['src'] == 2):
							rating = int(str(d['rating'])[0])
						if (d['src'] == 3):
							rating = int(math.ceil(float(d['rating'].lstrip().rstrip().split(' ')[0])/2))
						if (d['src'] == 4):
							rating =(d['rating'].lstrip().rstrip().split(' ')[0])
						if (d['src'] == 5):
							rating = (['ONE','TWO','THREE','FOUR','FIVE'].index(d['rating']))+1
						if (d['src'] == 6):
                                                        rating = int(str(d['rating'])[0])
					else:
						rating =3
					if ((rating >3)== condition):
						print documents[sim[0]]
                                                print 'Title: ', str(d['title'])
                                                print 'Comment: ', str(d['comment'])
						print " "
                                                print 'Response: ', str(d['m_response'])
                                                print " "
                                                s = s_coll.find_one({'review_id': review_ids[sim[0]] })
                                                try:
                                                        #if (any(d['sentence'] == documents[sim[0]] for d in s['aspects'])):
                                                        for d in s['aspects']:
                                                                if str(re.sub('[.]', '', d['sentence'])) == documents[sim[0]]:
                                                                        print d
                                                        #print s['aspects']
                                                except:
                                                        print 'no aspects'
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
					senti_url = 'http://{0}/api/get_aspects_sentences_for_hotel/{1}?d1={2}&d2={3}'.format(DATA_SERVER,x, d2, d1)
                                        senti_response = requests.get(url=senti_url, timeout=connect_timeout)
                                        print senti_response.status_code
                                        sresult = senti_response.json()
                                        all_senti_data = []

					if 'all' in jresult:
						for d in jresult['all']:
							if 'identifier' in d and 'rating' in d and 'title' in d and 'comment' in d and 'review_date' in d and 'has_m' in d and 'm_response' in d:
								tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
								if d['comment']:
									doc_comments = tokenizer.tokenize(d['comment'])
									for sentence in doc_comments:
										documents.append( sentence  if re.match("^[a-zA-Z0-9_]*$", sentence[-1]) else sentence[:-1])
										review_ids.append(d['review_id'])
									#all_reviews_data.append({'identifier': d['identifier'], 'rating': d['rating'], 'title': d['title'], 'comment': d['comment'], 'src': d['src'], 'review_date': datetime.strptime(d['review_date'], '%Y-%m-%d'), 'has_m': d['has_m'], 'm_response': d['m_response']})
									review_data = {'identifier': d['identifier'], 'rating': d['rating'], 'title': d['title'], 'comment': d['comment'], 'src': d['src'], 'review_date': datetime.strptime(d['review_date'], '%Y-%m-%d'), 'has_m': d['has_m'], 'm_response': d['m_response'],'review_id':d['review_id']}
									coll.insert_one(review_data)
					if sresult['status'] == 'success':
                                                if 'all' in sresult:
                                                        for d in sresult['all']:
                                                                if 'aspects' in d and 'review_id' in d:
                                                                        senti_data = { 'review_id': d['review_id'], 'aspects':d['aspects'] }
                                                                        s_coll.insert_one(senti_data)

				#docs_file = open (os.path.join(TEMP_FOLDER, str(sys.argv[1]) +'/'+ str(sys.argv[1])+'.docs'),"ab")
			else:
				url = 'http://{0}/api/get_reviews/{1}?d1={2}&d2={3}'.format(DATA_SERVER,sys.argv[1], d2, d1)
				connect_timeout = 10.0
				response = requests.get(url=url, timeout=connect_timeout)
				#print response.status_code
				jresult = response.json()
				all_reviews_data = []
				senti_url = 'http://{0}/api/get_aspects_sentences_for_hotel/{1}?d1={2}&d2={3}'.format(DATA_SERVER,sys.argv[1], d2, d1)
				senti_response = requests.get(url=senti_url, timeout=connect_timeout)
				print senti_response.status_code
				sresult = senti_response.json()
				all_senti_data = []
				if 'all' in jresult:
					for d in jresult['all']:
						if 'identifier' in d and 'rating' in d and 'title' in d and 'comment' in d and 'review_date' in d and 'has_m' in d and 'm_response' in d:
							tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
							if d['comment']:
                                                                doc_comments = tokenizer.tokenize(d['comment'])
                                                                for sentence in doc_comments:
                                                                        documents.append( sentence  if re.match("^[a-zA-Z0-9_]*$", sentence[-1]) else sentence[:-1])
									review_ids.append(d['review_id'])
								review_data = {'identifier': d['identifier'], 'rating': d['rating'], 'title': d['title'], 'comment': d['comment'], 'src': d['src'], 'review_date': datetime.strptime(d['review_date'], '%Y-%m-%d'), 'has_m': d['has_m'], 'm_response': d['m_response'],'review_id':d['review_id']}
								coll.insert_one(review_data)
				if sresult['status'] == 'success':
					if 'all' in sresult:
						for d in sresult['all']:
							if 'aspects' in d and 'review_id' in d:
								senti_data = { 'review_id': d['review_id'], 'aspects':d['aspects'] }
								s_coll.insert_one(senti_data)
			np.save(docs_file,documents)
			docs_file.close()
			review_ids_file = open (os.path.join(TEMP_FOLDER, str(sys.argv[1]) +'/'+ str(sys.argv[1])+'.review_ids'),"w")
			np.save(review_ids_file,review_ids)
			review_ids_file.close()
                        #print documents
                        print('Number of sentences compared : %s (%s to %s)' % (len(documents) , d2, d1) )
                        stoplist = set('for a of the and to in we our'.split())
                        texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
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
                                        print documents[sim[0]]
					d = coll.find_one({'review_id': review_ids[sim[0]] })
					print  d['m_response']
					s = s_coll.find_one({'review_id': review_ids[sim[0]] })
					print s['aspects']
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
					d = coll.find_one({'review_id': review_ids[sim[0]] })
					print  d['m_response']
					s = s_coll.find_one({'review_id': review_ids[sim[0]] })
					print s['aspects']
        else:
                #print 'Dictionary, Corpus and Index files needs to be updated for location id(server id) '+ str(server_id)
                #os.mkdir(os.path.join(TEMP_FOLDER, str(sys.argv[1])+'/'))
		os.mkdir(os.path.join(TEMP_FOLDER, str(sys.argv[1])+'/'+'tfidf'+'/'))
		os.mkdir(os.path.join(TEMP_FOLDER, str(sys.argv[1])+'/'+'lda'+'/'))
		if (str(sys.argv[1]) == 'ALL'):
			documents = []
			name_no_ext = '/tmp/'+str(sys.argv[1])+'/'+str(sys.argv[1])
			print 'came here'
			coll.delete_many({})
			s_coll.delete_many({})
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
					senti_url = 'http://{0}/api/get_aspects_sentences_for_hotel/{1}?d1={2}&d2={3}'.format(DATA_SERVER,x, d2, d1)
					connect_timeout = 120.0
                                        senti_response = requests.get(url=senti_url, timeout=connect_timeout)
					print senti_response.status_code
					sresult = senti_response.json()
					all_senti_data = []

					if 'all' in jresult:
						for d in jresult['all']:
							if 'identifier' in d and 'rating' in d and 'title' in d and 'comment' in d and 'review_date' in d and 'has_m' in d and 'm_response' in d:
								tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
								#print d['comment']
								if d['comment']:
									doc_comments = tokenizer.tokenize(d['comment'])
									for sentence in doc_comments:
										documents.append( sentence  if re.match("^[a-zA-Z0-9_]*$", sentence[-1]) else sentence[:-1])
										review_ids.append(d['review_id'])
									if (d['comment_negative']):
										review_data = {'identifier': d['identifier'], 'rating': d['rating'], 'title': d['title'], 'comment': d['comment'], 'n_comment': d['comment_negative'], 'src': d['src'], 'review_date': datetime.strptime(d['review_date'], '%Y-%m-%d'), 'has_m': d['has_m'], 'm_response': d['m_response'],'review_id':d['review_id']}
									else:
										review_data = {'identifier': d['identifier'], 'rating': d['rating'], 'title': d['title'], 'comment': d['comment'], 'src': d['src'], 'review_date': datetime.strptime(d['review_date'], '%Y-%m-%d'), 'has_m': d['has_m'], 'm_response': d['m_response'],'review_id':d['review_id']}
									coll.insert_one(review_data)
					try:
						if sresult['status'] == 'success':
							if 'all' in sresult:
								for d in sresult['all']:
									if 'aspects' in d and 'review_id' in d:
										senti_data = { 'review_id': d['review_id'], 'aspects':d['aspects'] }
										s_coll.insert_one(senti_data)
					except:
						print 'could not get sentiments'
					print 'came here2'
				#docs_file = open (os.path.join(TEMP_FOLDER, str(sys.argv[1]) +'/'+ str(sys.argv[1])+'.docs'),"ab")
		else:
			docs_file = open (os.path.join(TEMP_FOLDER, str(sys.argv[1]) +'/'+ str(sys.argv[1])+'.docs'),"wb")
			url = 'http://{0}/api/get_reviews/{1}?d1={2}&d2={3}'.format(DATA_SERVER,sys.argv[1], d2, d1)
			connect_timeout = 10.0
			response = requests.get(url=url, timeout=connect_timeout)
			print response.status_code
			jresult = response.json()
			all_reviews_data = []
			senti_url = 'http://{0}/api/get_aspects_sentences_for_hotel/{1}?d1={2}&d2={3}'.format(DATA_SERVER,sys.argv[1], d2, d1)
			connect_timeout = 30.0
			senti_response = requests.get(url=senti_url, timeout=connect_timeout)
			print senti_response.status_code
			sresult = senti_response.json()
			all_senti_data = []
			if 'all' in jresult:
				for d in jresult['all']:
					if 'identifier' in d and 'rating' in d and 'title' in d and 'comment' in d and 'review_date' in d and 'has_m' in d and 'm_response' in d:
						tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
						if d['comment']:
							doc_comments = tokenizer.tokenize(d['comment'])
							for sentence in doc_comments:
								documents.append( sentence  if re.match("^[a-zA-Z0-9_]*$", sentence[-1]) else sentence[:-1])
								review_ids.append(d['review_id'])
							if (d['comment_negative']):
								review_data = {'identifier': d['identifier'], 'rating': d['rating'], 'title': d['title'], 'comment': d['comment'], 'n_comment': d['comment_negative'], 'src': d['src'], 'review_date': datetime.strptime(d['review_date'], '%Y-%m-%d'), 'has_m': d['has_m'], 'm_response': d['m_response'],'review_id':d['review_id']}
							else:
								review_data = {'identifier': d['identifier'], 'rating': d['rating'], 'title': d['title'], 'comment': d['comment'], 'src': d['src'], 'review_date': datetime.strptime(d['review_date'], '%Y-%m-%d'), 'has_m': d['has_m'], 'm_response': d['m_response'],'review_id':d['review_id']}
							coll.insert_one(review_data)
			if sresult['status'] == 'success':
				if 'all' in sresult:
					for d in sresult['all']:
						if 'aspects' in d and 'review_id' in d:
							senti_data = { 'review_id': d['review_id'], 'aspects':d['aspects'] }
							s_coll.insert_one(senti_data)
		np.save(docs_file,documents)
		docs_file.close()
		review_ids_file = open (os.path.join(TEMP_FOLDER, str(sys.argv[1]) +'/'+ str(sys.argv[1])+'.review_ids'),"w")
		np.save(review_ids_file,review_ids)
		review_ids_file.close()
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
				d = coll.find_one({'review_id': review_ids[sim[0]] })
				print  d['m_response']
				s = s_coll.find_one({'review_id': review_ids[sim[0]] })
				print s['aspects']
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
				d = coll.find_one({'review_id': review_ids[sim[0]] })
                                print  d['m_response']
				s = s_coll.find_one({'review_id': review_ids[sim[0]] })
                                print s['aspects']
get_reviews()


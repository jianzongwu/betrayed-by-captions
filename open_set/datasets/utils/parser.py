from .lvis_v1_categories import LVIS_CATEGORIES

import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# from tqdm import tqdm
# import pickle

def normalize_class_names(thing_classes):
    new_thing_classes = []
    for name in thing_classes:
        new_name = name.replace('_',' ')
        new_name = new_name.replace('/',' ')
        new_name = new_name.replace('(',' ')
        new_name = new_name.replace(')',' ')
        new_name = new_name.lower()

        new_thing_classes.append(new_name)

    return new_thing_classes

class ImageNet21KParser():
    def __init__(self, add_adj=False):
        self.nlp = spacy.load("en_core_web_sm")
        self.look_up = {}
        with open('datasets/class_names/imagenet-21k.txt') as f:
            class_names = f.read()
        class_names = class_names.split()
        self.class_names = ['']*len(class_names)
        self.add_adj = add_adj
        for i, synonym in enumerate(class_names):
            synonym = synonym.lower()
            synonym = synonym.replace('_', ' ')

            self.class_names[i] = synonym

            doc = self.nlp(synonym)

            lemma_s = []
            for token in doc:
                word = token.lemma_
                if word.startswith('('):    #<< skip word in ()
                    break
                lemma_s.append(word)
            lemma_s = ' '.join(lemma_s)
            lemma_s = lemma_s.replace(' - ','-')

            self.look_up[lemma_s] = i

    def parse(self, sentence):
        sentence = sentence.lower()

        doc = self.nlp(sentence)
        lemma_sentence = []
        for token in doc:
            lemma_sentence.append(token.lemma_)
        lemma_sentence = ' '.join(lemma_sentence)

        nns = []
        category_ids = []

        for s in self.look_up:
            if ' {} '.format(s) in lemma_sentence or lemma_sentence.startswith(s+' ') or lemma_sentence.endswith(' '+s) or lemma_sentence == s:
                nns.append(s)
                category_ids.append(self.look_up[s])
        
        if self.add_adj:
            words = nltk.word_tokenize(sentence)
            words = [word for word in words if word not in set(stopwords.words('english'))]
            tagged = nltk.pos_tag(words)
            for (word, tag) in tagged:
                if tag in ['JJ', 'JJR', 'JJS']: # If the word is a proper noun
                    if word not in nns:
                        nns.append(word)
        
        return nns, category_ids

class LVISParser():
    def __init__(self, add_adj=False):
        self.nlp = spacy.load("en_core_web_sm")
        self.look_up = {}
        self.class_names = ['']*len(LVIS_CATEGORIES)
        self.add_adj = add_adj
        for item in LVIS_CATEGORIES:
            synonyms = item['synonyms']

            synonyms = [s.lower() for s in synonyms]
            synonyms = [s.replace('_',' ') for s in synonyms]

            id = item['id']-1       # convert to 0 base

            self.class_names[id] = item['name']

            for s in synonyms:
                doc = self.nlp(s)

                lemma_s = []
                for token in doc:
                    word = token.lemma_
                    if word.startswith('('):    #<< skip word in ()
                        break
                    lemma_s.append(word)
                lemma_s = ' '.join(lemma_s)
                lemma_s = lemma_s.replace(' - ','-')

                # if lemma_s in self.look_up:
                #     print('Duplication {}'.format(lemma_s))

                self.look_up[lemma_s] = id

        # print('lvis parser vocab size {}'.format(len(self.look_up)))

    def parse(self, sentence):
        sentence = sentence.lower()

        doc = self.nlp(sentence)
        lemma_sentence = []
        for token in doc:
            lemma_sentence.append(token.lemma_)
        lemma_sentence = ' '.join(lemma_sentence)

        nns = []
        category_ids = []

        for s in self.look_up:
            if ' {} '.format(s) in lemma_sentence or lemma_sentence.startswith(s+' ') or lemma_sentence.endswith(' '+s) or lemma_sentence == s:
                nns.append(s)
                category_ids.append(self.look_up[s])
        
        if self.add_adj:
            words = nltk.word_tokenize(sentence)
            words = [word for word in words if word not in set(stopwords.words('english'))]
            tagged = nltk.pos_tag(words)
            for (word, tag) in tagged:
                if tag in ['JJ', 'JJR', 'JJS']: # If the word is a proper noun
                    if word not in nns:
                        nns.append(word)
        
        return nns, category_ids

class NLTKParser():
    def __init__(self, allowed_tags=['NN', 'NNS']):
        self.allowed_tags = allowed_tags

    def parse(self, sentence):
        words = nltk.word_tokenize(sentence)
        words = [word for word in words if word not in set(stopwords.words('english'))]
        tagged = nltk.pos_tag(words)
        nns = []
        for (word, tag) in tagged:
            if tag in self.allowed_tags: # If the word is a proper noun
                nns.append(word)
        return nns, None
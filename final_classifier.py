#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 00:18:48 2020

@author: prannoynoel
"""

import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle

# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

## Want BERT instead of distilBERT? Uncomment the following line:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

class Classifier:
    """The Classifier"""
        
        
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""        
        # Loading the data into a datframe
        df = pd.read_csv(trainfile, delimiter='\t', header=None, 
                         names = ["polarity", "aspect_category", "aspect_term", "at_location", "sentence"])
        
        # tokenize the sentences -- break them up into word and subwords
        tokenized = df['sentence'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
        # Finding the largest sentence
        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)
                
        # pad all lists to the same size, so we can represent the input as one 2-d array
        padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
        
        # Ignores the padding if any error occurs while processing the input
        attention_mask = np.where(padded != 0, 1, 0)
        
        # Converting the pad and mask into tensor
        input_ids = torch.tensor(padded)  
        attention_mask = torch.tensor(attention_mask)
        
        # The BERT model takes input 
        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)
            
        # Removes [CLS] token 
        features = last_hidden_states[0][:,0,:].numpy()
        labels = df['polarity'] #labels of 
        
        # Training the model using a classifier 
        clf = LogisticRegression(max_iter=2000)
        clf.fit(features, labels)
        
        # save the weights of the model
        filename = 'finalized_model.sav'
        pickle.dump(clf, open(filename, 'wb'))
        print('Finished training')
        
       
    def predict(self, datafile):
       """Predicts class labels for the input instances in file 'datafile'
       Returns the list of predicted labels
       """
       # Loading the data into a datframe
       df_test = pd.read_csv(datafile, delimiter='\t', header=None, 
                             names = ["polarity", "aspect_category", "aspect_term", "at_location", "sentence"])
       
       # tokenize the sentences -- break them up into word and subwords
       tokenized_test = df_test['sentence'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
       # Finding the largest sentence
       max_len = 0
       for i in tokenized_test.values:
           if len(i) > max_len:
               max_len = len(i)
        
       # pad all lists to the same size, so we can represent the input as one 2-d array 
       padded_test = np.array([i + [0]*(max_len-len(i)) for i in tokenized_test.values])
       
       # Ignores the padding if any error occurs while processing the input
       attention_mask_test = np.where(padded_test != 0, 1, 0)
        
       # Converting the pad and mask into tensor
       input_ids_test = torch.tensor(padded_test)  
       attention_mask_test = torch.tensor(attention_mask_test)
       
       # The BERT model takes input 
       with torch.no_grad():
           last_hidden_states_test = model(input_ids_test, attention_mask = attention_mask_test)
 
       test_features = last_hidden_states_test[0][:,0,:].numpy()
       labels_test = df_test['polarity']
       
       filename = 'finalized_model.sav'
       loaded_model = pickle.load(open(filename, 'rb')) # loading the model
       result = loaded_model.predict(test_features) # predicting the output
       return result
   
    
'''
dev_file = '/Users/prannoynoel/Documents/DS/NLP/exercise2/data/devdata.csv'
train_file =  '/Users/prannoynoel/Documents/DS/NLP/exercise2/data/traindata.csv'
k = Classifier()
k1 = k.train(train_file)
k2 = k.predict(dev_file)
print(k2)        
'''
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
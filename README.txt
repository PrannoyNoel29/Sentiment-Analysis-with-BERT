
NLP Exercise 2: Aspect-Based Sentiment Analysis
Veerendarnath NALADALA, Priyanka PIPPIRI, Shamir KAZI, Prannoy Noel BHUMANA


Our goal is to create a model that takes a sentence (just like the ones in our dataset) and produces either positive indicating the sentence carries a positive sentiment or a negative indicating the sentence carries a negative sentiment. 

Pre-trained model:
DistilBERT processes the sentence and passes along some information it extracted from it on to the next model. DistilBERT is a smaller version of BERT developed and open sourced by the team at HuggingFace. It's a lighter and faster version of BERT that roughly matches its performance.

Classifier Model:
The next model, a basic Logistic Regression model from scikit learn will take in the result of DistilBERT's processing, and classify the sentence as either positive, negative, neutral.

Methodology:
1. Load the Pre-trained BERT model.
2. Preparing the Dataset.
a) Tokenization: Tokenize the sentences -- break them up into word and sub words in the format BERT is comfortable with.
b) Padding: After tokenization, tokenized is a list of sentences -- each sentence is represented as a list of tokens. BERT process our examples all at once (as one batch). It's just faster that way. For that reason, we need to pad all lists to the same size, so we can represent the input as one 2-d array, rather than a list of lists (of different lengths).
c) Masking: If we directly send padded to BERT, that would slightly confuse it. We need to create another variable to tell it to ignore (mask) the padding we've added when it's processing its input. 
3. Training the model using Deep Learning. The Pre-trained Bert model function runs our sentences . The results of the processing will be returned.
4. We sliced only the part of the output that we needed. That is the output corresponding the first token of each sentence. The way BERT does sentence classification, is that it adds a token called [CLS] (for classification) at the beginning of every sentence. The output corresponding to that token can be thought of as an embedding for the entire sentence. 
5. We'll save it in the features variable. It serves as the features to our logistic regression model and labels are the polarity in the data. The training is done with classifier and the weights of the model are saved. 
6. The same series of above steps are used for the devdata. Then the model is loaded with weights saved and predicts the results. The accuracy scores then obtained for us is 81.6%. We have obtained 271 positive, 98 negative, 7 neutral.

Conclusion:
BERT yields high quality results at some expense(cleaning of text).  The lighter version of BERT(DistilBERT) produced better results than BERT base-uncased. If the DistilBERT can be trained to improve its score, namely fine-tuning which updates BERT's weights to make it achieve a better performance and leads to better results.

Requirements:
This needs a pre-trained BERT model. The appropriate code for downloading it is included in the scripts. We hope you don¡¯t consider this time for the performance of the model. The python libraries required are: 
1. Numpy : 1.18.1
2. Pandas: 1.0.3
3. Torch : 1.4.0
4. Transformers : 2.8.0
5. Sklearn : '0.22.2.post1'
6. Tensorflow: 2.1.0

The scripts require all of the above libraries including with the versions to yield the best results without any errors.

Environment:

* OS = Debian GNU/Linux 9 (stretch)
* Kernel = Linux 4.9.0-12-amd64
* NAME = Debian GNU/Linux
* VERSION = 9 (stretch)
* GPU = Red Hat, Inc Virtio SCSI [1af4:0008]

The execution of the scripts is performed on the virtual machine with the above environment details. The testing process can be seen below along with the results. 

References:
1. A Visual Guide to Using BERT href:'http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/'
2. Exploiting BERT for End-to-End Aspect-based Sentiment Analysis. href:'https://github.com/lixin4ever/BERT-E2E-ABSA'


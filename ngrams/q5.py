# your code goes here

import pdb
import nltk
import collections
import random

def generate_sentence( probs ):
    cur_fw = '<s>'
    end = '</s>'
    cur_bigram = ('','')
    sent = ''
    while cur_bigram[1] != end:
        brange = 0
        ranges = {}
        for bigram in probs:
            if bigram[0] == cur_fw:
                ranges[bigram] = [brange, brange + probs[bigram]]
                brange += probs[bigram]
        select = random.uniform(0, 1)
        for bigram in ranges:
            if ranges[bigram][0] <= select <= ranges[bigram][1]:
                if bigram[1] != '</s>':
                    sent += bigram[1] + ' '
                cur_bigram = bigram
                cur_fw = bigram[1]
                break

    return sent

def getCommonNgrams():
    with open('t8.shakespeare.txt', 'r') as f:
        text = f.read()

    tokens = nltk.word_tokenize(text)
    tokens = [ token.lower() for token in tokens if len(token) > 1 ] # unigrams
    bi_tokens = nltk.bigrams(tokens)

    unigram_frequencies = collections.Counter(tokens)
    common_unigrams = unigram_frequencies.most_common(15)

    bigram_frequencies = collections.Counter(bi_tokens)
    common_bigrams = bigram_frequencies.most_common(15)

    print( common_unigrams )
    print( common_bigrams )

def generateSentences():
    with open('t8.shakespeare.txt', 'r') as f:
        text = f.read()

    sent_tokenize_list = nltk.tokenize.sent_tokenize(text)
    sent_tokenize_list = map( lambda sent: ['<s>'] + nltk.word_tokenize(sent) + ['</s>'], sent_tokenize_list )

    tokens = [ word for sublist in sent_tokenize_list for word in sublist ] #flattened
    #tokens = [ token.lower() for token in tokens if len(token) > 1 ] # unigrams
    bi_tokens = nltk.bigrams(tokens)
    bi_tokens = list(bi_tokens)

    unigram_frequencies = collections.Counter(tokens)
    unigram_probs = {}

    for unigram in unigram_frequencies:
        unigram_probs[unigram] = float(unigram_frequencies[unigram]) / len(tokens)

    bigram_frequencies = collections.Counter(bi_tokens)
    bigram_probs = {}

    for bigram in bigram_frequencies:
        bigram_probs[bigram] = float(bigram_frequencies[bigram]) / unigram_frequencies[bigram[0]]


    for i in range(5):
        sentence = generate_sentence(bigram_probs).strip()
        print sentence

if __name__ == '__main__':
    getCommonNgrams()
    generateSentences()

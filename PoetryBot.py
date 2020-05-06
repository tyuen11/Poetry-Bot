import random

import nltk
from hyphenate import hyphenate_word
import prosodic as p
from nltk.corpus import brown, gutenberg
import numpy as np

tokens = nltk.word_tokenize("Will Will give May candy?")
print(nltk.pos_tag(tokens))
tokens = nltk.word_tokenize("No answer me. Stand and unfold your self")
print(nltk.pos_tag(tokens))
# print(nltk.corpus.abc.words())
#
# print(hyphenate_word('before'))

shakespeare_works = ['shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt']


# text = p.Text("thee to today")
# text.parse()
# print(text.bestParses()[0])

##########################################################################################
# Poetry bot generating sonnets
# Sonnet: 14 lines of 4 parts
# 3 quatrians and 1 couplet
# Each quatrain has rhyme of ABAB that is independent of each other written in iambic pentameter
# End with a couplet that has the last word of both lines rhyme
##########################################################################################


def parse_text(text_file):
    """
    Parse text file to generate markov chain
    :param text_file:
    :return:
    """

    bi_gram = dict()
    text = gutenberg.words(text_file)[1:]
    print(len(text))
    # Get all bi-grams in text_file
    for x in range(len(text) - 1):
        first_word = text[x].lower()
        second_word = text[x + 1].lower()

        if first_word in bi_gram.keys():  # if the word is a key in bi-gram
            if second_word not in bi_gram[first_word].keys():  # if the bi-gram we are looking at is not in bi-gram
                bi_gram[first_word][second_word] = 1  # add the bi-gram to bi-gram
            else:
                bi_gram[first_word][second_word] += 1  # update the freq of the bi-gram we are looking at
        elif first_word not in bi_gram.keys():
            bi_gram[first_word] = {second_word: 1}

    return bi_gram


def generate_sentence(mc, length, start_node=None):
    if length is 0:
        return []
    start = random.choice(list(mc)) if start_node is None else start_node
    weights = np.array(
        list(mc[start].values()),
        dtype=np.float64)
    weights /= weights.sum()

    choices = list(mc[start].keys())
    print(start, choices)
    chosen_word = np.random.choice(choices, None, p=weights)

    return [chosen_word] + generate_sentence(mc, length=length-1, start_node=chosen_word)


markov_chain = parse_text("austen-emma.txt")

print(' '.join(generate_sentence(markov_chain, length=12)))

import random
import re
import nltk
from nltk.corpus import brown, treebank, cmudict, gutenberg
import numpy as np
import syllables as syllables_p
import pronouncing


##########################################################################################

# Poetry bot generating sonnets
# Sonnet: 14 lines of 4 parts
# 3 quatrians and 1 couplet
# Each quatrain has rhyme of ABAB that is independent of each other
# End with a couplet that has the last word of both lines rhyme

##########################################################################################


def generate_markov_chain(mc, n_gram=2):
    """
    Generates a markov chain and puts it in a dictionary
    :param mc:
    :param n_gram:
    :return:
    """
    text = brown.words(categories="lore")
    regex = re.compile("^([A-Z])\w+([a-zA-Z]+[-'][a-zA-Z]+)|([a-zA-Z]+\.)|([a-zA-Z])+$")
    text = [word for word in text if regex.fullmatch(word)]
    n_grams = nltk.ngrams(text, n_gram)
    ngram_counter = {}
    # Get the frequency of an n-gram in all generated n-grams from text
    for ng in n_grams:
        if ng in ngram_counter.keys():
            ngram_counter[ng] += 1
        else:
            ngram_counter[ng] = 1
    # Create the markov chain for each n-gram
    for ng in ngram_counter:
        current_subtree = mc
        for index in range(len(ng)):
            word = ng[index]
            if current_subtree.get(word):
                current_subtree = current_subtree[word]
            elif index is not len(ng) - 1:
                current_subtree[word] = {}
                current_subtree = current_subtree[word]
            else:
                current_subtree[word] = ngram_counter[ng]


def calulate_weights(mc, word):
    """
    Calculates the weights of a word's markov chain
    :param mc:
    :param word:
    :return:
    """
    weights = np.array(
        list(mc[word].values()),
        dtype=np.float64)
    weights /= weights.sum()
    return weights


def generate_sentence(mc, sent_struc, start_node=None, num_syllables=10, a_rhyme=None, b_rhyme=None):
    """
    Generates a sentence from the markov chain, mc
    :param mc:
    :param sent_struc:
    :param start_node:
    :param num_syllables:
    :param a_rhyme:
    :param b_rhyme:
    :return:
    """
    # Base case is when the number of syllables is reached
    if num_syllables is 0:
        return []

    # Get a random word to start the sentence
    start = random.choice(list(mc)) if start_node is None else start_node
    weights = calulate_weights(mc, start)
    # print(sent_struc) if start_node is None else ()
    redo = True
    chosen_words = []  # words that don't fulfill syllable requirement
    while redo:  # keep looping until we find a word that does not exceed the syllable limit and satisfies the other conditions
        # find a random word from the markov chain
        choices = list(mc[start].keys())
        chosen_word = np.random.choice(choices, None, p=weights)
        chosen_word_pos = nltk.pos_tag(nltk.word_tokenize(chosen_word))[0][1]

        prev_word_pos = nltk.pos_tag(nltk.word_tokenize(start))[0][1]

        # If the word we chose is not in the rejected words list and in mc key
        if chosen_word not in chosen_words and chosen_word in mc.keys():
            # Get remaining number of syllables we need
            chosen_word_syllable = syllables(chosen_word)
            new_num_syllables = num_syllables - chosen_word_syllable
            # if the chosen word makes the total number of syllables > 10 or has the same POS as the previous word,
            # then choose another word
            if new_num_syllables >= 0 and chosen_word_pos is not prev_word_pos:
                redo = False

            # Check if we are generating the second sentence of A or B
            if new_num_syllables is 0:
                if a_rhyme is not None and b_rhyme is not None:  # Second sentence of A
                    chosen_word = get_rhyme_word(mc, a_rhyme, None, chosen_word_syllable)
                if a_rhyme is None and b_rhyme is not None:  # Second sentence of B
                    chosen_word = get_rhyme_word(mc, None, b_rhyme, chosen_word_syllable)
                # print("NEW WORD IS " + chosen_word)
            chosen_words.append(chosen_word)
        # Case of only having one choice and it not being compatible for the sentence, get a new word to branch off of
        elif chosen_word not in mc.keys() or len(choices) is len(chosen_words):
            start = random.choice(list(mc))
            weights = calulate_weights(mc, start)
            chosen_words = []
    return [chosen_word] + generate_sentence(mc, sent_struc=sent_struc[1:], start_node=chosen_word,
                                             num_syllables=new_num_syllables, a_rhyme=a_rhyme, b_rhyme=b_rhyme)


def get_rhyme_word(mc, a_rhyme, b_rhyme, syllable):
    """
    Gets a word of a certain syllable that rhymes with the a_rhyme or b_rhyme
    Calls rhyme_all_words()

    :param mc:
    :param a_rhyme:
    :param b_rhyme:
    :param syllable:
    :return:
    """
    # Run rhymes() which will return rhymes of a a_rhyme or b_rhyme
    # filter out all rhymes that do not have syllable = syllable
    # Calculate the mc of the filtered rhymes and their probabilities
    # Randomly pick a rhyme based on its probability

    rhyme = a_rhyme if b_rhyme is None else b_rhyme
    # rhymes_prob = rhymes(rhyme, mc)
    # rhymes_prob = {r: rhymes_prob[r] for r in rhymes_prob if syllables_p.estimate(r) == syllable}  # dict of rhymes and their probabilities

    # *
    # chosen_word = random.choice(list(rhymes_prob.keys()))
    # return chosen_word

    # weights = np.array(
    #     list(rhymes_prob.values()),
    #     dtype=np.float64)
    # weights /= weights.sum()

    # choices = list(rhymes_prob.keys())
    # print('rhyme choices', choices)
    # chosen_word = "@" if not choices else np.random.choice(choices, None, p=weights)  # case where we can find no rhyme in the mc
    # Could fix this by randomly choosing a rhyme from rhymes_prob.keys() [code for this at *] and desired POS

    chosen_word = "@"
    rhymes_prob = rhymes_all_words(rhyme, mc)
    rhymes_prob = [r for r in rhymes_prob if syllables_p.estimate(r) == syllable]
    try:
        chosen_word = random.choice(rhymes_prob)
    except:
        pass
    return chosen_word


def rhymes_all_words(word, mc):
    """
    Look for all words that rhyme with word in our markov chain
    :param word:
    :param mc:
    :return: rhyme_list
    """
    if "-" in word:
        word = word.split("-")[-1]
    # Finding all words that rhyme with word, disregarding the word's mc
    words = mc.keys()
    regex = re.compile("^([A-Z])\w+([a-zA-Z]+[-'][a-zA-Z]+)|([a-zA-Z]+\.)|([a-zA-Z])+$")
    words = [w for w in words if regex.match(w)]
    try:
        word_pron = pronouncing.phones_for_word(word)[0].split()
    except:
        print('no pron ' + word)
        return 'a'

    index = -1
    for pron in reversed(word_pron):
        if not pron.isalpha():
            index = word_pron.index(pron)
            break

    word_prons = []
    # get array of the parts of the word pronunciation that must be compared for rhyming
    for wp in pronouncing.phones_for_word(word):
        wp = wp.split()[index - len(word_pron):]
        word_prons.append(wp)

    rhyme_list = []
    for w in words:  # look at all words in word mc
        if pronouncing.phones_for_word(w):  # if we can get the words pron
            for w_pron in pronouncing.phones_for_word(w):
                w_pron = w_pron.split()
                if len(w_pron) > (len(word_pron) - index) and w_pron[index - len(word_pron):] in word_prons:
                    rhyme_list.append(w)
                    break
    return rhyme_list


def rhymes(wd, mc):
    word = wd
    # case that there is "-" in the word
    if "-" in word:
        word = wd.split("-")[-1]
    # find all words that rhyme with word in words mc
    # loop through all of mc's keys of only words to see if their last pronunciation is the same as word
    regex = re.compile("([a-zA-Z]+[-'][a-zA-Z]+)|([a-zA-Z]+\.)")
    words = mc[wd]
    # filtering all punctuation
    words = {w: words[w] for w in words if w.isalpha()}

    # word_pron = [pron for w, pron in cmudict.entries() if w == word][0]

    word_pron = pronouncing.phones_for_word(word)[0].split()

    index = -1
    for pron in reversed(word_pron):
        if not pron.isalpha():
            index = word_pron.index(pron)
            break

    rhyme_list = {}
    for w in words:  # look at all words in word mc
        if pronouncing.phones_for_word(w):  # if we can get the words pron
            w_pron = pronouncing.phones_for_word(w)[0].split()
            if len(w_pron) > (len(word_pron) - index) and w_pron[index - len(word_pron):] == word_pron[
                                                                                             index - len(word_pron):]:
                rhyme_list[w] = words[w]

    return rhyme_list


def get_random_sent_struc(sents):
    regex = re.compile("^([A-Z])\w+([a-zA-Z]+[-'][a-zA-Z]+)|([a-zA-Z]+\.)|([a-zA-Z])+$")
    sent = random.choice(sents)
    sent = [s for s in sent if regex.match(s)]
    sent = nltk.word_tokenize(' '.join(sent))
    return nltk.pos_tag(sent)


def generate_abab(mc, sents):
    """
    Create a quatrain in ABAB form
    :param mc:
    :param sents:
    :return:
    """

    # start with a random word that will follow the markov chain
    # traversing the path in the markov chain will reflect the POS sentence structure

    # keep track of the last word in the sentence
    # if second A or B, make sure the last word rhymes with the first
    # keep track of the number of syllables
    # might need backtracking to get the right number of syllables
    a1 = generate_sentence(mc, get_random_sent_struc(sents))
    a_rhyme = a1[-1]
    b1 = generate_sentence(mc, get_random_sent_struc(sents), a_rhyme=a_rhyme)
    b_rhyme = b1[-1]
    a2 = generate_sentence(mc, get_random_sent_struc(sents), a_rhyme=a_rhyme, b_rhyme=b_rhyme)
    b2 = generate_sentence(mc, get_random_sent_struc(sents), b_rhyme=b_rhyme)

    return ' '.join(a1), ' '.join(b1), ' '.join(a2), ' '.join(b2),


def generate_couplet(mc, sents):
    """
    Create a sonnet
    :param sents:
    :param mc:
    :return:
    """
    first_sent = generate_sentence(mc, get_random_sent_struc(sents))
    a_rhyme = first_sent[-1]
    second_sent = generate_sentence(mc, get_random_sent_struc(sents), a_rhyme=a_rhyme, b_rhyme="placeholder")
    return ' '.join(first_sent), ' '.join(second_sent)


def syllables(word):
    """
    Get the number of syllables in a word
    :param word:
    :return:
    """
    consonants = ['A', 'E', 'I', 'O', 'U']
    try:
        if word in consonants:
            return 1
        if '-' in word:
            total_syllables = 0
            word_split = re.split("- | ' ", word)
            for word in word_split:
                total_syllables += syllables(word)
            return total_syllables
        # Syllables using pronouncing package
        pronouncing_list = pronouncing.phones_for_word(word)[0]
        syll_count = pronouncing.syllable_count(pronouncing_list)
        return syll_count
    except:  # case where the word is not in cmudict.entries()
        regex = re.compile('[aeiou]{2}')
        word_pron = regex.sub('a', word)
        regex = re.compile('[aeiou]')
        number_syllables = len(regex.findall(word_pron))
        return number_syllables


def main():
    sentences = [sent for sent in gutenberg.sents('austen-emma.txt')[3:] if 10 <= len(sent) <= 15]
    markov_chain = dict()
    generate_markov_chain(markov_chain)

    print("Enter '1' for sonnet. Enter '2' for couplet.")
    user_input = input()
    if user_input is 1:
        a, b, c, d = generate_abab(markov_chain, sentences)
        print(a + '.' + "\n" + b + '.' + "\n" + c + '.' + "\n" + d + '.')
        a, b = generate_couplet(markov_chain, sentences)
        print(a + '.' + "\n" + b + '.' + "\n")
    elif user_input is 2:
        a, b = generate_couplet(markov_chain, sentences)
        print(a + '.' + "\n" + b + '.' + "\n")


if __name__ == "__main__":
    main()

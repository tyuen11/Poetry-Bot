import random
import re
import nltk
from nltk.corpus import brown, treebank, cmudict, gutenberg
import numpy as np
import syllables as syllables_p
import pronouncing


def generate_markov_chain(mc, n_gram=2):
    """
    Generates a markov chain and puts it in a dictionary on the brown corpus
    Note: As text increases so will this function's runtime
    :param mc: dict
    :param n_gram: int
    :return:

    Examples
    ---------
    >>> generate_markov_chain({}, 2}
         {'In': {'American': 1, 'other': 5, 'the': 51, 'a': 17, 'one': 2, 'that': 2, 'this': 10, 'most': 1, ... } }
    >>> generate_markov_chain({}, 3}
    """
    text = brown.words(categories=["lore", "romance", "humor"])
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

def calculate_weights(mc, word):
    """
    Calculates the weights of a word's markov chain
    :param mc: dict
    :param word: string
    :return: weights: list

    Examples
    ---------
    >>> calculate_weights(markov_chain, 'investment')
        [0.5 0.5]
    >>> calculate_weights(markov_chain, 'income')
    [0.08333333 0.08333333 0.08333333 0.08333333 0.08333333 0.16666667
     0.08333333 0.08333333 0.08333333 0.08333333 0.08333333]
     """

    weights = np.array(
        list(mc[word].values()),
        dtype=np.float64)
    weights /= weights.sum()
    return weights


def generate_sentence(mc, start_node=None, num_syllables=10, a_rhyme=None, b_rhyme=None):
    """
    Generates a sentence from the markov chain, mc
    :param mc: dict
    :param sent_struc: list
    :param start_node: string
    :param num_syllables: int
    :param a_rhyme: string
    :param b_rhyme: string
    :return: list

    Examples
    ---------
    >>>generate_sentence(markov_chain)
    ['in', 'the', 'nearly', 'all', 'the', 'patent', 'rights', 'in']
    >>>generate_sentence(markov_chain, num_syllables=20)
    ['were', 'referred', 'to', 'have', 'hastened', 'to', 'respond', 'slowly', 'emerging', 'language', 'barrier']

    """

    # Base case is when the number of syllables is reached
    if num_syllables is 0:
        return []

    # Get a random word to start the sentence
    start = random.choice(list(mc)) if start_node is None else start_node
    weights = calculate_weights(mc, start)
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
            weights = calculate_weights(mc, start)
            chosen_words = []
    return [chosen_word] + generate_sentence(mc, start_node=chosen_word,
                                             num_syllables=new_num_syllables, a_rhyme=a_rhyme, b_rhyme=b_rhyme)


def get_rhyme_word(mc, a_rhyme, b_rhyme, syllable):
    """
    Gets a word of a certain syllable that rhymes with the a_rhyme or b_rhyme
    Calls rhyme_all_words()

    :param mc: dict
    :param a_rhyme: string
    :param b_rhyme: string
    :param syllable: int
    :return: chosen_word: list

    Examples
    --------
    >>> get_rhyme_word(markov_chain, 'the', None, 2)
    lightly
    >>> get_rhyme_word(markov_chain, None, 'people', 3)
    Seasonal

    """
    # Run rhyme_all_words()which will return rhymes of a a_rhyme or b_rhyme
    # filter out all rhymes that do not have syllable = syllable
    # Calculate the mc of the filtered rhymes and their probabilities
    # Randomly pick a rhyme based on its probability

    rhyme = a_rhyme if b_rhyme is None else b_rhyme

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
    :param word: string
    :param mc: dict
    :return: rhyme_list: list

    Examples
    --------
    >>>rhymes_all_words('people', markov_chain)
    ['adorable', 'example', 'little', 'professional', 'people', 'sentimental', 'mutual'... ]

    >>>rhymes_all_words('5-foot', markov_chain)
    ['put', 'underfoot', 'foot']
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

def syllables(word):
    """
    Get the number of syllables in a word
    :param word: string
    :return: number_syllables: int

    >>> syllables('dog')
    1
    >>> syllables('5-foot')
    2
    >>> syllables('mr.')
    2
    >>> syllables('ms.')
    1
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

def generate_abab(mc):
    """
    Create a quatrain in ABAB form
    :param mc: dict
    :return: tuple

    Examples
    --------
    >>> generate_abab_(markov_chain)
    Before audiences began it is.
    Everything was found a wound One of their.
    Their needs only should expect But his Does.
    At all six novel The strings were eased bear.
    Were reinstated at the fact wreck a.
    A little farther along the villa.
    """

    # start with a random word that will follow the markov chain
    # traversing the path in the markov chain will reflect the POS sentence structure

    # keep track of the last word in the sentence
    # if second A or B, make sure the last word rhymes with the first
    # keep track of the number of syllables
    # might need backtracking to get the right number of syllables
    a1 = generate_sentence(mc)
    a_rhyme = a1[-1]
    b1 = generate_sentence(mc, a_rhyme=a_rhyme)
    b_rhyme = b1[-1]
    a2 = generate_sentence(mc, a_rhyme=a_rhyme, b_rhyme=b_rhyme)
    b2 = generate_sentence(mc, b_rhyme=b_rhyme)

    return ' '.join(a1).capitalize(), ' '.join(b1).capitalize(), ' '.join(a2).capitalize(), ' '.join(
        b2).capitalize(),

def generate_couplet(mc):
    """
    Create a sonnet
    :param mc: dict
    :return: tuple

    Examples
    --------
    >>> generate_couplet(markov_chain)
    Determines whether a short time and was.
    Of the island as she called her Causes.
    """
    first_sent = generate_sentence(mc)
    a_rhyme = first_sent[-1]
    second_sent = generate_sentence(mc, a_rhyme=a_rhyme, b_rhyme="placeholder")
    return ' '.join(first_sent).capitalize(), ' '.join(second_sent).capitalize()


def main():
    """
    """
    markov_chain = dict()
    generate_markov_chain(markov_chain, 2)
    print("Enter 1 for a sonnet \n"
          "Enter 2 for a quatrain \n"
          "Enter 3 for couplet")
    user_input = input()
    print()
    if user_input is '1':
        for x in range(3):
            a, b, c, d = generate_abab(markov_chain)
            print(a + '.' + "\n" + b + '.' + "\n" + c + '.' + "\n" + d + '.')
        a, b = generate_couplet(markov_chain)
        print(a + '.' + "\n" + b + '.')
    elif user_input is '2':
        a, b, c, d = generate_abab(markov_chain)
        print(a + '.' + "\n" + b + '.' + "\n" + c + '.' + "\n" + d + '.')
        a, b = generate_couplet(markov_chain)
    else:
        a, b = generate_couplet(markov_chain)
        print(a + '.' + "\n" + b + '.' + "\n")

if __name__ == "__main__":
    main()

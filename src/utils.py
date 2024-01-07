from nltk import word_tokenize, pos_tag
from collections import defaultdict


def count_digits(text):
    digit_count = 0
    for char in text:
        if char.isdigit():
            digit_count += 1
    return digit_count


def count_pos_tags(text):
    if not isinstance(text, str):
        text = ''
    
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    pos_tag_frequency_dict = defaultdict(int)
    
    # Define the POS tags of interest
    pos_tags_of_interest = {'VB': 'VRB', 'VBD': 'VRB', 'VBG': 'VRB', 'VBN': 'VRB', 'VBP': 'VRB', 'VBZ': 'VRB',
                   'NN': 'NN', 'NNS': 'NN', 'NNP': 'NN', 'NNPS': 'NN',
                   'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
                   'MD': 'MDA',
                   'PDT': 'PD',
                   'UH': 'I',
                   'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV',
                   'WDT': 'WH', 'WP': 'WH', 'WP$': 'WH', 'WRB': 'WH',
                   'PRP': 'PN', 'PRP$': 'PN'}
    
    # Count the frequency of each POS tag
    for _, tag in pos_tags:
        if tag in pos_tags_of_interest:
            mapped_tag = pos_tags_of_interest[tag]
            pos_tag_frequency_dict[mapped_tag] += 1

    pos_tag_frequency_dict = {mapped_tag: pos_tag_frequency_dict[mapped_tag] 
                              for mapped_tag in pos_tags_of_interest.values()}
    
    return pos_tag_frequency_dict
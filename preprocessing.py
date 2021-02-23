import re
import string
import nltk
from nltk.corpus import stopwords

ps = nltk.stem.PorterStemmer()
stopwords_set = set(stopwords.words('english'))

def replace_umlaute(str):
    umlaut_map = {
        '\u00dc': 'UE',
        '\u00c4': 'AE',
        '\u00d6': 'OE',
        '\u00fc': 'ue',
        '\u00e4': 'ae',
        '\u00f6': 'oe',
        '\u00df': 'ss',
    }

    for k,v in umlaut_map.items():
        str = str.replace(k, v)
    return str

def preprocessing_normalization_func(text_in, ignore_caps=True):
    replacements = {
        '.': '',
        ',': '',
        '!': '',
        '?': '',
        '…': '',
        ':': '',
        ';': '',
        '-': '',
        '_': '',

        "'ve ": ' have ',
        "'re ": ' are ',
        "'m ": ' am ',
        "'em ": ' them ',
        " he's ": ' he is ',
        "'d ": ' would',
        "n't ": ' not ',
        " y'": ' you ',
        "'ll ": ' will ',
        " kinda ": ' kind of ',
        " gonna ": ' going to ',
        " wanna ": ' want to ',
        " dunno ": ' do not know ',
        " because ": ' cause ',
        " ma'am ": ' madam ',

        ' & ': ' and ',
        " <unk> ": ' ',
    }
    uhms = ['uh', 'oh', 'em', 'um', 'ah', 'uhum', 'mmhmm', 'uhm', 'ahm', 'ähm', 'äh', 'ähmm', 'ähh']
    for uhm in uhms:
        replacements[' '+uhm+' '] = ' '

    if ignore_caps:
        text_in = text_in.lower() # lowercase

    # Remove annotations (e.g. "[laughter]"")
    text_in = re.sub("\[[a-zA-Z0-9]*\]", '', text_in)

    # Remove newline characters
    text_in = ' ' + text_in.replace('\r', ' ').replace('\n', ' ')
    
    # text_in = ' ' + text_in

    for k, v in replacements.items():
        text_in = text_in.replace(k, v)

    return text_in

def ewer_normalization_func(txt):
    txt = txt.replace('.','').replace('?','').replace(',','') # remove punctuation
    txt = [' '.join(w) for w in txt.split() if w not in stopwords_set] # remove stopwords
    txt = replace_umlaute(txt)
    txt = ' '.join([ps.stem(w) for w in txt.split()]) # Apply stemming
    txt = txt.replace(' ', '')
    return txt

def remove_punctuation(text):
    """
    Removes the punctuation r\"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~\""" from the string and replaces them with an
    empty string.
    :param text: The text to process
    :return: The input text without the punctuation symbols
    """
    transtable = str.maketrans('', '', string.punctuation)
    return text.translate(transtable)


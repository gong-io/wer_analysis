import subprocess
import shutil
import re
import os
import numpy as np
import pandas as pd
import json
from zipfile import ZipFile
import string
from collections import Counter
import s3fs
from pyathena import connect

from IPython.core.display import display, HTML
import Levenshtein


# from nlp.punctuation.src.utils.strip_punctuations import extract_punctuation_marks
# from nlp.punctuation.src.utils.capitalize_text import capitalize_txt
# import nlp.punctuation.src.data.parse_json as parse_json



def wrap(txt, cls=None):
    # return txt+' '
    if cls is not None:
        return '<span class="{cls}" title="{cls}">{txt}</span>'.format(cls=cls, txt=txt)
    return txt

def get_css():
    result = """
    <style>
    .rendered_html {
        color: #999;
        font-family: monospace;
        /* white-space: pre-wrap; */
    }
    span.insert {
        background: green;
        font-weight: bold;
        color: white;
        display: flex;
    }
    span.delete {
        background: red;
        font-weight: bold;
        color: white;
        display: flex;
    }
    span.block {
        display: inline-block;
        margin-bottom: 2em;
    }
    .probability_tower {
        position: relative;
        min-width: 1em;
        display: inline-block;
    }
    </style>    
        """
    return result

def print_css():
    return display(HTML(get_css()))


def get_html_of_edits(df):
    s_joined = ''
    for i, r in df.iterrows():
        if r['edit_tag'] == 'equal':
            txt = r['text_reference']
        elif r['edit_tag'] == 'delete':
            txt = wrap(r['text_reference'], 'delete')
        elif r['edit_tag'] == 'insert':
            txt = wrap(r['text_hypothesis'], 'insert')
        elif r['edit_tag'] == 'replace':
            txt = wrap(r['text_reference'], 'delete')
            txt += wrap(r['text_hypothesis'], 'insert')
        s_joined += wrap(txt, 'block') + ' '

    return s_joined






























sentence_termination_marks = ['.', '?']
TH_PAUSE = 1.5

def parse_monologues_to_word_dicts(monologues):
    result = []
    for monologue_id, monologue in enumerate(monologues):
        extra_data = {
            'paragraph_id': monologue_id
        }
        if 'speaker' in monologue:
            extra_data['speaker_id'] = monologue['speaker'].get('id')
            extra_data['speaker_name'] = monologue['speaker'].get('name')

        if 'terms' in monologue:
            for i,word in enumerate(monologue['terms']):
                if word.get('type', 'PUNCTUATION')=='PUNCTUATION':
                    if i>0 and len(result)>0:
                        result[-1]['punctuation'] = word['text']
                        continue
                    else:
                        continue
                if word.get('duration', None) is None:
                    word['duration'] = float(word['end']) - float(word['start'])
                word['text'] = word['text'].replace(' ', '%^%^%^%').strip()
                word.update(extra_data)
                result.append(word)

            result = enrich_words_json(result)

        if 'tokens' in monologue:
            for i, word in enumerate(monologue['tokens']):
                word = {
                    'start':        word['s'],
                    'duration':     word['d'],
                    'end':          word['s']+word['d'],
                    'text':         word['t'],
                    'type':         word.get('i', 'word')
                }
                if word.get('type', '')=='P':
                    if i>0 and len(result)>0:
                        result[-1]['punctuation'] = word['text']
                        continue
                    else:
                        continue
                word.update(extra_data)
                result.append(word)
            result = enrich_words_json(result)

    return result

def terminate_sentence_func_sentence_termination_marks(word, sentence):
    result = \
        word['text'] in sentence_termination_marks \
        or word['text'][-1] in sentence_termination_marks \
        or word['speaker_change'] \
        or word['paragraph_change']

    return result

def terminate_sentence_func_sentence_termination_marks_in_punctuation_field(word, sentence):
    result = \
        word['punctuation'] in sentence_termination_marks \
        or word['speaker_change'] \
        or word['paragraph_change']

    return result

def terminate_sentence_func_pause(word, sentence):
    result = \
        word['pause'] > TH_PAUSE \
        or word['speaker_change'] \
        or word['paragraph_change']

    return result

def enrich_words_json(words):
    L = len(words)-1

    for i,word in enumerate(words):
        # Pause
        if i<L:
            word['pause'] = float(words[i+1]['start']) - float(word['end'])
        else:
            word['pause'] = np.nan

        # Speaker change
        if i<L:
            word['speaker_change'] = ( words[i+1].get('speaker_id',0) != word.get('speaker_id',0) )
        else:
            word['speaker_change'] = True

        # Paragraph change
        if i < L:
            word['paragraph_change'] = (words[i + 1].get('paragraph_id', 0) != word.get('paragraph_id', 0))
        else:
            word['speaker_change'] = True
        word['index'] = i

    return words

def enrich_sentences_dict(sentences):
    for sentence in sentences:
        if sentence['duration']>0:
            sentence['chars_per_second'] = len(sentence['text']) / sentence['duration']
        else:
            sentence['chars_per_second'] = np.nan
    return sentences

def calc_words_stats(words, k, func):
    return func([word[k] for word in words])

def group_timed_words_to_sentences(words, additional_data={}, terminate_sentence_func=None, return_words=False):
    """
    Turn JSON transcript data that includes timing for each word into a list of sentences with relevant stats
    :param words: A list of dict values, each with the following fields:
                text
                start
                end
                paragraph_id
    :param additional_data: Arbitrary dict that will be merged with data of each sentence. Can be used for example to pass filename

    :return: A list of sentences, each being a dict with the following fields:
                text
                duration
                start
                end
                paragraph_id
                sentence_id
                relative_time
    """
    # init values
    result = []
    text = ''
    is_last_word_in_sentence = False
    is_first_word_in_sentence = True
    # sentence_longest_inner_pause = 0
    # sentence_total_inner_pause = 0
    sentence_words = []
    sentence_id = 0
    sentence_start_id = 0 # Number of word that starts the sentence

    if terminate_sentence_func is None:
        terminate_sentence_func = terminate_sentence_func_sentence_termination_marks

    for ind, word in enumerate(words):
        if len(word['text'])==0:
            continue

        # Ignore comments in text (e.g. "[inaudible]" )
        if word['text'][0] == '[' and word['text'][-1] == ']':
            continue

        word['index'] = ind

        if is_first_word_in_sentence:
            sentence_start_id = ind

        text += word['text'] + word.get('punctuation', ' ')
        if text[-1]!=' ':
            text += ' '

        if word['text'] in sentence_termination_marks:
            # Word is actually a punctuation mark - so remove the spaces we added automatically in previous steps
            # This code turns for example the text "Okay ? " to "Okay? "
            text = text[:-3] + text[-2:]

        # if terminate_sentence_func(word, text, previous_sentence):
        if terminate_sentence_func(word, text):
            is_last_word_in_sentence = True
        else:
            pass
            # sentence_total_inner_pause += word['pause']
            # if word['pause'] > sentence_longest_inner_pause:
            #     # Keep track of the longest pause during the sentence (it might indicate talk speed and hesitation
            #     sentence_longest_inner_pause = word['pause']

        # sentence_words = words[sentence_start_id:ind+1]
        sentence_words.append(word)
        if is_last_word_in_sentence:
            sentence_id += 1
            sentence_data = {
                'text': text.strip(),
                'duration': word['end'] - sentence_words[0]['start'],
                'start': sentence_words[0]['start'],
                'end': word['end'],
                'pause': word.get('pause', 0),
                # 'longest_inner_pause': sentence_longest_inner_pause,
                'longest_inner_pause': calc_words_stats(sentence_words, 'pause', np.nanmax),
                # 'total_inner_pause': sentence_total_inner_pause,
                'total_inner_pause': calc_words_stats(sentence_words, 'pause', np.nansum),
                'paragraph_id': word.get('paragraph_id', 1),
                'sentence_id': sentence_id,
                'speaker_change': word.get('speaker_change', 0),
                'paragraph_change': word.get('paragraph_change', 0),
                'speaker_id': word.get('speaker_id', '')
            }


            if return_words:
                sentence_data['words'] = sentence_words

            sentence_data.update(additional_data)
            result.append(sentence_data)
            # previous_sentence = sentence_data
            text = ''
            # sentence_longest_inner_pause = 0
            # sentence_total_inner_pause = 0
            sentence_words = []
            is_last_word_in_sentence = False
            is_first_word_in_sentence = True # Next word will be the first in sentence
        else:
            is_first_word_in_sentence = False

    # Add data on relative time of sentence in call
    call_duration = result[-1]['end']
    for sentence in result:
        sentence['relative_time'] = sentence['start'] / call_duration

    return result


class TimedWords:
    def __init__(self):
        self._words = []

    def __iter__(self):
        for word in self._words:
            yield word

    def __getitem__(self, key):
        return self._words[key]

    def from_list(self, list_timed_words, append=False):
        if append:
            words = self._words + list_timed_words._words
            self.words = words
        else:
            self.words = list_timed_words
        self.extract_punctuations()
        return self

    def from_json_file(self, filename):
        if '.zip' in filename:
            parts = filename.split('.zip')
            with ZipFile(parts[0]+'.zip') as f_zip:
                with f_zip.open(parts[1][1:], 'r', encoding='utf-8') as f:
                    data = json.load(f)
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

        self.words = parse_monologues_to_word_dicts(data['monologues'])
        self.extract_punctuations()
        return self

    def reset_punctuations(self):
        for i, word in enumerate(self._words):
            self._words[i]['punctuation'] = ''
        return self

    @property
    def words(self):
        # return [dict(w, **({'text': w.get('text','').replace('----', ' ')})) for w in self._words]
        return self._words

    @words.setter
    def words(self, value):
        self._words = enrich_words_json(value)

    @property
    def length(self):
        return len(self._words)

    @property
    def text_by_speaker(self):
        result = []
        sentence_words = []
        for word in self._words:
            sentence_words.append(word)
            if word['speaker_change']:
                result.append((word.get('speaker_id', 0), TimedWords().from_list(sentence_words).text))
                sentence_words = []
        if len(sentence_words):
            result.append((sentence_words[-1]['speaker_id'], TimedWords().from_list(sentence_words).text))
        return result

    def text_by_sentences(self, attribute):
        sentences = getattr(self, attribute)
        result = [(sentence['speaker_id'], sentence['text']) for sentence in sentences]
        return result

    @property
    def sentences_by_monolog(self):
        def terminate_sentence_by_monologue(word, text):
            return word['speaker_change'] or word['paragraph_change']

        sentences = group_timed_words_to_sentences(self.words,
                                                   terminate_sentence_func=terminate_sentence_by_monologue,
                                                   return_words=True)
        sentences = enrich_sentences_dict(sentences)
        return sentences

    @property
    def sentences_by_pause(self, terminate_sentence_func=terminate_sentence_func_pause):
        sentences = group_timed_words_to_sentences(self.words,
                                                   terminate_sentence_func=terminate_sentence_func,
                                                   return_words=True)
        sentences = enrich_sentences_dict(sentences)
        return sentences

    @property
    def sentences_by_punctuation(self):
        sentences = group_timed_words_to_sentences(self.words,
                                                   terminate_sentence_func=terminate_sentence_func_sentence_termination_marks_in_punctuation_field,
                                                   return_words=True)
        sentences = enrich_sentences_dict(sentences)
        return sentences

    def get_text(self, return_punctuation=True, capitalize_text=True):
        if return_punctuation:
            result = " ".join([word['text'] + word.get('punctuation', '') for word in self.words])
        else:
            result = " ".join([word['text'] for word in self.words])
        # result = result.replace('----', ' ')
        return result

    @property
    def text(self):
        return self.get_text(capitalize_text=True)

    def update(self, list_with_mapped_values):
        if len(list_with_mapped_values) != self.length:
            raise ValueError("Error! Length of values to update should match length of words.")

        for word, new_values in zip(self._words, list_with_mapped_values):
            word.update(new_values)
        return self

    def extract_punctuations(self, drop_punctuations=False):
        drop_punctuations = True
        for word in self._words:
            # if 'punctuation' in word:
            #     continue
            word['text'] = word['text'].replace('.', '').replace(',', '').replace('?', '').replace('!', '')
        return self

    def find_word_by_time(self, word_start_time):
        for word in self.words:
            if word['start'] >= word_start_time:
                return word

    def find_sentence_by_time(self, section_start_time, section_end_time=None):
        sentence_start_index = 0
        sentence_end_index = self.length-1

        if section_end_time is None:
            section_end_time = section_start_time
        start_word = self.find_word_by_time(section_start_time)
        end_word = self.find_word_by_time(section_end_time)

        for word in self.words[start_word['index']::-1]:
            if word['punctuation'] in ['.', '?']:
                sentence_start_index = word['index']+1
                break
        for word in self.words[end_word['index']-1:]:
            if word['punctuation'] in ['.', '?']:
                sentence_end_index = word['index']
                break
        return self.words[sentence_start_index:sentence_end_index+1]

    def find(self, text):
        index_in_text = self.text.lower().find(text.lower())
        if index_in_text:
            num_word = len(self.text[:index_in_text].split())
            return num_word
        return None


        # TODO: Write function (currently it's not working)
        text_list = text.strip().split() #.lower()
        current_counter = 0
        for word_ind, word in enumerate(self.words):
            ind = word_ind
            while self.words[ind]['text'] == text_list[current_counter]:
                ind += 1

    def to_dataframe(self):
        df = pd.DataFrame(self.words)
        return df

    def to_json(self):
        df = pd.DataFrame(self.words)
        return df

    def set(self, index, key, value):
        self.words[index].set(key, value)


def read_ctm(fname):
    ctm = np.loadtxt(fname, str)
    ctm = ctm[ctm[:, 2].argsort()]  # sort by time
    return ' '.join(ctm[:, 4])


def read_text(fname):
    ref_text = ''
    with open(fname) as fin:
        ref_text = fin.read()
    return ref_text


def read_file(fname):
    if fname.endswith('.ctm'):
        func = read_ctm
    elif fname.endswith('.txt'):
        func = read_text
    elif fname.endswith('.json'):
        func = lambda x: TimedWords().from_json_file(x).get_text(capitalize_text=False)
        # from gong_utils.transcript_reader import TranscriptReader
        # func = lambda x: TranscriptReader(x).get_text()
    else:
        raise ValueError('Unexpected file format')
    return func(fname)


def simple_norm(text_in):
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

        ' uh ': ' ',
        ' oh ': ' ',
        ' em ': ' ',
        ' um ': ' ',
        ' ah ': ' ',
        ' uhum ': ' ',
        ' mmhmm ': ' ',
        ' & ': ' and ',
        " <unk> ": ' ',
    }
    text_in = text_in.lower()
    text_in = re.sub("\[[a-zA-Z0-9]*\]", '', text_in)

    text_in = ' ' + text_in.replace('\r', ' ').replace('\n', ' ')
    # text_in = ' ' + text_in

    for k, v in replacements.items():
        text_in = text_in.replace(k, v)

    return text_in


def to_hms(seconds):
    return str(datetime.timedelta(seconds=float(seconds)))[:-4]


def get_edit_distance_verbosely(ref_lst, hyp_lst):
    alphabet = set(ref_lst + hyp_lst)
    word2char = {k: chr(i) for i, k in enumerate(alphabet)}
    char2word = {v: k for k, v in word2char.items()}

    enc_ref = ''.join([word2char[w] for w in ref_lst])
    enc_hyp = ''.join([word2char[w] for w in hyp_lst])

    printout = ''
    compare = []
    wer = Counter()
    prev_res = 'None', 0, 0, 0, 0
    opcodes = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(enc_ref, enc_hyp):
        if tag == 'replace' and prev_res[0] in ['insert', 'delete']:
            prev_res = opcodes.pop()
            opcodes.append((tag, prev_res[1], i2, prev_res[3], j2))
        else:
            opcodes.append((tag, i1, i2, j1, j2))
        prev_res = tag, i1, i2, j1, j2

    for tag, i1, i2, j1, j2 in opcodes:
        words_ref = ' '.join([char2word[x] for x in enc_ref[i1:i2]])
        words_rcg = ' '.join([char2word[x] for x in enc_hyp[j1:j2]])
        if tag == 'replace':
            match = True
            for ltag, li1, li2, lj1, lj2 in Levenshtein.opcodes(words_ref, words_rcg):
                if ltag == 'equal' or \
                        (ltag in ['delete', 'insert'] and
                         ((words_ref[li1:li2] or words_rcg[lj1:lj2]) in ['s', 'ed', 'd', ' '])):
                    # TODO: Shouldn't we still log these? and give them a low EWER score?
                    continue
                match = False

            if match:
                weight = i2 - i1
                tag = 'equal'
                comments = ''
                wer.update({'equal': i2 - i1})
                printout += (f'{i1:5d}:{"equal":8s} {words_ref} === {words_rcg}')
                compare.append((i1, tag, words_ref, words_rcg, weight, comments))
                continue
            else:
                pass
                printout += (f'{i1:5d}:{tag:8s} {words_ref} <> {words_rcg}')
                comments = ''
                # compare.append((i1, tag, words_ref, words_rcg, ''))

        elif tag in ['delete', 'insert'] and ((words_ref or words_rcg) in [
            'in', 'it', 'its', 'of', 'to', 'okay', 'a', 'the', 'and', 'for', 'so', 'yeah', 'yep', 'yes', 'yup'
        ]):
            pass
            printout += (f'{i1:5d}:{tag:8s} {words_ref or "<<=== " + words_rcg}{" ===--" if not words_rcg else ""}')
            comments = '<<==='
            # compare.append((i1, tag, words_ref, words_rcg, '<<==='))

        else:
            pass
            printout += (f'{i1:5d}:{tag:8s} {words_ref or "<< " + words_rcg}{" --" if not words_rcg else ""}')
            comments = '<<'
            # compare.append((i1, tag, words_ref, words_rcg, '<<'))

        weight = i2 - i1 or j2 - j1
        compare.append((i1, tag, words_ref, words_rcg, weight, comments))
        wer.update({tag: i2 - i1 or j2 - j1})

    nom = wer["insert"] + wer["delete"] + wer["replace"]
    den = wer["equal"] + wer["delete"] + wer["replace"] - 1

    return nom, den, printout, compare


def print_wer(ref_path, hyp_path, norm_func=None, verbose=True):
    ttl_dist, ttl_length = 0, 0
    for fn, ref_lst, hyp_lst in generate_file_contents(ref_path, hyp_path, norm_func):
        if verbose:
            dist, length, printout, compare = get_edit_distance_verbosely(ref_lst, hyp_lst)
        else:
            pass
            # dist, length = get_edit_distance(ref_lst, hyp_lst)

        print(f'{fn:25} {100 * dist / length:.2f}   {dist} / {length}')

        ttl_dist += dist
        ttl_length += length
        # break
    print(f'Total {100 * ttl_dist / ttl_length:.2f}  {ttl_dist} / {ttl_length}')



def remove_punctuation(text):
    """
    Removes the punctuation r\"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~\""" from the string and replaces them with an
    empty string.
    :param text: The text to process
    :return: The input text without the punctuation symbols
    """
    transtable = str.maketrans('', '', string.punctuation)
    return text.translate(transtable)


def generate_file_contents(ref_path, hyp_path, norm_func, limit=None):
    counter = 0
    ref_fnames = {f.split('.')[0].replace('-test50', ''): f for f in os.listdir(ref_path) if f[0].isdigit()}
    hyp_fnames = {f.split('.')[0].replace('-test50', ''): f for f in os.listdir(hyp_path) if f[0].isdigit()}

    for fn in ref_fnames:
        if isinstance(limit, int) and counter >= limit:
            return

        if fn not in hyp_fnames:
            print(f'!!!! File corresponding to {ref_path + os.sep + ref_fnames[fn]} is not found in {hyp_path}')
            continue

        ref_text = read_file(ref_path + os.sep + ref_fnames[fn])
        hyp_text = read_file(hyp_path + os.sep + hyp_fnames[fn])

        if norm_func:
            ref_text = norm_func(ref_text)
            hyp_text = norm_func(hyp_text)

        ref_lst = ref_text.split()
        hyp_lst = hyp_text.split()
        counter += 1
        yield fn, ref_lst, hyp_lst


def get_edit_df(REF_PATH, HYP_PATH, norm_func=simple_norm, limit=None):
    full_compare = []
    for fn, ref_lst, hyp_lst in generate_file_contents(REF_PATH, HYP_PATH, norm_func, limit):
        dist, length, printout, compare = get_edit_distance_verbosely(ref_lst, hyp_lst)
        full_compare.extend([[fn] + list(x) for x in compare])

    df = pd.DataFrame(full_compare,
                      columns=['filename', 'word_start_index', 'edit_tag', 'text_reference', 'text_hypothesis',
                               'weight', 'comments'])
    return df


def save_to_s3(data, s3_filename, format=None):
    s3_filename = s3_filename.replace('//', '/').replace('s3:/', 's3://')

    if format is None:
        format = s3_filename.split('.')[-1]

    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)

    if isinstance(data, pd.DataFrame):
        if format=='csv':
            data = data.to_csv()
        elif format == 'tsv':
            data = data.to_csv(sep='\t')
        elif format=='json':
            data = data.T.to_json()
        elif format=='html':
            data = data.to_html()
        else:
            data = data.to_html()

    s3 = s3fs.S3FileSystem(anon=False)
    with s3.open(s3_filename, 'wb') as f:
        f.write(data.encode())
    return data

def save_transcript_compare_html_to_s3(df, s3_filename):
    save_to_s3((get_css() + get_html_of_edits(df)), s3_filename)


def display_transcript_compare_html(df):
    display(HTML((get_css() + get_html_of_edits(df))))


def get_pivot_table_of_edits(df, groupby=['filename']):
    _ = df.groupby(groupby + ['edit_tag'])['weight'].sum()
    # filenames = [z for z in _.index.levels[0]]
    # Create a pivot table of edit tags by filename
    df_edit_counts = _.reset_index().pivot_table(values='weight', index='filename', columns='edit_tag').fillna(0)
    df_edit_counts['edits'] = df_edit_counts.get('insert', 0) + df_edit_counts.get('delete', 0) + df_edit_counts.get(
        'replace', 0)
    df_edit_counts['denominator'] = df_edit_counts.get('equal', 0) + df_edit_counts.get('delete',
                                                                                        0) + df_edit_counts.get(
        'replace', 0) - 1
    df_edit_counts['wer'] = df_edit_counts['edits'] / df_edit_counts['denominator'] * 100
    return df_edit_counts


def get_top_errors(df, groupby=['text_reference', 'text_hypothesis']):
    # Get the most common errors
    return df.query('edit_tag!="equal"').reset_index().groupby(groupby)['index'].count().sort_values(
        ascending=False).reset_index()


def copy_s3_folder_to_local_folder(s3_folder, local_folder):
    shutil.rmtree(local_folder)
    os.makedirs(local_folder, exist_ok=True)
    os.system(f'aws s3 cp {s3_folder} {local_folder} --recursive')

def install_required_packages():
    output = subprocess.check_output("pip install PyAthena python-Levenshtein", shell=True)
    return output.decode()

def get_calls_metadata(filenames):
    if len(filenames)==0:
        raise ValueError('Please pass valid call IDs')

    conn = connect(s3_staging_dir='s3://gong-transcripts-aws-glue/notebook-temp/bla',
                   region_name='us-east-1')
    athena_query = """
    SELECT *
    FROM awsdatacatalog.research.bi_call_facts
    WHERE 
    call_id IN ({})
    """.format(','.join([str(c) for c in filenames]))
    df_calls_metadata = pd.read_sql(athena_query, conn)
    return df_calls_metadata


def analyze_wer_folders(folder_truth, folder_hypothesis, folder_output):
    print('Copying truth files...')
    copy_s3_folder_to_local_folder(folder_truth, './data/truth')
    print('Copying hypothesis files...')
    copy_s3_folder_to_local_folder(folder_hypothesis, './data/hypothesis')

    print('Computing transcription differences...')
    REF_PATH = './data/truth'
    HYP_PATH = './data/hypothesis'
    df = get_edit_df(REF_PATH, HYP_PATH, norm_func=simple_norm, limit=None)
    df['filename'] = df['filename'].astype(int)
    print(f'Found {df.shape[0]} differences in {df["filename"].nunique()} files.')

    average_wer = get_pivot_table_of_edits(df, groupby=['filename'])['wer'].mean()
    print(f'Average WER is {average_wer}')

    filenames = df['filename'].unique()
    df_calls_metadata = get_calls_metadata(filenames)
    df_calls_metadata['gong_link'] = [f'https://app.gong.io/call?id={call_id}' for call_id in df_calls_metadata['call_id']]
    df_calls_metadata['speaker_count_total'] = df_calls_metadata['speaker_count_in_company'] + df_calls_metadata['speaker_count_outside_company'] + df_calls_metadata['speaker_count_company_unknown']
    df_calls_metadata['speaker_count_total'] = df_calls_metadata['speaker_count_total'].fillna(0)

    wer_by_filename_with_metadata = pd.merge(left=get_pivot_table_of_edits(df), right=df_calls_metadata, left_on='filename', right_on='call_id')
    save_to_s3( wer_by_filename_with_metadata, s3_filename=folder_output+'/wer_by_filename_with_metadata.csv')

    wer_by_company = wer_by_filename_with_metadata.groupby('company_name')['wer'].mean()
    save_to_s3( wer_by_company, s3_filename=folder_output+'/wer_by_company.csv')

    wer_by_conferencing_provider = wer_by_filename_with_metadata.groupby('conferencing_provider')['wer'].mean()
    save_to_s3( wer_by_conferencing_provider, s3_filename=folder_output+'/wer_by_conferencing_provider.tsv')

    wer_by_field = lambda x: wer_by_filename_with_metadata.groupby(x)['wer'].describe().sort_values('mean')

    print('\n=== WER by language: ===')
    print( wer_by_field('language') )

    print('\n=== WER by internal_meeting: ===')
    print( wer_by_field('internal_meeting') )

    print('\n=== WER by direction: ===')
    print( wer_by_field('direction') )

    print('\n=== WER by owner_name: ===')
    print( wer_by_field('owner_name') )

    print('\n=== WER by speaker_count_total: ===')
    print( wer_by_field('speaker_count_total') )

    print('Saving HTML of transcription differences...')
    # Save HTML of edits
    for filename in df['filename'].unique():
        save_transcript_compare_html_to_s3(df[df.filename == filename], s3_filename=folder_output+f'/transcription_edits_{filename}.html')

    # Top edits
    save_to_s3(get_top_errors(df), s3_filename=folder_output + '/top_edits.tsv')
    # Top errors
    save_to_s3( get_top_errors(df, groupby=['text_reference']), s3_filename=folder_output+'/top_errors.tsv' )

    transcription_edits_with_metadata = pd.merge(left=df, right=df_calls_metadata, left_on='filename', right_on='call_id')
    save_to_s3( transcription_edits_with_metadata, s3_filename=folder_output+'/transcription_edits_with_metadata.csv')

    return transcription_edits_with_metadata

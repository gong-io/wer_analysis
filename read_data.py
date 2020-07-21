import os
import numpy as np
import json
import pandas as pd
from zipfile import ZipFile


def generate_file_contents(ref_path, hyp_path, norm_func, limit=None):
    counter = 0
    def transform_filename(filename):
        return filename.split('.')[0].replace('-test50', '')

    ref_fnames = {transform_filename(f): f for f in os.listdir(ref_path) if f[0].isdigit()}
    hyp_fnames = {transform_filename(f): f for f in os.listdir(hyp_path) if f[0].isdigit()}
    print(f'!!!! Found {len(ref_fnames)} files in the ref folder and {len(hyp_fnames)} files in the hypothesis folder.')

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
            for i, word in enumerate(monologue['terms']):
                if word.get('type', 'PUNCTUATION') == 'PUNCTUATION':
                    if i > 0 and len(result) > 0:
                        result[-1]['punctuation'] = word['text']
                        continue
                    else:
                        continue
                if word.get('duration', None) is None:
                    if word.get('start') is not None and word.get('end') is not None:
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
        if i<L and word.get('start') is not None and word.get('end') is not None:
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

    def __str__(self):
        from datetime import datetime
        for monolog in self.sentences_by_monolog:
            print(f"{datetime.timedelta(seconds=monolog['start'])}\t{monolog['speaker']}: \n{monolog['text']}\n")

def read_ctm(fname):
    ctm = np.loadtxt(fname, str)
    ctm = ctm[ctm[:, 2].argsort()]  # sort by time
    return ' '.join(ctm[:, 4])


def read_text(fname):
    ref_text = ''
    with open(fname, encoding='utf-8') as fin:
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


if __name__=='__main__':
    x = r'C:\data\wer\confidence\hyp\153923516977231343_out.json'
    x = TimedWords().from_json_file(x)
    print(x)
    z = 1

import subprocess
from collections import Counter
import s3fs
from pyathena import connect
import copy
import datetime

from preprocessing import *
from read_data import *
from s3 import *
from html_display import *

import Levenshtein


# from nlp.punctuation.src.utils.strip_punctuations import extract_punctuation_marks
# from nlp.punctuation.src.utils.capitalize_text import capitalize_txt
# import nlp.punctuation.src.data.parse_json as parse_json


def to_hms(seconds):
    return str(datetime.timedelta(seconds=float(seconds)))[:-4]


class Err_Stat:
    def __init__(self, ins, sub, del_, total):
        self.ins_num = ins
        self.sub_num = sub
        self.del_num = del_
        self.total_cost = total
        self.tag = []

    def __repr__(self):
        return (f'[{self.total_cost} {self.ins_num} ins, {self.del_num} del, {self.sub_num} sub] {self.tag}')


def get_edit_distance_verbosely(ref_lst, hyp_lst):
    alphabet = set(ref_lst + hyp_lst)
    word2char = {k: chr(i) for i, k in enumerate(alphabet)}
    char2word = {v: k for k, v in word2char.items()}

    enc_ref = ''.join([word2char[w] for w in ref_lst])
    enc_hyp = ''.join([word2char[w] for w in hyp_lst])

    wer = Counter()
    opcodes = Levenshtein.opcodes(enc_ref, enc_hyp)
    for tag, i1, i2, j1, j2 in opcodes:
        wer.update({tag: i2 - i1 or j2 - j1})

    # print(wer)
    nom = wer["insert"] + wer["delete"] + wer["replace"]
    den = wer["equal"] + wer["delete"] + wer["replace"]

    return nom, den, opcodes


def get_edit_distance_kaldi(ref_lst, hyp_lst):
    '''
    This function is an implementation of the WER code in Kaldi, that attempts to be an exact copy.
    It is slow and inefficient, and is not recommended for use.
    :param ref_lst:
    :param hyp_lst:
    :return:
    '''
    alphabet = set(ref_lst + hyp_lst)
    word2char = {k: chr(i) for i, k in enumerate(alphabet)}
    ref = ''.join([word2char[w] for w in ref_lst])
    hyp = ''.join([word2char[w] for w in hyp_lst])

    e, cur_e = [], []
    for i in range(len(ref) + 1):
        e.append(Err_Stat(0, 0, i, i))
        cur_e.append(Err_Stat(0, 0, i, i))

    # // for other alignments
    for hyp_index in range(1, len(hyp) + 1):
        # cur_e[0] = copy.deepcopy(e[0])
        cur_e[0].ins_num += 1
        cur_e[0].total_cost += 1

        for ref_index in range(1, len(ref) + 1):
            ins_err = e[ref_index].total_cost + 1
            del_err = cur_e[ref_index - 1].total_cost + 1
            sub_err = e[ref_index - 1].total_cost
            if hyp[hyp_index - 1] != ref[ref_index - 1]:
                sub_err += 1
            else:
                e[ref_index - 1].tag.append(('equal', ref_index - 1, ref_index, hyp_index - 1, hyp_index))

            if sub_err < ins_err and sub_err < del_err:
                cur_e[ref_index] = copy.deepcopy(e[ref_index - 1])
                if hyp[hyp_index - 1] != ref[ref_index - 1]:
                    cur_e[ref_index].sub_num += 1  # // substitution error should be increased
                    cur_e[ref_index].tag.append(('replace', ref_index - 1, ref_index, hyp_index - 1, hyp_index))
                cur_e[ref_index].total_cost = sub_err
            elif del_err < ins_err:
                cur_e[ref_index] = copy.deepcopy(cur_e[ref_index - 1])
                cur_e[ref_index].total_cost = del_err
                cur_e[ref_index].del_num += 1  # // deletion number is increased.
                cur_e[ref_index].tag.append(('delete', ref_index - 1, ref_index, hyp_index - 1, hyp_index - 1))
            else:
                cur_e[ref_index] = copy.deepcopy(e[ref_index])
                cur_e[ref_index].total_cost = ins_err
                cur_e[ref_index].ins_num += 1  # // insertion number is increased.
                cur_e[ref_index].tag.append(('insert', ref_index - 1, ref_index - 1, hyp_index - 1, hyp_index))
        e = copy.deepcopy(cur_e)  # // alternate for the next recursion.

    # join similar entries
    ops = [e[-1].tag[0]]
    for el in e[-1].tag[1:]:
        if el[0] == ops[-1][0]:
            ops[-1] = (el[0], ops[-1][1], el[2], ops[-1][3], el[4])
        else:
            ops.append(el)

    return e[-1].total_cost, len(ref_lst), ops


def ops2str(ref_lst, hyp_lst, ops):
    printout = ''
    for tag, i1, i2, j1, j2 in ops:
        if tag == 'replace':
            printout += f'{tag:8s} {" ".join(ref_lst[i1:i2])} <> {" ".join(hyp_lst[j1:j2])} \t{(i1, i2), (j1, j2)}\n'
        else:
            printout += f'{tag:8s} {" ".join(ref_lst[i1:i2]) or "<< " + " ".join(hyp_lst[j1:j2])} \t{(i1, i2), (j1, j2)}\n'
    return printout


def compute_effective_wer(row, func_text_normalization):
    FULL_COST = 10000
    MINIMAL_COST = 1
    try:
        if row['edit_tag'] == 'equal':
            return 0
        if row['edit_tag'] == 'insert':
            return FULL_COST
        if row['edit_tag'] == 'delete':
            return FULL_COST
        if row['edit_tag'] == 'replace':
            normalized_text_ref = func_text_normalization(row['text_reference'])
            normalized_text_hyp = func_text_normalization(row['text_hypothesis'])
            if normalized_text_ref == normalized_text_hyp:
                return MINIMAL_COST
            else:
                return FULL_COST
            pass
    except Exception as e:
        print(row)
        raise (e)

    raise ValueError


def get_edit_df(REF_PATH, HYP_PATH, preprocessing_normalization_func=preprocessing_normalization_func,
                ewer_normalization_func=ewer_normalization_func, limit=None):
    full_compare = []
    for fn, ref_lst, hyp_lst in generate_file_contents(REF_PATH, HYP_PATH, preprocessing_normalization_func, limit):
        dist, length, ops = get_edit_distance_verbosely(ref_lst, hyp_lst)
        full_compare.extend(
            [[fn, ' '.join(ref_lst[x[1]:x[2]]), ' '.join(hyp_lst[x[3]:x[4]]), x[2] - x[1] or x[4] - x[3], *x] for x in
             ops])

    df = pd.DataFrame(full_compare,
                      columns=['filename', 'text_reference', 'text_hypothesis', 'weight', 'edit_tag',
                               'text_reference_beg', 'text_reference_end', 'text_hypothesis_beg',
                               'text_hypothesis_end'])

    import functools
    partial_func = functools.partial(compute_effective_wer, ewer_normalization_func)
    # df['effective_weight'] = df.apply(partial_func, axis=1)
    return df


def get_pivot_table_of_edits(df, groupby=['filename']):
    _ = df.groupby(groupby + ['edit_tag'])['weight'].sum()
    # filenames = [z for z in _.index.levels[0]]
    # Create a pivot table of edit tags by filename
    df_edit_counts = _.reset_index().pivot_table(values='weight', index=groupby, columns='edit_tag').fillna(0)
    df_edit_counts['edits'] = df_edit_counts.get('insert', 0) + df_edit_counts.get('delete', 0) + df_edit_counts.get(
        'replace', 0)
    df_edit_counts['denominator'] = df_edit_counts.get('equal', 0) + df_edit_counts.get('delete',
                                                                                        0) + df_edit_counts.get(
        'replace', 0)
    df_edit_counts['wer'] = df_edit_counts['edits'] / df_edit_counts['denominator'] * 100
    return df_edit_counts


def get_top_errors(df, groupby=['text_reference', 'text_hypothesis']):
    # Get the most common errors
    return df.query('edit_tag!="equal"').reset_index().groupby(groupby)['index'].count().sort_values(
        ascending=False).reset_index()


def install_required_packages():
    output = subprocess.check_output("pip install PyAthena python-Levenshtein", shell=True)
    return output.decode()


def get_calls_metadata(filenames):
    if len(filenames) == 0:
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


def analyze_wer_folders(folder_truth, folder_hypothesis, folder_output,
                        preprocessing_normalization_func=preprocessing_normalization_func,
                        ewer_normalization_func=ewer_normalization_func):
    print('Copying truth files...')
    copy_s3_folder_to_local_folder(folder_truth, './data/truth')
    print('Copying hypothesis files...')
    copy_s3_folder_to_local_folder(folder_hypothesis, './data/hypothesis')

    print('Computing transcription differences...')
    REF_PATH = './data/truth'
    HYP_PATH = './data/hypothesis'
    df = get_edit_df(REF_PATH, HYP_PATH, preprocessing_normalization_func=preprocessing_normalization_func,
                     ewer_normalization_func=ewer_normalization_func, limit=None)
    df['filename'] = df['filename'].astype(int)  # TODO: check if filename is really int !!!
    print(f'Found {df.shape[0]} differences in {df["filename"].nunique()} files.')
    df['common_value'] = 1

    df_edit_counts_edits = get_pivot_table_of_edits(df, groupby=['common_value']).iloc[0]
    print(
        f"Total WER is {df_edit_counts_edits.wer} ({df_edit_counts_edits['equal']} equal, {df_edit_counts_edits['insert']} insert, {df_edit_counts_edits['replace']} replace, {df_edit_counts_edits['delete']} delete)")

    average_wer = get_pivot_table_of_edits(df, groupby=['filename'])['wer'].mean()
    print(f'Average WER per file is {average_wer}')

    filenames = df['filename'].unique()
    df_calls_metadata = get_calls_metadata(filenames)
    df_calls_metadata['gong_link'] = [f'https://app.gong.io/call?id={call_id}' for call_id in
                                      df_calls_metadata['call_id']]
    df_calls_metadata['speaker_count_total'] = df_calls_metadata['speaker_count_in_company'] + \
                                               df_calls_metadata['speaker_count_outside_company'] + df_calls_metadata[
                                                   'speaker_count_company_unknown']
    df_calls_metadata['speaker_count_total'] = df_calls_metadata['speaker_count_total'].fillna(0)

    wer_by_filename_with_metadata = pd.merge(left=get_pivot_table_of_edits(df), right=df_calls_metadata,
                                             left_on='filename', right_on='call_id', how='left')
    save_to_s3(wer_by_filename_with_metadata, s3_filename=folder_output + '/wer_by_filename_with_metadata.csv')

    wer_by_company = wer_by_filename_with_metadata.groupby('company_name')['wer'].mean()
    save_to_s3(wer_by_company, s3_filename=folder_output + '/wer_by_company.csv')

    wer_by_conferencing_provider = wer_by_filename_with_metadata.groupby('conferencing_provider')['wer'].mean()
    save_to_s3(wer_by_conferencing_provider, s3_filename=folder_output + '/wer_by_conferencing_provider.tsv')

    def wer_by_field(x):
        if wer_by_filename_with_metadata[x].nunique() > 0:
            return wer_by_filename_with_metadata.groupby(x)['wer'].describe().sort_values('mean')
        else:
            return None

    print('\n=== WER by company: ===')
    print(wer_by_field('company_name'))

    print('\n=== WER by language: ===')
    print(wer_by_field('language'))

    print('\n=== WER by internal_meeting: ===')
    print(wer_by_field('internal_meeting'))

    print('\n=== WER by direction: ===')
    print(wer_by_field('direction'))

    print('\n=== WER by owner_name: ===')
    print(wer_by_field('owner_name'))

    print('\n=== WER by speaker_count_total: ===')
    print(wer_by_field('speaker_count_total'))

    print('Saving HTML of transcription differences...')
    # Save HTML of edits
    for filename in df['filename'].unique():
        save_transcript_compare_html_to_s3(df[df.filename == filename],
                                           s3_filename=folder_output + f'/transcription_edits_{filename}.html')

    # Top edits
    save_to_s3(get_top_errors(df), s3_filename=folder_output + '/top_edits.tsv')
    # Top errors
    save_to_s3(get_top_errors(df, groupby=['text_reference']), s3_filename=folder_output + '/top_errors.tsv')

    transcription_edits_with_metadata = pd.merge(left=df, right=df_calls_metadata, left_on='filename',
                                                 right_on='call_id')
    save_to_s3(transcription_edits_with_metadata, s3_filename=folder_output + '/transcription_edits_with_metadata.csv')

    return transcription_edits_with_metadata


def run_md_eval(REF_PATH, HYP_PATH, OUT_PATH):
    script = "./der_val.sh {} {} {}".format(REF_PATH, HYP_PATH, OUT_PATH)
    subprocess.check_output(script, shell=True)
    return pd.read_csv("{}/der_comparison.csv".format(OUT_PATH))


def analyze_der_folders(folder_truth, folder_hypothesis, folder_output,
                        preprocessing_normalization_func=preprocessing_normalization_func,
                        ewer_normalization_func=ewer_normalization_func):
    print('Copying truth files...')
    copy_s3_folder_to_local_folder(folder_truth, './data/truth')
    print('Copying hypothesis files...')
    copy_s3_folder_to_local_folder(folder_hypothesis, './data/hypothesis')

    print('Computing diarization differences...')
    REF_PATH = './diar_data/truth'
    HYP_PATH = './diar_data/hypothesis'
    OUT_PATH = './diar_data/output'
    df = run_md_eval(REF_PATH, HYP_PATH, OUT_PATH)
    df['filename'] = df['filename'].astype(int)     # TODO: check if filename is really int !!!
    return df
    # print(f'Found {df.shape[0]} differences in {df["filename"].nunique()} files.')
    # df['common_value'] = 1
    #
    # df_edit_counts_edits = get_pivot_table_of_edits(df, groupby=['common_value']).iloc[0]
    # print(f"Total WER is {df_edit_counts_edits.wer} ({df_edit_counts_edits['equal']} equal, {df_edit_counts_edits['insert']} insert, {df_edit_counts_edits['replace']} replace, {df_edit_counts_edits['delete']} delete)")
    #
    # average_wer = get_pivot_table_of_edits(df, groupby=['filename'])['wer'].mean()
    # print(f'Average WER per file is {average_wer}')
    #
    # filenames = df['filename'].unique()
    # df_calls_metadata = get_calls_metadata(filenames)
    # df_calls_metadata['gong_link'] = [f'https://app.gong.io/call?id={call_id}' for call_id in df_calls_metadata['call_id']]
    # df_calls_metadata['speaker_count_total'] = df_calls_metadata['speaker_count_in_company'] + \
    #     df_calls_metadata['speaker_count_outside_company'] + df_calls_metadata['speaker_count_company_unknown']
    # df_calls_metadata['speaker_count_total'] = df_calls_metadata['speaker_count_total'].fillna(0)
    #
    # wer_by_filename_with_metadata = pd.merge(left=get_pivot_table_of_edits(df), right=df_calls_metadata, left_on='filename', right_on='call_id', how='left')
    # save_to_s3(wer_by_filename_with_metadata, s3_filename=folder_output+'/wer_by_filename_with_metadata.csv')
    #
    # wer_by_company = wer_by_filename_with_metadata.groupby('company_name')['wer'].mean()
    # save_to_s3(wer_by_company, s3_filename=folder_output+'/wer_by_company.csv')
    #
    # wer_by_conferencing_provider = wer_by_filename_with_metadata.groupby('conferencing_provider')['wer'].mean()
    # save_to_s3(wer_by_conferencing_provider, s3_filename=folder_output+'/wer_by_conferencing_provider.tsv')
    #
    # def wer_by_field(x):
    #     if wer_by_filename_with_metadata[x].nunique()>0:
    #         return wer_by_filename_with_metadata.groupby(x)['wer'].describe().sort_values('mean')
    #     else:
    #         return None
    #
    # print('\n=== WER by company: ===')
    # print( wer_by_field('company_name') )
    #
    # print('\n=== WER by language: ===')
    # print(wer_by_field('language'))
    #
    # print('\n=== WER by internal_meeting: ===')
    # print(wer_by_field('internal_meeting'))
    #
    # print('\n=== WER by direction: ===')
    # print(wer_by_field('direction'))
    #
    # print('\n=== WER by owner_name: ===')
    # print(wer_by_field('owner_name'))
    #
    # print('\n=== WER by speaker_count_total: ===')
    # print(wer_by_field('speaker_count_total'))
    #
    # print('Saving HTML of transcription differences...')
    # # Save HTML of edits
    # for filename in df['filename'].unique():
    #     save_transcript_compare_html_to_s3(df[df.filename == filename], s3_filename=folder_output+f'/transcription_edits_{filename}.html')
    #
    # # Top edits
    # save_to_s3(get_top_errors(df), s3_filename=folder_output + '/top_edits.tsv')
    # # Top errors
    # save_to_s3(get_top_errors(df, groupby=['text_reference']), s3_filename=folder_output+'/top_errors.tsv')
    #
    # transcription_edits_with_metadata = pd.merge(left=df, right=df_calls_metadata, left_on='filename', right_on='call_id')
    # save_to_s3(transcription_edits_with_metadata, s3_filename=folder_output+'/transcription_edits_with_metadata.csv')
    #
    # return transcription_edits_with_metadata


def main():
    REF_PATH = r'C:\data\wer\zoominfo_wer\rev\parsed'
    HYP_PATH = r'C:\data\wer\zoominfo_wer\other_tool\parsed'
    folder_output = r'C:\data\wer\zoominfo_wer\rev_other_tool_output'

    REF_PATH = r'C:\data\wer\zoominfo_wer\rev\parsed'
    HYP_PATH = r'C:\data\wer\zoominfo_wer\gong'
    folder_output = r'C:\data\wer\zoominfo_wer\rev_gong_output'

    REF_PATH = r'C:\data\wer\zoominfo_wer\gong'
    HYP_PATH = r'C:\data\wer\zoominfo_wer\other_tool\parsed'
    folder_output = r'C:\data\wer\zoominfo_wer\gong_other_tool_output'

    REF_PATH = r'C:\data\wer\confidence\ref\15-human-transcriptions-normalized'
    HYP_PATH = r'C:\data\wer\confidence\hyp'
    folder_output = r'C:\data\wer\confidence\wer_results'

    # REF_PATH = r'C:\data\wer\german\ref'
    # HYP_PATH = r'C:\data\wer\german\hyp'
    # folder_output = r'C:\data\wer\german\wer_results'

    df = get_edit_df(REF_PATH, HYP_PATH, preprocessing_normalization_func=preprocessing_normalization_func,
                     ewer_normalization_func=ewer_normalization_func, limit=None)
    df['filename'] = df['filename'].astype(str)  # TODO: check if filename is really int !!!
    print(f'Found {df.shape[0]} differences in {df["filename"].nunique()} files.')
    df['common_value'] = 1

    df_edit_counts_edits = get_pivot_table_of_edits(df, groupby=['common_value']).iloc[0]
    print(
        f"Total WER is {df_edit_counts_edits.wer} ({df_edit_counts_edits['equal']} equal, {df_edit_counts_edits['insert']} insert, {df_edit_counts_edits['replace']} replace, {df_edit_counts_edits['delete']} delete)")

    average_wer = get_pivot_table_of_edits(df, groupby=['filename'])['wer'].mean()
    print(f'Average WER per file is {average_wer}')

    print('Saving HTML of transcription differences...')
    # Save HTML of edits
    for filename in df['filename'].unique():
        save_transcript_compare_html_to_s3(df[df.filename == filename],
                                           s3_filename=folder_output + f'/transcription_edits_{filename}.html')

    # Top edits
    save_to_s3(get_top_errors(df), s3_filename=folder_output + '/top_edits.tsv')
    # Top errors
    save_to_s3(get_top_errors(df, groupby=['text_reference']), s3_filename=folder_output + '/top_errors.tsv')

    save_to_s3(df, s3_filename=folder_output + '/transcription_edits.csv')


if __name__ == '__main__':
    main()
    # print( normalize_text("I'm trying to view the views we've imported") )

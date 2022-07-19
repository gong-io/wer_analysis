import json
import os
import sys
import pandas as pd
import boto3


def process_s3(path_in, path_out):
    if not path_in.lower().startswith('s3://') or not path_out.lower().startswith('s3://'):
        print('S3 path should start with s3://')
        return
    s3_session = boto3.Session(profile_name='research')
    s3 = s3_session.resource('s3')

    bucket_in = s3.Bucket(path_in.split('/')[2])
    folder_in = '/'.join(path_in.split('/')[3:])
    bucket_out = s3.Bucket(path_out.split('/')[2])
    folder_out = '/'.join(path_out.split('/')[3:])

    objects = [obj for obj in bucket_in.objects.filter(Prefix=folder_in) if obj.key.lower().endswith('.json')]
    for s3_obj in objects:
        json_data = json.loads(s3_obj.get()['Body'].read().decode('utf-8'))
        new_json = convert_amazon_2_gong(json_data)

        new_fname = s3_obj.key.split('/')[-1].split('.')
        new_fname.insert(-1, 'gong-transcription')
        new_fname = '.'.join(new_fname)
        bucket_out.put_object(Key=folder_out + '/' + new_fname, Body=new_json.encode('utf-8'))


def process_local(indir, outdir):
    files = [fname for fname in os.listdir(indir) if fname.lower().endswith('.json')]
    for fname in files:
        with open(os.path.join(indir,fname)) as fin:
            json_data = json.load(fin)
        res = convert_amazon_2_gong(json_data)

        new_fname = fname.split('.')
        new_fname.insert(-1, 'gong-transcription')
        new_fname = '.'.join(new_fname)
        with open(os.path.join(outdir, new_fname), 'w') as fout:
            fout.write(res)


def convert_amazon_2_gong(json_data):
    monologues = []
    segments = []
    for seg in json_data['results']['speaker_labels']['segments']:
        segments += seg['items']
    df_segments = pd.DataFrame(segments)
    items_gong = []
    prev_end = 0
    for it in json_data['results']['items']:
        if 'start_time' in it:
            curr_itm = {'start': it['start_time'], 'end': it['end_time']}
            prev_end = it['end_time']
        else:
            curr_itm = {'start': prev_end, 'end': prev_end}

        curr_itm['type'] = "WORD" if it['type'] == 'pronunciation' else 'PUNCTUATION'
        curr_itm['text'] = it['alternatives'][0]['content']
        items_gong.append(curr_itm)
    df_items = pd.DataFrame(items_gong)
    df = pd.merge(df_items, df_segments, how='left', left_on=['start', 'end'], right_on=['start_time', 'end_time'])
    df = df.fillna(method='ffill')
    df['monolog_idx'] = (df.speaker_label != df.speaker_label.shift()).astype(int).cumsum()
    monologues = []
    for idx, grp in df.groupby('monolog_idx'):
        speaker = {'name': grp.iloc[0].speaker_label, 'id': abs(hash(grp.iloc[0].speaker_label))}
        terms = grp[['start', 'end', 'text', 'type']].to_dict(orient='records')
        monologues.append(dict(speaker=speaker, terms=terms))
    result_dict = {'schemaVersion': '2.0', 'monologues': monologues}
    return json.dumps(result_dict, indent=2)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Convert Amazon transcript to Gong JSON format:')
        print()
        print('     amazon2gong <input-folder> <output-folder>')
        exit(1)
    path_in = sys.argv[1]
    path_out = sys.argv[2]
    process_s3(path_in, path_out)
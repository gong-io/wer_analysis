import s3fs
import os
import shutil
import pandas as pd
from .html_display import get_css, get_html_of_edits

def save_to_s3(data, s3_filename, format=None):
    s3_filename = s3_filename.replace('//', '/').replace('s3:/', 's3://')

    if format is None:
        format = s3_filename.split('.')[-1]

    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)

    if isinstance(data, pd.DataFrame):
        if format == 'csv':
            data = data.to_csv()
        elif format == 'tsv':
            data = data.to_csv(sep='\t')
        elif format == 'json':
            data = data.T.to_json()
        elif format == 'html':
            data = data.to_html()
        else:
            data = data.to_html()

    if s3_filename[:5]=='s3://':
        s3 = s3fs.S3FileSystem(anon=False)
        with s3.open(s3_filename, 'wb') as f:
            f.write(data.encode())
    else:
        os.makedirs(os.path.dirname(s3_filename), exist_ok=True)
        with open(s3_filename, 'wb') as f:
            f.write(data.encode())

    return data


def save_transcript_compare_html_to_s3(df, s3_filename):
    save_to_s3((get_css() + get_html_of_edits(df)), s3_filename)


def copy_s3_folder_to_local_folder(s3_folder, local_folder):
    shutil.rmtree(local_folder)
    os.makedirs(local_folder, exist_ok=True)
    os.system(f'aws s3 cp {s3_folder} {local_folder} --recursive')



from setuptools import setup, find_packages

setup(name='wer_analysis',
    version='0.1',
    description='Testing installation of Package',
    url='https://github.com/gong-io/wer_analysis/tree/package',
    author='Omri Allouche, Igal Grinis',
    author_email='',
    license='',
    packages=find_packages(include=['wer_analysis', 'wer_analysis.*']),
    install_requires=[
        's3fs',
        'pyathena',
        'nltk',
        'Levenshtein'
    ],
    zip_safe=False)
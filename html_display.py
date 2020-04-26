from IPython.core.display import display, HTML

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
        try:
            if r['edit_tag'] == 'equal':
                txt = r['text_reference']
            elif r['edit_tag'] == 'delete':
                txt = wrap(r['text_reference'], 'delete')
            elif r['edit_tag'] == 'insert':
                txt = wrap(r['text_hypothesis'], 'insert')
            elif r['edit_tag'] == 'replace':
                txt = wrap(r['text_reference'], 'delete')
                txt += wrap(r['text_hypothesis'], 'insert')
            s_joined += wrap(txt, 'block', **r.to_dict()) + ' '

        except Exception as e:
            print(i, r)
            raise(e)

    return s_joined

def display_transcript_compare_html(df):
    display(HTML((get_css() + get_html_of_edits(df))))

def wrap(txt, cls=None, weight='', filename='', edit_tag='', word_start_index='', **kwargs):
    # return txt+' '
    if cls is not None:
        return f'<span class="{cls}" title="{cls}" data-weight="{weight}" data-call-id="{filename}" data-edit="{edit_tag}" data-word-start-index="{word_start_index}" >{txt}</span>'
        # return '<span class="{cls}" title="{cls}">{txt}</span>'.format(cls=cls, txt=txt)
    return txt


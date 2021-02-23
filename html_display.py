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
    span.cap {
        background: blue;
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
                if r['text_reference'].count(" ") == r['text_hypothesis'].count(" "):
                    t_ref_split = r['text_reference'].split(" ")
                    t_hyp_split = r['text_hypothesis'].split(" ")
#                     print(t_ref_split, t_hyp_split)
                    for j in range(len(t_ref_split)):
                        if t_ref_split[j] == t_hyp_split[j]:
                            txt = t_ref_split[j]
                        elif t_ref_split[j].lower() == t_hyp_split[j].lower():
                            txt = wrap(t_ref_split[j], 'cap')
                            txt += wrap(t_hyp_split[j], 'cap')
                        else:
                            txt = wrap(t_ref_split[j], 'delete')
                            txt += wrap(t_hyp_split[j], 'insert')
                        s_joined += wrap(txt, 'block', **r.to_dict()) + ' '
                    continue
                elif r['text_reference'].lower() == r['text_hypothesis'].lower():
                    txt = wrap(r['text_reference'], 'cap')
                    txt += wrap(r['text_hypothesis'], 'cap')
                else:
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


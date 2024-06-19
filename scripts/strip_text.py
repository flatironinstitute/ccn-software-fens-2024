#!/usr/bin/env python3

from mkdocs_gallery import py_source_parser, gen_single
import re
import click
import os
import glob

# regex from mkdocs_gallery for grabbing titles
MD_TITLE_MARKER = re.compile(r"^[ ]?([#]+)[ ]*(.*)[ ]*$")  # One or more starting hash with optional whitespaces before.
RE_3_OR_MORE_NON_ASCII = r"([\W _])\1{3,}"  # 3 or more identical chars
FIRST_NON_MARKER_WITHOUT_HASH = re.compile(rf"^[# ]*(?!{RE_3_OR_MORE_NON_ASCII})[# ]*(.+)", re.MULTILINE)
DIV_START = re.compile(r"<div")
DIV_END = re.compile(r"</div")

def convert(path: str, follow_strip_classes=True):
    """Strip out code.

    This uses mkdocs_gallery's parser to go through one of our notebooks and:

    - Remove all markdown cells except:
      - headers
      - notes (within <div> tags with class=notes)
    - If a header has the class .strip-headers, then we remove all headers
      beneath it (but not itself)
    - If a header has the class .strip-code, then we remove all code blocks
      beneath it, until we hit another header with the *same* level without
      that class
    - Else, code is preserved

    if follow_strip_classes is False, we ignore the .strip-code and
    .strip-headers classes (and thus keep all code and headers)

    """
    conf, source_blocks = py_source_parser.split_code_and_text_blocks(path)
    title_paragraph = gen_single.extract_paragraphs(source_blocks[0][1])[0]
    match = FIRST_NON_MARKER_WITHOUT_HASH.search(title_paragraph)
    title = match.group(2).strip()
    text = []
    most_recent_header_chain = [(1, title)]
    output_blocks = [f"# -*- coding: utf-8 -*-\n\"\"\"{match.group(0)}\"\"\""]
    in_note = False
    for block_type, block_txt, line_no in source_blocks:
        if block_type == 'text':
            block = []
            in_paragraph = False
            for line in block_txt.splitlines():
                header = MD_TITLE_MARKER.search(line)
                if header:
                    header_lvl = len(header.group(1))
                    header_txt = header.group(2).strip()
                    # skip the highest-level title
                    if header_txt == title:
                        continue
                    while header_lvl <= most_recent_header_chain[-1][0]:
                        most_recent_header_chain = most_recent_header_chain[:-1]
                    most_recent_header_chain.append((header_lvl, header_txt))
                    # strip headers beneath this one, not including it
                    if not any(['.strip-headers' in h[1] for h in most_recent_header_chain[:-1]]) or not follow_strip_classes:
                        block.append(header.group(0).split('{')[0].strip())
                else:
                    if any(['.keep-text' in h[1] for h in most_recent_header_chain]):
                        block.append(line)
                    elif any(['.keep-paragraph' in h[1] for h in most_recent_header_chain]):
                        # copy paragraph
                        if not in_paragraph:
                            in_paragraph = line.strip().startswith("<p ") and line.strip().endswith(">")
                        if in_paragraph:
                            block.append(line)
                            in_paragraph = not line.strip().endswith("</p>")
                    if DIV_END.search(line):
                        in_note = False
                        block.append('')
                    elif in_note:
                        block.append(line)
                    elif DIV_START.search(line) and 'notes' in line:
                        in_note = True
                        block.append('')
            if block:
                output_blocks.append('\n# %%\n# ' + '\n# '.join(block))
        if block_type == 'code':
            if not any(['.strip-code' in h[1] for h in most_recent_header_chain]) or not follow_strip_classes:
                output_blocks.append('\n# %%\n' + block_txt)
            elif '.keep-code' in block_txt:
                # remove the line containing "keep-code"
                output_txt = '\n'.join([l for l in block_txt.split('\n') if '.keep-code' not in l])
                output_blocks.append('\n# %%\n' + output_txt)
            # don't duplicate these lines
            elif '# enter code here' not in output_blocks[-1]:
                output_blocks.append('\n# enter code here\n')
    return output_blocks


@click.command()
@click.option("--input_dir", default='docs/examples', help='Directory containing files to convert',
              show_default=True)
@click.option("--output_dir", default='docs/for_users', show_default=True,
              help='Directory to place converted files at. Must exist. (Intended for workshop attendees)')
@click.option("--ignore_classes_dir", default='docs/just_code', show_default=True,
              help='Directory to place converted files (ignoring classes) at. Must exist. These files ignore the .strip-code and .strip-headers classes. (Intended for workshop presenters)')
def main(input_dir: str, output_dir: str, ignore_classes_dir: str):
    for f in glob.glob(os.path.join(input_dir, '*py')):
        blocks = convert(f, True)
        out_fn = os.path.split(f)[-1].replace('.py', '_users.py')
        with open(os.path.join(output_dir, out_fn), 'w') as out_f:
            out_f.write('\n'.join(blocks))
        blocks = convert(f, False)
        out_fn = os.path.split(f)[-1].replace('.py', '_code.py')
        with open(os.path.join(ignore_classes_dir, out_fn), 'w') as out_f:
            out_f.write('\n'.join(blocks))


if __name__ == '__main__':
    main()

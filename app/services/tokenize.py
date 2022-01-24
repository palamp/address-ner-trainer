import re
from typing import List, Tuple

from deeple_preprocessor.tokenize import newmm_tokenize
from pythainlp.tag import pos_tag

re_tag = re.compile("(\[.*?\])")


# ใช้สำหรับกำกับ pos tag เพื่อใช้กับ NER
def postag(listtxt: List[List[str]]) -> List[List[str]]:
    list_word = [word[0] for word in listtxt]
    list_word = pos_tag(list_word, engine="perceptron", corpus="orchid_ud")

    for idx, pos in enumerate(list_word):
        listtxt[idx].insert(1, pos[1])
    return listtxt


# จัดการกับ tag ที่ไม่ได้ tag
def toolner_to_tag(text_list: List[str]) -> List[Tuple[str, str, str]]:
    """
    Parameter:
        text_list: List of text
    Return:
        results: List of Tuples which contain (open_tag, text, close_tag)
    """
    text_list = list(filter(None, re_tag.split(text_list)))
    i = 0
    results = []
    while i < len(text_list):
        if "[" not in text_list[i]:
            results.append(("word", text_list[i], "word"))
            i += 1
        else:
            results.append((text_list[i][1:-1], text_list[i + 1], text_list[i + 2][2:-1]))
            i += 3
    return results


# แปลง text ให้เป็น conll2002
def text2conll2002(text: List[str], pos=True) -> List[List[str]]:
    """
    ใช้แปลงข้อความให้กลายเป็น conll2002
    """
    tag = toolner_to_tag(text)
    conll2002 = []
    for tagopen, text, tagclose in tag:
        word_cut = newmm_tokenize(text)  # ใช้ตัวตัดคำ newmm
        for i, tokenized_text in enumerate(word_cut):
            if tokenized_text == "''" or tokenized_text == '"':
                continue
            elif tagopen != "word" and i:
                conll2002.append([tokenized_text, f"I-{tagopen}"])
            elif tagopen != "word" and not i:
                conll2002.append([tokenized_text, f"B-{tagopen}"])
            else:
                conll2002.append([tokenized_text, "O"])
    if not pos:
        return conll2002
    return postag(conll2002)

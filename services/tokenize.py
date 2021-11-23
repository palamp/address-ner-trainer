import re

from deeple_preprocessor.tokenize import newmm_tokenize
from pythainlp.tag import pos_tag

re_tag = re.compile("(\[.*?\])")


# ใช้สำหรับกำกับ pos tag เพื่อใช้กับ NER
def postag(text):
    listtxt = [i for i in text.split("\n") if i != ""]
    list_word = []
    for data in listtxt:
        list_word.append(data.split("\t")[0])
    list_word = pos_tag(list_word, engine="perceptron", corpus="orchid_ud")

    text = ""
    for data, pos in zip(listtxt, list_word):
        s = data.split("\t")
        text += s[0] + "\t" + pos[1] + "\t" + s[1] + "\n"
    return text


# จัดการกับ tag ที่ไม่ได้ tag
def toolner_to_tag(text_list):
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
def text2conll2002(text, pos=True):
    """
    ใช้แปลงข้อความให้กลายเป็น conll2002
    """
    tag = toolner_to_tag(text)
    conll2002 = ""
    for tagopen, text, tagclose in tag:
        word_cut = newmm_tokenize(text)  # ใช้ตัวตัดคำ newmm
        tmp = ""
        for i, tokenized_text in enumerate(word_cut):
            if tokenized_text == "''" or tokenized_text == '"':
                continue
            elif tagopen != "word" and i:
                tmp += f"{tokenized_text}\tI-{tagopen}\n"
            elif tagopen != "word" and not i:
                tmp += f"{tokenized_text}\tB-{tagopen}\n"
            else:
                tmp += f"{tokenized_text}\tO\n"
        conll2002 += tmp
    if not pos:
        return conll2002
    return postag(conll2002)

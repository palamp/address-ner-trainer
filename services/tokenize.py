import re

from deeple_preprocessor.tokenize import newmm_tokenize
from nltk.tokenize import RegexpTokenizer
from pythainlp.tag import pos_tag

# เตรียมตัวตัด tag ด้วย re
pattern = r"\[(.*?)\](.*?)\[\/(.*?)\]"
# ใช้ nltk.tokenize.RegexpTokenizer เพื่อตัด [TIME]8.00[/TIME] ให้เป็น ('TIME','ไง','TIME')
tokenizer = RegexpTokenizer(pattern)
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
def toolner_to_tag(text):
    text = (
        text.strip()
        .replace("FACILITY", "LOCATION")
        .replace("[AGO]", "")
        .replace("[/AGO]", "")
        .replace("[T]", "")
        .replace("[/T]", "")
    )
    text = re.sub("<[^>]*>", "", text)
    text = re.sub(
        "(\[\/(.*?)\])", "\\1***", text
    )  # .replace('(\[(.*?)\])','***\\1')# text.replace('>','>***') # ตัดการกับพวกไม่มี tag word
    text = re.sub("(\[\w+\])", "***\\1", text)
    text2 = []
    for i in text.split("***"):
        if "[" in i:
            text2.append(i)
        else:
            text2.append("[word]" + i + "[/word]")
    text = "".join(text2)  # re.sub("[word][/word]","","".join(text2))
    return text.replace()


# แปลง text ให้เป็น conll2002
def text2conll2002(text, pos=True):
    """
    ใช้แปลงข้อความให้กลายเป็น conll2002
    """
    text = toolner_to_tag(text)
    text = text.replace("''", '"')
    text = text.replace("’", '"').replace("‘", '"')  # .replace('"',"")
    tag = tokenizer.tokenize(text)
    j = 0
    conll2002 = ""
    for tagopen, text, tagclose in tag:
        word_cut = newmm_tokenize(text)  # ใช้ตัวตัดคำ newmm
        i = 0
        txt5 = ""
        while i < len(word_cut):
            if word_cut[i] == "''" or word_cut[i] == '"':
                pass
            elif i == 0 and tagopen != "word":
                txt5 += word_cut[i]
                txt5 += "\t" + "B-" + tagopen
            elif tagopen != "word":
                txt5 += word_cut[i]
                txt5 += "\t" + "I-" + tagopen
            else:
                txt5 += word_cut[i]
                txt5 += "\t" + "O"
            txt5 += "\n"
            # j+=1
            i += 1
        conll2002 += txt5
    if pos == False:
        return conll2002
    return postag(conll2002)

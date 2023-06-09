import codecs
from pathlib import Path

import dill
from tqdm import tqdm

from app.constants import name_prefixs
from app.services.data_cleaning import (
    is_unique,
    normalize_quote,
    normalize_space,
    remove_emojis,
    remove_prefix,
    replace_and_remove_tags,
)
from app.services.tokenize import text2conll2002


def alldata_list(lists):
    data_all = []
    for data in tqdm(lists):
        try:
            txt = text2conll2002(data, pos=False)
            data_all.append(list(map(tuple, txt)))
            if any([prefix in data for prefix in name_prefixs]):
                data = remove_prefix(data)
                txt = text2conll2002(data, pos=False)
                data_all.append(list(map(tuple, txt)))
        except:
            print(data)
    return data_all


# อ่านข้อมูลจากไฟล์
def read_text(fileopen):
    """
    สำหรับใช้อ่านทั้งหมดทั้งในไฟล์ทีละรรทัดออกมาเป็น list
    """
    with codecs.open(fileopen, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()

    # เอาไม่ซ้ำกัน
    unique = set()
    results = []
    for line in tqdm(lines):
        if is_unique(line, unique):
            text = remove_emojis(line)
            text = normalize_space(text)
            text = normalize_quote(text)
            text = replace_and_remove_tags(text)
            results.append(text)
    return results


if __name__ == "__main__":
    datatofile = []
    for file in Path("dataset").glob("*_train.txt"):
        text_lines = read_text(file)
        datatofile += alldata_list(text_lines)
    with open("dataset/ner.data", "wb") as dill_file:
        dill.dump(datatofile, dill_file)

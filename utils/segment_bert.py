import sys
import tokenization
from tqdm import tqdm
from multiprocessing import Pool

tokenizer = tokenization.FullTokenizer(vocab_file="./vocab.txt", do_lower_case=False)

def split(line):
    line = line.strip()
    origin_line = line
    line = line.replace(" ", "")
    line = tokenization.convert_to_unicode(line)
    if not line:
        return ''
    tokens = tokenizer.tokenize(line)
    return ' '.join(tokens)
    
with Pool(64) as pool:
    for ret in pool.imap(split, tqdm(sys.stdin), chunksize=1024):
        print(ret)
    

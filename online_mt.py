import sys
from datetime import datetime
import subprocess
import os
now = datetime.now()
ts = datetime.timestamp(now)
st = datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
folder_dir = '/usr0/home/glai1/research/XLM/tmp/{}'.format(st)
subprocess.Popen('mkdir -p {}'.format(folder_dir), shell=True).wait()

# get input file

inp_fn = os.path.join(folder_dir, 'input.txt')
with open(inp_fn, 'w') as fout:
  for line in sys.stdin:
    fout.write(line)

# preprocess file

lang = 'de'
XLM_path = '/usr0/home/glai1/research/XLM/'
tool_path = os.path.join(XLM_path, 'tools')
replace_punct=os.path.join(tool_path, "mosesdecoder/scripts/tokenizer/replace-unicode-punctuation.perl")
norm_func = os.path.join(tool_path, "mosesdecoder/scripts/tokenizer/normalize-punctuation.perl")
rem_no_print = os.path.join(tool_path, "mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl")
tokenizer = os.path.join(tool_path, "mosesdecoder/scripts/tokenizer/tokenizer.perl")
detok_path = os.path.join(tool_path, "mosesdecoder/scripts/tokenizer/detokenizer.perl")
fastbpe = os.path.join(tool_path, "fastBPE/fast")
preprocess = os.path.join(XLM_path, 'preprocess.py')
code_path = os.path.join(XLM_path, 'de-en/codes_ende')
vocab_path = os.path.join(XLM_path, 'de-en/vocab_ende')
#bpe_fn = os.path.join(folder_dir, 'bpe.txt')
tokenizer_cmd = 'cat {0} | {1} | {2} -l {3} | {4} | {5} -l {6} -no-escape -threads 8|{7} applybpe_stream {8}'\
                .format(inp_fn, replace_punct, norm_func, lang, rem_no_print, tokenizer, lang, fastbpe, code_path)
#subprocess.Popen(tokenizer_cmd, shell=True).wait()

#translation

out_fn = os.path.join(folder_dir, 'hyp.txt')
translation_cmd = tokenizer_cmd +' | bash translate.sh {}'.format(out_fn)
subprocess.Popen(translation_cmd, shell=True).wait()

#remove bpe

remove_bpe_cmd = "sed -i -r 's/(@@ )|(@@ ?$)//g' {}".format(out_fn)
subprocess.Popen(remove_bpe_cmd, shell=True).wait()

#detokenizer 
detok_cmd = "cat {} | perl {} -q -l {}".format(out_fn, detok_path, lang)
subprocess.Popen(detok_cmd, shell=True).wait()

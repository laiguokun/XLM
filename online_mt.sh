{ st=$(date '+%d-%m-%Y-%H:%M:%S');\
folder_dir=/usr0/home/glai1/research/XLM/tmp/$st;\
lang=de;\
XLM_path=/usr0/home/glai1/research/XLM;\
tool_path=$XLM_path/tools;\
replace_punct=$tool_path/mosesdecoder/scripts/tokenizer/replace-unicode-punctuation.perl;\
norm_func=$tool_path/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl;\
rem_no_print=$tool_path/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl;\
tokenizer=$tool_path/mosesdecoder/scripts/tokenizer/tokenizer.perl;\
detok_path=$tool_path/mosesdecoder/scripts/tokenizer/detokenizer.perl;\
fastbpe=$tool_path/fastBPE/fast;\
preprocess=$XLM_path/preprocess.py;\
code_path=$XLM_path/de-en/codes_ende;\
vocab_path=$XLM_path/de-en/vocab_ende;\
out_fn=${folder_dir}-hyp.txt;\
$replace_punct | $norm_func -l $lang | $rem_no_print | $tokenizer -l $lang -no-escape -threads 8 | $fastbpe applybpe_stream $code_path | $XLM_path/translate.sh $out_fn && \
sed -i -r 's/(@@ )|(@@ ?$)//g' $out_fn && \
cat $out_fn | perl $detok_path -q -l $lang; }
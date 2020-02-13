python /usr0/home/glai1/research/XLM/translate.py --exp_name translate \
--src_lang $2 --tgt_lang en --batch_size 128 \
--model_path /usr0/home/glai1/research/XLM/best-valid_en-ro_mt_bleu.pth --output_path $1

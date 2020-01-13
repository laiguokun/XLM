num=$1
CUDA_VISIBLE_DEVICES=1 python generate_mono.py \
--exp_name generator \
--dump_path ./dumped/ \
--reload_model 'best-valid_en-de_mt_bleu.pth,best-valid_en-de_mt_bleu.pth' \
--data_path ./data/10M_mono/processed/$num/ \
--lgs 'en-de' \
--mt_steps 'de-en' \
--encoder_only false \
--emb_dim 1024  \
--n_layers 6 \
--n_heads 8 \
--dropout 0.1 \
--attention_dropout 0.1 \
--gelu_activation true \
--tokens_per_batch 16000 \
--batch_size 256 \
--bptt 256 \
--eval_only true \
--generate

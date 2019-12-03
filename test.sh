CUDA_VISIBLE_DEVICES=0 python glue-xnli.py \
--exp_name test_trans_mlm15 \
--dump_path ./dumped/trans_mlm15/ \
--model_path mlm_xnli15_1024.pth \
--data_path ./data/processed/ \
--transfer_tasks TRANS \
--optimizer_e adam,lr=0.000025 \
--optimizer_p adam,lr=0.000025 \
--finetune_layers "0:_1" \
--batch_size 4 \
--n_epochs 100 \
--epoch_size 1000 \
--max_len 64 \
--max_vocab 95000 \

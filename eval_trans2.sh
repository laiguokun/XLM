CUDA_VISIBLE_DEVICES=0,1,2,3 python trans2.py \
--exp_name one_tower_hinge \
--dump_path ./dumped/ \
--model_path mlm_xnli15_1024.pth \
--data_path ./data/processed/ \
--transfer_tasks TRANS2 \
--optimizer_e adam,lr=0.000025 \
--optimizer_p adam,lr=0.000025 \
--finetune_layers "0:_1" \
--batch_size 8 \
--n_epochs 10 \
--epoch_size 1000 \
--max_len 64 \
--max_vocab 95000 \
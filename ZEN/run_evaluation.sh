python run_sequence_level_classification.py \
    --task_name ChnSentiCorp \
    --do_eval \
    --do_lower_case \
    --data_dir /media/data/ChnSentiCorp情感分析酒店评论 \
    --bert_model /media/data/competition/CCF/CarForum/ZEN/results/result-seqlevel-2019-11-09-11-05-54/checkpoint-1000/ \
    --max_seq_length 256 \
    --train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 30.0
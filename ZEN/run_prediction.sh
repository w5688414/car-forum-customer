python run_sequence_level_classification.py \
    --task_name carforum \
    --do_predict \
    --do_lower_case \
    --data_dir /media/data/CCF_data/car_forum_data \
    --bert_model /media/data/competition/CCF/CarForum/ZEN/results/result-seqlevel-2019-11-10-18-47-20/checkpoint-8000/ \
    --max_seq_length 256 \
    --train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 30.0
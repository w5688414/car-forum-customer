export BERT_BASE_DIR=/media/data/nlp_models/albert_large_zh
export TEXT_DIR=/media/data/CCF_data/car_forum_data
python3 run_classifier.py   \
            --task_name=carforum \
            --do_train=true  \
            --do_eval=true  \
            --do_predict=true  \
            --data_dir=$TEXT_DIR  \
            --vocab_file=./albert_config/vocab.txt  \
            --bert_config_file=$BERT_BASE_DIR/albert_config_large.json \
            --max_seq_length=256 \
            --train_batch_size=8 \
            --learning_rate=2e-5 \
            --num_train_epochs=30 \
            --output_dir=/media/data/checkpoints/albert_large_carforum \
            --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt

# export BERT_BASE_DIR=/media/data/albert_base_zh
# export TEXT_DIR=/media/data/CCF_data/car_forum_data
# python3 run_classifier.py   \
#             --task_name=carforum \
#             --do_train=true  \
#             --do_eval=true  \
#             --data_dir=$TEXT_DIR  \
#             --vocab_file=./albert_config/vocab.txt  \
#             --bert_config_file=$BERT_BASE_DIR/albert_config_base.json \
#             --max_seq_length=256 \
#             --train_batch_size=16 \
#             --learning_rate=1e-4 \
#             --num_train_epochs=100 \
#             --output_dir=/media/data/albert_base_carforum \
#             --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt


# export BERT_BASE_DIR=/media/data/albert_tiny_zh
# export TEXT_DIR=/media/data/CCF_data/car_forum_data
# python3 run_classifier.py   \
#             --task_name=carforum \
#             --do_train=true  \
#             --do_eval=true  \
#             --data_dir=$TEXT_DIR  \
#             --vocab_file=./albert_config/vocab.txt  \
#             --bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json \
#             --max_seq_length=256 \
#             --train_batch_size=16 \
#             --learning_rate=1e-4 \
#             --num_train_epochs=100 \
#             --output_dir=/media/data/albert_senti_checkpoints \
#             --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt

# nohup python3 run_classifier.py   \
#             --task_name=lcqmc_pair \
#             --do_train=true  \
#             --do_eval=true  \
#             --data_dir=$TEXT_DIR  \
#             --vocab_file=$BERT_BASE_DIR/vocab.txt  \
#             --bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json \
#             --max_seq_length=128 \
#             --train_batch_size=64 \
#             --learning_rate=1e-4 \
#             --num_train_epochs=5 \
#             --output_dir=/media/data/albert_lcqmc_checkpoints \
#             --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt
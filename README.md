# Dialogue Classifier

This is a hierarchical classifier that can be trained to process the data in the dialogue format.

## How to run

1. The data must be in `.jsonl` format, where each line is one dialogue. Each line has the folowing structure:

    ```json
    {
        "id": "000",
        "turns": ["Dialogue turn #0", "Dialogue turn #1", "Dialogue turn #2"], 
        "labels": [0, 0, 1, 2, 0, 0, 2, 3]
    }
    ```

    Train, validation and test sets must be called `train.jsonl`, `validation.jsonl` and `test.jsonl` and place into `data/json/{DATASET_NAME}` directory, where `{DATASET_NAME}` is the name of the dataset.

2. Example command to run the training:

    ```sh
    python run_train.py --dataset_name MY_DATASET --model_name_or_path intfloat/e5-base-v2 \
    --output_dir saved_models/MY_DATASET-e5v2-fixed_steps-smoothl1-optimized --do_train --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 --evaluation_strategy steps --eval_steps 200 --max_steps 2000 --warmup_ratio 0.1 \
    --logging_strategy steps --logging_steps 25 --save_strategy steps --save_steps 200 --seed 0 --data_seed 0 \
    --word_encoder_layers_to_train ".*attention.*$" \
    --phrase_hidden_size 768 --phrase_intermediate_size 1536 --loss_reduction mean \
    --optim adamw_torch --learning_rate 0.00002
    ```

3. Example command to run the prediction on the test set:

    ```sh
    python predict_optimized.py --model_name saved_models/MY_DATASET-e5v2-fixed_steps-smoothl1-optimized/checkpoint-1000 --dataset_name MY_DATASET --output predictions/fixed/MY_DATASET-e5v2-fixed_steps-smoothl1-optimized-1000-test.json --device cuda --data_split test
    ```


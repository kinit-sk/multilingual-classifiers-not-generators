#!/bin/bash
languages=('az' 'cy' 'de' 'en' 'he' 'id' 'ro' 'sl' 'sw' 'te' 'th')
tasks=('sentiment' 'topic' 'intent' 'sarcasm')
models=('llama3' 'gemma' 'qwen')


for lang in ${languages[@]}; do
    for task in ${tasks[@]}; do
        # Fine-Tuning
        python main.py --seed=1337 --repeat=20 --batch_size=4 --epochs=50 --experiment_type=finetuning --model=roberta-large --task=$task --language=$lang --human=0
        python main.py --seed=1337 --repeat=20 --batch_size=4 --epochs=50 --experiment_type=finetuning --model=roberta-large --task=$task --language=$lang --human=1

        # Generator Model
        python main.py --seed=1337 --repeat=1 --batch_size=4 --epochs=10 --experiment_type=llm --model=llama3-large --task=$task --language=$lang --shots=0

        # Prompting + ICL + Instruction-Tuning
        for model in ${models[@]}; do
            python main.py --seed=1337 --repeat=1 --batch_size=4 --epochs=10 --experiment_type=llm --model=$model --task=$task --language=$lang --shots=0
            python main.py --seed=1337 --repeat=20 --batch_size=4 --epochs=10 --experiment_type=llm --model=$model --task=$task --language=$lang --shots=2 --human=0
            python main.py --seed=1337 --repeat=20 --batch_size=4 --epochs=10 --experiment_type=llm --model=$model --task=$task --language=$lang --shots=2 --human=0

            python main.py --seed=1337 --repeat=20 --batch_size=4 --epochs=10 --experiment_type=llm --model=$model --task=$task --language=$lang --shots=2 --human=1
            python main.py --seed=1337 --repeat=20 --batch_size=4 --epochs=10 --experiment_type=llm --model=$model --task=$task --language=$lang --shots=2 --human=1

            python main.py --seed=1337 --repeat=20 --batch_size=4 --epochs=10 --experiment_type=instruction_tuning --model=$model --task=$task --language=$lang --shots=0
        done
    done
done
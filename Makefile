.PHONY: setup train train_fusion eval app fl

setup:
	pip install -r requirements.txt
	python -m ipykernel install --user --name=fakenews-next

train:
	python -m src.train --mode text --datasets ISOT FakeNews LIAR --model_name roberta-base --lora

train_fusion:
	python -m src.train --mode fusion --datasets MFND --model_name roberta-base --lora

eval:
	python -m src.evaluate --datasets ISOT FakeNews LIAR --ensemble soft

app:
	python -m src.app

fl:
	python -m src.federated_train --rounds 5 --clients 9


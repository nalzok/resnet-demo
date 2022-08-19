.PHONY: main

main:
	time pipenv run python3 \
		-m demo.main \
		--batch_size 512 \
		--epochs 8 \
		--learning_rate 1e-2

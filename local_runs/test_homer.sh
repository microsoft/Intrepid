cd ..
export PYTHONPATH=$$PYTHONPATH:src
python3 src/experiments/run_homer.py --env diabcombolock --encoder_training_num_samples 5000 --horizon 5 --debug -1 --noise hadamhardg --save_path ./results  --seed 1234 --name test-homer

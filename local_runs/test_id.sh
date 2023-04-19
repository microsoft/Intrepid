cd ..
export PYTHONPATH=$$PYTHONPATH:src
python3 src/experiments/run_id.py --env temporal_combolock --encoder_training_num_samples 5000 --horizon 5 --exo_dim -1 --noise hadamhardg --classifier_type ff --save_path ./results  --seed 1234 --name test-ppe

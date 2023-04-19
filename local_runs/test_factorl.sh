cd ..
export PYTHONPATH=$$PYTHONPATH:src
python3 src/experiments/run_factorl.py --env slotfactoredmdp --encoder_training_num_samples 5000 --horizon 5 --noise hadamhardg --save_path ./results  --seed 1234 --name test-factorl

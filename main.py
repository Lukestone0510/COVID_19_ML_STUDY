import subprocess


subprocess.run(['python', 'src/data/load_data.py'])


subprocess.run(['python', 'src/data/preprocess.py'])

subprocess.run(['python', 'src/features/build_features.py'])

subprocess.run(['python', 'src/models/train_model.py'])
subprocess.run(['python', 'src/models/evaluate_model.py'])
subprocess.run(['python', 'src/visualization/visualize.py'])

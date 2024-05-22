# UWBWASSA2024SharedTask2
This is the repository for the paper UWB at WASSA-2024 Shared Task 2: Cross-lingual Emotion Detection.

## Requirements
Python 3.10+.
Required packages are listed in `requirements.txt`.


## Data
The official data can be downloaded from [here](https://huggingface.co/datasets/pranaydeeps/EXALT-v1).
If you would like to access additional data (translated data), please contact the authors.

## Running the code
To run the code for the emotion detection subtask, please run the following command:
```
python main_llm.py
```

To run the code for the trigger words detection subtask, please run the following command:
```
python main.py
```

Command arguments can be found in `src/args_utils.py`.

The script saves the best model and also writes results to files at the end.
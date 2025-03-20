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

## Citation
If you find this repository helpful for your research, please cite our paper as follows:
```
@inproceedings{smid-etal-2024-uwb,
    title = "{UWB} at {WASSA}-2024 Shared Task 2: Cross-lingual Emotion Detection",
    author = "{\v{S}}m{\'i}d, Jakub  and
      P{\v{r}}ib{\'a}{\v{n}}, Pavel  and
      Kr{\'a}l, Pavel",
    editor = "De Clercq, Orph{\'e}e  and
      Barriere, Valentin  and
      Barnes, Jeremy  and
      Klinger, Roman  and
      Sedoc, Jo{\~a}o  and
      Tafreshi, Shabnam",
    booktitle = "Proceedings of the 14th Workshop on Computational Approaches to Subjectivity, Sentiment, {\&} Social Media Analysis",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.wassa-1.47/",
    doi = "10.18653/v1/2024.wassa-1.47",
    pages = "483--489",
    abstract = "This paper presents our system built for the WASSA-2024 Cross-lingual Emotion Detection Shared Task. The task consists of two subtasks: first, to assess an emotion label from six possible classes for a given tweet in one of five languages, and second, to predict words triggering the detected emotions in binary and numerical formats. Our proposed approach revolves around fine-tuning quantized large language models, specifically Orca 2, with low-rank adapters (LoRA) and multilingual Transformer-based models, such as XLM-R and mT5. We enhance performance through machine translation for both subtasks and trigger word switching for the second subtask. The system achieves excellent performance, ranking 1st in numerical trigger words detection, 3rd in binary trigger words detection, and 7th in emotion detection."
}
```

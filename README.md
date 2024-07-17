# tensorflow_training

This is a small cheat sheet for training neural networks of various types and tasks.

## Objective of the project:
The goal of this project is to compile all the essential cheat sheets for creating AI for the most popular tasks. This will enable users who are not familiar with AI or Python to use ready-made code to train their own model on their own dataset.

## Tasks:
- [x] Image classification
- [x] Music genre classification
- [x] Text classification
- [ ] Recommendation
- [ ] Generative
- [ ] Translator

## Code changes:
- [x] Transition from local dataset storage to s3 server
> - [x] Image classification
> - [x] Music genre classification
- [ ] Code optimization and debugging

## Example folder location:
### Image classification
```
.
├── Cat [1581]
└── Dog [1560]

2 directories
```

### Music genre classification

```
Wait for it to be added in the next commit
```

### Text classification

```
Wait for it to be added in the next commit
```

## Dataset processing:
### Image classification
```
python fit\pictures\save_in_db.py
```

### Music genre classification

__Only wav is used__
```
# converter from mp3 to wav
python fit\audio\convert_mp3_in_wav.py

python fit\audio\save_in_db.py
```

### Text classification
```
Wait for it to be added in the next commit
```

## Model training
### Image classification
```
python fit\pictures\tensorflow_fit.py
```

### Music genre classification
```
python fit\audio\tensorflow_fit.py
```

### Text classification
```
python fit\text\train_model.py
```

## Using the model
### Image classification
```
python fit\pictures\use_pictures_model.py
```

### Music genre classification
```
python fit\audio\use_audio_model.py
```

### Text classification
```
python fit\text\use_model.py
```


## P.S.

I'm not a professional in AI; I'm just starting to learn. If you have suggestions on how to optimize the model, constructive criticism is always welcome.
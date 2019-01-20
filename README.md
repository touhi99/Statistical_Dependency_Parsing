## Transition-based (Arc Eager) Dependency Parsing

### Author: Touhidul Alam

#### Usage:

```
python3 parser.py [train/dev/test] [en/de] file_path_location
```

Example:

```
python3 parser.py train en /mount/studenten/dependency-parsing/data/english/train/wsj_train.only-projective.conll06 
```

### Dependencies:

* Numpy
* pickle


### Components

* parser.py
	* State - class initialize initial stack, buffer, arcs, left-dependency and right-dependency
	* Instance - class saves the transition label with its corresponding features
	* Parser - class does oracle parsing during training and predicting and making a parser

* reader.py
	* Corpus - class to read sentences from dataset and build them into sentence
	* Sentence - holds the datasets content

* model.py
	* FeatureMapper - takes the unique feature and add it to the feature dictionary. Feature contains baseline feature, and extended features from Nivre (2011)
	* Model - saves the FeatureMapper and weights, implemented with Averaged perceptron


### Output File

```
pred_en.conll06 - English
pred_de.conll06 - German
```

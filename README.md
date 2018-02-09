Knowledge Transfer in Neural Language Models
====

Peter John Hampton, Hui Wang, and Zhiwei Lin

### Requirements

The only requirement is **Python 3.4+**

### Set up

```
$ pyvenv .
$ source bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
$ pip install git+git://github.com/verigak/colors
$ bash ./get_data.sh
```

These experiments _currently_ only work with a tensorflow backend. There are plans in the future to support theano. By default, Keras uses theano as it's backend. You will need to update your global keras config by running `vim ~/.keras/keras.json` (outside PyVenv) and altering the backend config like below:

```
{
    "floatx": "float32",
    "backend": "tensorflow",
    "epsilon": 1e-07,
    "image_dim_ordering": "tf"
}
```

### Exiting PyVenv

```
$ deactivate
```

### Running Experiments

```
$ python model.py
$ ...
```

To turn the gaz on / off, see line 26 of utils/glove_conll2003.py. It is off by default

```
GAZ = False
```

### Data

The **GloVe** Word Embeddings are not included by default. They are too big for Github. To download them run:

```
bash ./get_data.sh
```

The relevant CoNLL Datasets (English) are included by default

 - **CoNLL**
   - CoNLL (2003): http://www.clips.uantwerpen.be/conll2003/ner/
      - eng.train (Training)
      - eng.testa (Validation)
      - eng.testb (Testing)

The CoNLL 2003 datasets are closed, and are only distributed with this repo when a paper is being reviewed (as and when needed). 

### Cite this paper as:

Hampton P.J., Wang H., Lin Z. (2017) Knowledge Transfer in Neural Language Models. In: Bramer M., Petridis M. (eds) Artificial Intelligence XXXIV. SGAI 2017. Lecture Notes in Computer Science, vol 10630. Springer.


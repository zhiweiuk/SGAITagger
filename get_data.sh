#!/bin/bash

date
echo "~~ @pjhampton 2017"
echo "I will start downloading Glove Embeddings (~800mb). Is this ok? [Y/n]"

read answer

if [ "$answer" == "y" ]
then
    echo "Ok. Downloading glove word embeddings"
    wget http://nlp.stanford.edu/data/glove.6B.zip
    mv glove.6B.zip data/glove/glove.6B.zip
    unzip data/glove/glove.6B.zip
fi

echo "Goodbye"
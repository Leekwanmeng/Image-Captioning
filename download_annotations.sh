#!/bin/bash

wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -P ./data/
unzip ./data/captions_train-val2014.zip -d ./data/
rm ./data/captions_train-val2014.zip
#!/bin/sh


echo "* Download the kaggle competitions dataset: otto-recommender-system"
kaggle competitions download -c otto-recommender-system && \
unzip -o otto-recommender-system.zip -d data/otto-recommender-system && \
rm otto-recommender-system.zip

echo "* Download the kaggle dataset: ranchantan/otto-recommender-system-tidy-data"
kaggle datasets download -d ranchantan/otto-recommender-system-tidy-data && \
unzip -o otto-recommender-system-tidy-data.zip -d data/otto-recommender-system-tidy-data && \
rm otto-recommender-system-tidy-data.zip

echo "* Download the kaggle dataset: ranchantan/otto-train-and-test-data-for-local-validation-7days-of-(3/4)weeks"
mkdir data/otto-train-and-test-data-for-local-validation
kaggle datasets download -d ranchantan/otto-train-and-test-data-for-lv-7days-of-4weeks && \
unzip -o otto-train-and-test-data-for-lv-7days-of-4weeks.zip -d data/otto-train-and-test-data-for-local-validation/7days-of-4weeks && \
rm otto-train-and-test-data-for-lv-7days-of-4weeks.zip
kaggle datasets download -d ranchantan/otto-train-and-test-data-for-lv-7days-of-3weeks && \
unzip -o otto-train-and-test-data-for-lv-7days-of-3weeks.zip -d data/otto-train-and-test-data-for-local-validation/7days-of-3weeks && \
rm otto-train-and-test-data-for-lv-7days-of-3weeks.zip

echo "export PYTHONPATH=$PYTHONPATH:$PWD" >> ~/.bashrc
python -m pip install -r requirements.txt
it

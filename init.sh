#!/bin/sh


echo "* Download the kaggle competitions dataset: otto-recommender-system"
kaggle competitions download -c otto-recommender-system && \
unzip otto-recommender-system.zip -d data/otto-recommender-system && \
rm otto-recommender-system.zip

echo "* Download the kaggle dataset: ranchantan/otto-recommender-system-tidy-data"
kaggle datasets download -d ranchantan/otto-recommender-system-tidy-data && \
unzip otto-recommender-system-tidy-data.zip -d data/otto-recommender-system-tidy-data && \
rm otto-recommender-system-tidy-data.zip

echo "* Download the kaggle dataset: ranchantan/otto-recommender-system-tidy-data"
kaggle datasets download -d ranchantan/otto-train-and-test-data-for-local-validation && \
unzip otto-train-and-test-data-for-local-validation.zip -d otto-train-and-test-data-for-local-validation && \
rm otto-train-and-test-data-for-local-validation.zip

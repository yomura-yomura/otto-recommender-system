#!/bin/sh

kaggle datasets download -d cdeotte/otto-validation && \
unzip -o otto-validation -d shaoroon/01_Data/add/local-validation_chris && \
rm otto-validation.zip

kaggle datasets download -d columbia2131/otto-chunk-data-inparquet-format && \
unzip -o otto-chunk-data-inparquet-format -d shaoroon/01_Data/add/convert_parquet && \
rm otto-chunk-data-inparquet-format.zip

kaggle kernels output syaorn13/create-co-matrix-cv-40-bugfix -p shaoroon/01_Data/add/co-matrix-cv-40-bugfix

kaggle kernels output syaorn13/create-co-matrix-cv-100-bugfix -p shaoroon/01_Data/add/co-matrix-cv-100

kaggle kernels output syaorn13/create-co-matrix-lb-40 -p shaoroon/01_Data/add/co-matrix-lb-40

mkdir -p shaoroon/01_Data/word2vec/svd/
echo "Download all under shaoroon/01_Data/word2vec/svd/ from https://drive.google.com/drive/folders/1e2sZg8cLaLOLbsWfE8TxA8X1PeeYk6fl"

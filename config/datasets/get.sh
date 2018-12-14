#!/bin/bash -i

wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip

unzip -o wiki-news-300d-1M.vec.zip -d app/data/files

rm wiki-news-300d-1M.vec.zip
#!/bin/bash
TMPFILE=`mktemp`
TEST_PATH='data/raw/test'

# https://cocodataset.org/#download for images.

# Create directory only if it doesn't exist
mkdir -p $TEST_PATH

wget http://images.cocodataset.org/zips/train2017.zip -O $TMPFILE
unzip -d 'data/raw' $TMPFILE
mv 'data/raw/train2017' 'data/raw/train'
rm $TMPFILE

wget http://images.cocodataset.org/zips/unlabeled2017.zip -O $TMPFILE
unzip -d 'data/raw/' $TMPFILE
mv data/raw/unlabeled2017/* 'data/raw/train'
rm -r 'data/raw/unlabeled2017'
rm $TMPFILE

wget http://images.cocodataset.org/zips/test2017.zip -O $TMPFILE
unzip -d 'data/raw/' $TMPFILE
mv data/raw/test2017/* 'data/raw/train'
rm -r 'data/raw/test2017'
rm $TMPFILE

wget http://images.cocodataset.org/zips/val2017.zip -O $TMPFILE
unzip -d 'data/raw/' $TMPFILE
mv 'data/raw/val2017' 'data/raw/val'
rm $TMPFILE

# TEST SET (BSD100 - SET5 - SET14)
wget https://figshare.com/ndownloader/files/38256840  -O $TMPFILE
unzip -d 'data/raw/test' $TMPFILE
rm $TMPFILE

wget https://figshare.com/ndownloader/files/38256852  -O $TMPFILE
unzip -d 'data/raw/test' $TMPFILE
rm $TMPFILE

wget https://figshare.com/ndownloader/files/38256855  -O $TMPFILE
unzip -d 'data/raw/test' $TMPFILE
rm $TMPFILE

# Remove all Low Resolution images as we will make them from High Resolution images
cd 'data/raw/test/BSD100/image_SRF_4' || exit
rm *LR*
mv * ../
cd '..'
rm -rf 'image_SRF_2'
rm -rf 'image_SRF_3'
rm -rf 'image_SRF_4'

cd '../Set5/image_SRF_4' || exit
rm *LR*
mv * ../
cd '..'
rm -rf 'image_SRF_2'
rm -rf 'image_SRF_3'
rm -rf 'image_SRF_4'

cd '../Set14/image_SRF_4' || exit
rm *LR*
mv * ../
cd '..'
rm -rf 'image_SRF_2'
rm -rf 'image_SRF_3'
rm -rf 'image_SRF_4'

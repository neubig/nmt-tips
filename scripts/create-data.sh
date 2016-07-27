#!/bin/bash

# NOTE: It's recommended that you create a data-prep directory, move to it,
# and then run this script as follows:
#   mkdir ../data-prep
#   cd ../data-prep
#   ../scripts/create-data.sh

WD=`pwd`

# Get and install KyTea
if [[ ! -e usr ]]; then
  git clone http://github.com/neubig/kytea
  cd kytea
  autoreconf -i
  ./configure --prefix=$WD/usr
  make install
  cd $WD
fi

# Download the Tanaka corpus
[[ -e examples.utf.gz ]] || wget ftp://ftp.monash.edu.au/pub/nihongo/examples.utf.gz

# Shuffle the order, and extract the English and Japanese
gunzip -c examples.utf.gz | perl -MList::Util -e 'print List::Util::shuffle <>' | grep '^A: ' > all.both
cat all.both | sed 's/A: \([^	]*\)	.*/\1/g' > all.ja.orig
cat all.both | sed 's/.*	\([^	]*\)#ID.*/\1/g' > all.en.orig

# Split into training and test sets
for l in ja en; do
  sed -n 1,500p all.$l.orig > test.$l.orig
  sed -n 501,1000p all.$l.orig > dev.$l.orig
  tail -n +1000 all.$l.orig > train-big.$l.orig
  head -n 10000 train-big.$l.orig > train.$l.orig
done

# Tokenize English and Japanese respectively
for f in *.en.orig; do 
  ../scripts/tokenizer.pl -a en < $f > ${f/orig/tok}
done
for f in *.ja.orig; do 
  usr/bin/kytea -notags < $f > ${f/orig/tok}
done

# Lowercase
for f in *.tok; do
  perl -nle 'print lc' < $f > ${f/.tok/}
done

# Move the data to the appropriate place
mv {train,dev,test}.{en,ja} ../data
mkdir -p data
mv train-big.{en,ja} data
tar -czf data-big.tar.gz data

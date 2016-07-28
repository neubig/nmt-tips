# Tips on Building Neural Machine Translation Systems

by [Graham Neubig](http://phontron.com) (Nara Institute of Science and Technology/Carnegie Mellon University)

This tutorial will explain some practical tips about how to train a neural machine translation system. It is partly based around examples using the [lamtram](http://github.com/neubig/lamtram) toolkit. Note that this will not cover the theory behind NMT in detail, nor is it a survey meant to cover all the work on neural MT, but it will show you how to use lamtram, and also demonstrate some things that you have to do in order to make a system that actually works well (focusing on ones that are implemented in my toolkit).

This tutorial will assume that you have already installed lamtram (and the [cnn](http://github.com/clab/cnn) backend library that it depends on). Then, use git to pull this tutorial and the corresponding data.

    git clone http://github.com/neubig/nmt-tips

The data in the `data/` directory is Japanese-English data that I have prepared doing some language-specific preprocessing (tokenization, lowercasing, etc.). Enter the `nmt-tips` directory

    cd nmt-tips

and make a link to the directory in which you installed lamtram:

    ln -s /full/path/to/lamtram/directory lamtram

## Machine Translation

Machine translation is a method for translating from a source sequence `F` with words `f_1, ..., f_J` to a target sequence `E` with words `e_1, ..., e_I`. This usually means that we translate between a sentence in a source language (e.g. Japanese) to a sentence in a target language (e.g. English). Machine translation can be used for other applications as well.

In recent years, the most prominent method is Statistical Machine Translation (SMT; Brown et al. (1992)), which builds a probabilistic model of the target sequence given the source sequence `P(E|F)`. This probabilistic model is trained using a large set of training data containing pairs of source and target sequences.

A good resource on machine translation in general, including a number of more traditional (non-Neural) methods is Koehn (2010)'s book "Statistical Machine Translation".

## Neural Machine Translation (NMT) and Encoder-decoder Models

Neural machine translation is a particular variety of SMT that learns the probabilistic model `P(E|F)` using neural networks. I will assume that readers already know basic concepts about neural networks: what a neural network is, particularly what a recurrent neural network is, and how they are trained. If you don't, a good tutorial is Goldberg (2015)'s primer on neural networks for natural language processing.

Encoder-decoder models (Kalchbrenner & Blunsom 2013, Sutskever et al. 2014) are the simplest version of NMT. The idea is relatively simple: we read in the words of a target sentence one-by-one using a recurrent neural network, then predict the words in the target sentence. 

First, we *encode* the source sentence. To do so, we convert the source word into a fixed-length word representation, parameterized by `Φ_wr`:

    wf_j = WORDREP(f_j; Φ_fwr)

Then,  we map this into a hidden state using a recurrent neural network, parameterized by `Φ_frnn`. (We assume `h_0` is a zero vector.)

    h_j = RNN(h_{j-1}, wf_j; Φ_frnn)

Next, we *decode* to generate the target sentence, one word at a time. This is done by initializing the first hidden state of the decoder `g_0` to be equal to the last hidden state of the encoder: `g_0 = h_J`. Next, we generate a word in the output by performing a softmax over the target vocabulary to predict the probability of each word in the output, parameterized by `Φ_esm`:

    pe_i = SOFTMAX(g_{i-1}; Φ_esm)

We then pick the word that has highest probability:

    e'_i = ARGMAX_k(pe_i[k])

We then update the hidden state with this predicted value:

    we'_i = WORDREP(e'_i; Φ_ewr)
    g_i = RNN(g_{i-1}, we'_i; Φ_ernn)

This process is continued until a special "end of sentence" symbol is chosen for `e'_i`. 

## Training NMT Models with Maximum Likelihood

Note that the various elements of the model explained in the previous model have parameters `Φ`. These need to be learned in order to generate high-quality translations. The standard way of training neural networks is by using maximum likelihood. This is done by maximizing the log likelihood of the training data:

    Φ' = ARGMAX_{Φ}( Σ_{E,F} log P(E|F;Φ) )

or equivalently minimizing the negative log likelihood:

    Φ' = ARGMIN_{Φ}( - Σ_{E,F} log P(E|F;Φ) )

The standard way we do this minimization is through *stochastic gradient descent* (SGD), where we calculate the gradient of the negative log probability for a single example `<F,E>`

    ∇_{Φ} -log P(E|F;Φ)

then update the parameters based on an update rule:

    Φ ← UPDATE(Φ, ∇_{Φ} -log P(E|F;Φ))

The most standard update rule simply subtracts the gradient of the negative log likelihood multiplied by a learning rate `γ`

    SGD_UPDATE(Φ, ∇_{Φ} -log P(E|F;Φ), γ) := Φ - γ * ∇_{Φ} -log P(E|F;Φ)

Let's try to do this with lamtram. First make a directory to hold the model:

    mkdir models

then train the model with the following commands:

    lamtram/src/lamtram/lamtram-train \
      --model_type encdec \
      --train_src data/train.ja \
      --train_trg data/train.en \
      --trainer sgd \
      --learning_rate 0.1 \
      --rate_decay 1.0 \
      --epochs 10 \
      --model_out models/encdec.mod

Here, `model_type` indicates that we want to train an encoder-decoder, `train_src` and `train_trg` indicate the source and target training data files. `trainer` specifies that we will use the standard update rule, and `learning_rate` specifies `γ`. `rate_decay` will be explained later. `epochs` is the number of passes over the training data, and `model_out` is the place where the model is written out to.

If training is going well, we will be able to see the following log output:

    Epoch 1 sent 100: ppl=1122.07, unk=0, rate=0.1, time=2.04832 (502.852 w/s)
    Epoch 1 sent 200: ppl=737.81, unk=0, rate=0.1, time=4.08551 (500.305 w/s)
    Epoch 1 sent 300: ppl=570.027, unk=0, rate=0.1, time=6.07408 (501.311 w/s)
    Epoch 1 sent 400: ppl=523.924, unk=0, rate=0.1, time=8.23374 (502.566 w/s)

`ppl` is reporting perplexity on the training set, which is equal to the exponent of the per-word negative log probability:

    PPL(Φ) = exp( -(Σ_{E,F} log P(E|F;Φ))/(Σ_E |E|) )

For this perplexity, lower is better, so if it's decreasing we're learning something.

One thing you'll notice is that training is really slow... There are 10,000 sentences in our small training corpus, but you're probably tired of waiting already. The next section will explain how we speed things up, so let's let it run for a while, and then when you get tired of waiting hit `CTRL+C` to stop training.

## Speeding Up Training

### Minibatches

One powerful tool to speed up training of neural networks is mini-batching. The idea behind minibatching is that instead of calculating the gradient for a single example `<E,F>`

    ∇_{Φ} log P(E|F;Φ)

we calculate the gradients for multiple examples at one time

    ∇_{Φ} Σ_{<E,F> in minibatch} log P(E|F;Φ)

then perform the update of the model's parameters using this aggregated gradient. This has several advantages:

* Gradient updates take time, so if we have N sentences in our minibatch, we can perform N times fewer gradient updates.
* More importantly, by sticking sentences together in a batch, we can share some of the calculations between them. For example, where non-minibatched neural networks might multiply the hidden vector `h_i` by a weight matrix `W`, when we are mini-batching  we can connect `h_i` from different sentences into a single matrix `H` and do a big matrix-matrix multiplication `W * H`, which is much more efficient.
* Also, using mini-batches can make the updates to the parameters more stable, as information from multiple sentences is considered at one time.

On the other hand, large minibatches do have disadvantages:

* If our mini-batch sizes are too big, sometimes we may run out of memory by trying to store too many calculated values in memory at once.
* While the calculation of each sentence becomes faster, because the total number of updates is fewer, sometimes training can be slower than when not using mini-batches.

Anyway, let's try this in lamtram by adding the `--minibatch_size NUM_WORDS` option, where `NUM_WORDS` is the number of words included in each mini-batch. If we set `NUM_WORDS` to be equal to 256, and re-run the previous command, we get the following log: 

    Epoch 1 sent 106: ppl=3970.52, unk=0, rate=0.1, time=0.526336 (2107.02 w/s)
    Epoch 1 sent 201: ppl=2645.1, unk=0, rate=0.1, time=1.00862 (2071.15 w/s)
    Epoch 1 sent 316: ppl=1905.16, unk=0, rate=0.1, time=1.48682 (2068.84 w/s)
    Epoch 1 sent 401: ppl=1574.61, unk=0, rate=0.1, time=1.82187 (2064.91 w/s)

Looking at the `w/s` (words per second) on the right side of the log, we can see that we're processing data 4 times faster than before, nice! Let's still hit `CTRL+C` though, as we'll speed up training even more in the next section.

### Other Update Rules

In addition to the standard `SGD_UPDATE` rule listed above, there are a myriad of additional ways to update the parameters, including "SGD With Momentum", "Adagrad", "Adadelta", "RMSProp", "Adam", and many others. Explaining these in detail is beyond the scope of this tutorial, but it suffices to say that these will more quickly find a good place in parameter space than the standard method above. My current favorite optimization method is "Adam" (Kingma et al. 2012), which can be run by setting `--trainer adam`. We'll also have to change the initial learning rate to `--learning_rate 0.001`, as a learning rate of 0.1 is too big when using Adam.

Try re-running the following command:

    lamtram/src/lamtram/lamtram-train \
      --model_type encdec \
      --train_src data/train.ja \
      --train_trg data/train.en \
      --trainer adam \
      --learning_rate 0.001 \
      --minibatch_size 256 \
      --rate_decay 1.0 \
      --epochs 10 \
      --model_out models/encdec.mod

You'll probably find that the perplexity drops significantly faster than when using the standard SGD update (after the first iteration, I had a perplexity of 287 with standard SGD, and 233 with Adam).

## Attention

### Basic Concept

One of the major advances in NMT has been the introduction of attention (Bahdanau et al. 2015). The basic idea behind attention is that when we want to generate a particular target word `e_i`, that we will want to focus on a particular source word `f_j`, or a couple words. In order to express this, attention calculates a "context vector" `c_i` that 

TODO: Make this explanation more complete.

If you want to try to train an attentional model with lamtram, just change all mentions of `encdec` above to `encatt` (for encoder/attentional), and an attentional model will be trained for you. For example, we can run the following command:

    lamtram/src/lamtram/lamtram-train \
      --model_type encatt \
      --train_src data/train.ja \
      --train_trg data/train.en \
      --trainer adam \
      --learning_rate 0.001 \
      --minibatch_size 256 \
      --rate_decay 1.0 \
      --epochs 10 \
      --model_out models/encatt.mod

If you compare the perplexities between these two methods you may see some difference in the perplexity results after 10 epochs. When I ran these, I got a perplexity of 19 for the encoder-decoder, and a perplexity of 11 for the attentional model, demonstrating that it's a bit easier for the attentional model to model the training corpus correctly.

### Types of Attention (Advanced)

There are several ways to calculate the attention values `a_{i,j}`, such as those investigated by Luong et al. (2015). The following ones are implemented in lamtram, and can be changed using the `--attention_type TYPE` option as noted below.

* Dot Product: Calculate the dot product `a_{i,j} = g_i * transpose(wf_j)` (`--attention_type dot`).
* Bilinear: A bilinear model that puts a parameterized transform `Φ_bilin` between the two vectors `a_{i,j} = g_i * Φ_bilin * transpose(wf_j)` (`--attention_type bilin`).
* Multi-layer Perceptron: Input the two vectors into a multi-layer perceptron with a hidden layer of size `LAYERNODES`, `a_{i,j} = MLP([g_i, wf_j]; Φ_mlp)` (`--attention_type mlp:LAYERNODES`)

In practice, I've found that dot product tends to work pretty well, and because of this it's the default setting in lamtram. However, the multi-layer perceptron also performs well in some cases, so sometimes it's worth trying.

In addition, Luong et al. (2015) introduced a method called "attention feeding," which uses the context vector `c_{i-1}` of the previous state as input to the decoder neural network. This is enabled by default using the `--attention_feed true` option in lamtram, as it seems to help somewhat consistently.

## Testing NMT Systems

Now that we have a couple translation models, and know how good they are doing on the training set (according to perplexity), we will want to test to see how well they will do on data that is not used in training. We do this by measuring accuracy on some test data, conveniently prepared in `data/test.ja` and `data/test.en`.

The first way we can measure accuracy is by calculating the perplexity on this held-out data. This will measure the accuracy of the NMT systems probability estimates `P(E|F)`, and see how well they generalize to new data. We can do this for the encoder-decoder model using the following command:

    lamtram/src/lamtram/lamtram \
      --operation ppl \
      --models_in encdec=models/encdec.mod \
      --src_in data/test.ja \
      < data/test.en

and likewise for the attentional model by replacing `encdec` with `encatt` (twice) in the command above. Note here that we're actually getting perplexities that are much worse for the test set than we did on the training set (I got train/test perplexities of 19/118 for the `encdec` model and 11/112 for the `encatt` model). This is for two reasons: lack of handling of words that don't occur in the training set, and overfitting of the training set. I'll discuss these later.

Next, let's try to actually generate translations of the input using the following command (likewise for the attentional model by swapping `encdec` into `encatt`):

    lamtram/src/lamtram/lamtram \
      --operation gen \
      --models_in encdec=models/encdec.mod \
      --src_in data/test.ja \
      < data/test.en > models/encdec.en

We can then measure the accuracy of this model using a measure called BLEU score (Papineni et al. 2002), which measures the similarity between the translation generated by the model and a reference translation created by a human (`data/test.en`):

    scripts/multi-bleu.pl data/test.en < models/encdec.en

This gave me a BLEU score of 1.76 for `encdec` and 2.17 for `encatt`, which shows that we're getting something. But generally we need a BLEU score of at least 15 or so to have something remotely readable, so we're going to have to try harder.

## Thresholding Unknown Words

The first problem that we have to tackle is that currently the model has no way of handling unknown words that don't exist in the training data. The most common way of fixing this problem is by replacing some of the words in the training data with a special `<unk>` symbol, which will also be used when we observe an unknown word in the testing data. For example, we can replace all words that appear only once in the training corpus by performing the following commands.

    lamtram/script/unk-single.pl < data/train.en > data/train.unk.en
    lamtram/script/unk-single.pl < data/train.ja > data/train.unk.ja

Then we can re-train the attentional model using this new data:

    lamtram/src/lamtram/lamtram-train \
      --model_type encatt \
      --train_src data/train.unk.ja \
      --train_trg data/train.unk.en \
      --trainer adam \
      --learning_rate 0.001 \
      --minibatch_size 256 \
      --rate_decay 1.0 \
      --epochs 10 \
      --model_out models/encatt-unk.mod

This greatly helps our accuracy on the test set: when I measured the perplexity and BLEU score on the test set, this gave me 58 and 3.32 respectively, a bit better than before! It also speeds up training quite a bit because it reduces the size of the vocabulary.

## Using a Development Set

The second problem, over-fitting, can be fixed somewhat by using a development set. The development set is a set of data separate from the training and test sets that we use to measure how well the model is generalizing during training. There are two simple ways to help use this set to prevent overfitting.

### Early Stopping

The first way we can prevent overfitting is regularly measure the accuracy on the development data, and stop training when we get the model that has the best accuracy on this data set. This is called "early stopping" and used in most neural network models. Running this in lamtram is easy, just specify the `dev_src` and `dev_trg` options as follows. (You may also want to increase the number of training epochs to 20 or so to really witness how much the model overfits in later stages of training.)

    lamtram/src/lamtram/lamtram-train \
      --model_type encatt \
      --train_src data/train.unk.ja \
      --train_trg data/train.unk.en \
      --dev_src data/dev.ja \
      --dev_trg data/dev.en \
      --trainer adam \
      --learning_rate 0.001 \
      --minibatch_size 256 \
      --rate_decay 1.0 \
      --epochs 10 \
      --model_out models/encatt-unk-stop.mod

You'll notice that now after every pass over the training data, we're measuring the perplexity on the development set, and the model is written out only when the perplexity is its best value yet. In my case, the development perplexity reaches its peak on the 8th iteration then starts getting worse. In my case, by stopping training on the 8th iteration, the perplexity improved a slight bit to 56, but this didn't make a big difference in BLEU.

### Rate Decay

Another trick that is often used is "rate decay" this reduces the learning rate `γ` every time the perplexity gets worse on the development set. This causes the model to update the parameters a bit more conservatively, which as an effect of controlling overfitting. We can enable rate decay by setting the `rate_decay` parameter to 0.5 (which will halve the learning rate everytime the development perplexity gets worse). This prolongs training a little bit, so let's also set the number of epochs to 15, just to make sure that training has run its course.

    lamtram/src/lamtram/lamtram-train \
      --model_type encatt \
      --train_src data/train.unk.ja \
      --train_trg data/train.unk.en \
      --dev_src data/dev.ja \
      --dev_trg data/dev.en \
      --trainer adam \
      --learning_rate 0.001 \
      --minibatch_size 256 \
      --rate_decay 0.5 \
      --epochs 15 \
      --model_out models/encatt-unk-decay.mod

In my case, the rate was decayed on every epoch after the 8th. This didn't result in an improvement on this particular data set, but in many cases this rate decay can be quite helpful.

## Using an External Lexicon

One way to further help out neural MT systems is to incorporate an external lexicon indicating mappings between words (and their probabilities).

### Training the Lexicon

First, we need to create the lexicon. This can be done using a word alignment tool that finds the correspondences between words in the source and target sentences. Here we will use [fast_align](https://github.com/clab/fast_align) because it is simple to use and fast, but other word alignment tools such as [GIZA++](https://github.com/moses-smt/giza-pp) or [Nile](https://github.com/jasonriesa/nile) might give you better results.

First, let's download and build fast_align:

    git clone https://github.com/clab/fast_align.git
    mkdir fast_align/build
    cd fast_align/build
    cmake ../
    make
    cd ../../

Then, we can run fast_align on the training data to build a lexicon, and use the `convert-cond.pl` script to convert it into a format that lamtram can use.

    mkdir lexicon
    paste data/train.{ja,en} | sed 's/	/ ||| /g' > lexicon/train.jaen
    fast_align/build/fast_align -i lexicon/train.jaen -v -p lexicon/train.jaen.cond > lexicon/train.jaen.align
    lamtram/script/convert-cond.pl < lexicon/train.jaen.cond > lexicon/train.jaen.prob    

### Unknown Word Replacement

The first way we can use this lexicon is by using it to map unknown words in the source language into the target language. Without a lexicon, when an unknown word is predicted in the target, the NMT system will find the word in the source sentence with the highest alignment weight `a_j` and map it into the target as-is. If we have a lexicon, if the source word has a lexicon entry, instead of mapping the word `f_j` as-is, the NMT system will output the word with the highest probability `P_{lex}(e|f_j)` in the lexicon.

This can be done in lamtram by specifying the `map_in` function during decoding:

    lamtram/src/lamtram/lamtram \
      --operation gen \
      --models_in encatt=models/encatt-unk-stop.mod \
      --src_in data/test.ja \
      --map_in lexicon/train.jaen.prob \
      < data/test.en > models/encatt-unk-stop-rep.en

This helped a little bit, raising the BLEU score from 2.58 to 2.63 for my model.

### Improving Translation Probabilities

Another way we can use lexicons is to use them to bootstrap translation probabilities (Arthur et al. 2016).

TODO: formal explanation

This can be done by adding the `attention_lex` options as follows. "alpha" is a parameter to adjust the strength of the lexicon, where smaller indicates that more weight will be put on the lexicon probabilities:

    lamtram/src/lamtram/lamtram-train \
      --model_type encatt \
      --train_src data/train.unk.ja \
      --train_trg data/train.unk.en \
      --dev_src data/dev.ja \
      --dev_trg data/dev.en \
      --attention_lex prior:file=lexicon/train.jaen.prob:alpha=0.0001 \
      --trainer adam \
      --learning_rate 0.001 \
      --minibatch_size 256 \
      --rate_decay 1.0 \
      --epochs 10 \
      --model_out models/encatt-unk-stop-lex.mod

## Changing Network Structure

### Network Size

TODO: Width, depth.

### Type of Recurrence

TODO: Currently using LSTMs. There are also vanilla recurrent networks and GRUs.

## Search

### Beam Search

### Adjusting for Sentence Length

## Ensembling

## More Advanced (but very useful!) Methods

### GPUs (Advanced)

If you have access to a machine with a GPU, this can make training much faster, particularly when training NMT systems with large vocabularies or large hidden layer sizes using minibatches. Running lamtram on GPUs is simple, you just need to compile the cnn library using the `CUDA` backend, then link lamtram to it appropriately.

### Using Subword Units

TODO: BPE, other methods

### Training for Other Evaluation Measures

TODO: minrisk

## Preparing Data

### Data Size

Up until now, you have just been working with the small data set of 10,000 that I've provided. Having about 10,000 sentences makes training relatively fast, but having more data will make accuracy significantly higher. Fortunately, there is a larger data set of about 140,000 sentences called `train-big.ja` and `train-big.en`, which you can download by running the following commands.

  wget http://phontron.com/lamtram/download/data-big.tar.gz
  tar -xzf data-big.tar.gz

Try re-running experiments with this larger data set, and you will see that the accuracy gets significantly higher. In real NMT systems, it's common to use several million sentences (or more!) to achieve usable accuracies. Sometimes in these cases, you'll want to evaluate the accuracy of your system more frequently than when you reach the end of the corpus, so try specifying the `--eval_every NUM_SENTENCES` command, where `NUM_SENTENCES` is the number of sentences after which you'd like to evaluate on the data set.

### Preprocessing

One thing to note is that up until now, we've taken it for granted that our data is split into words and lower-cased. When you build an actual system, this will not be the case, so you'll have to perform these processes yourself. Here, for tokenization we're using:

* English: [Moses](http://) (Koehn et al. 2008)
* Japanese: [KyTea](http://phontron.com/kytea/) (Neubig et al. 2011)

And for lowercasing we're using:

  perl -nle 'print lc'

Make sure that you do tokenization, and potentially lowercasing, before feeding your data into lamtram, or any MT toolkit. You can see an example of how we converted the Tanaka Corpus into the data used for the tutorial by looking at `scripts/create-data.sh` script.

## Final Word

Now, you know a few practical things about making an accurate neural MT system. This is a very fast-moving field, so this guide might be obsolete in a few months from the writing (or even already!) but hopefully this helped you learn the basics to get started, start reading papers, and come up with your own methods/applications.

## References

* Brown et al. 1992
* Koehn et al. 2008
* Koehn 2010
* Kingma et al. 2012
* Neubig et al. 2011
* Kalchbrenner & Blunsom 2013
* Sutskever et al. 2014
* Bahdanau et al. 2015
* Luong et al. 2015
* Goldberg 2015
* Arthur et al. 2016

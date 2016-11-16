# Tips on Building Neural Machine Translation Systems

by [Graham Neubig](http://phontron.com) (Nara Institute of Science and Technology/Carnegie Mellon University)

This tutorial will explain some practical tips about how to train a neural machine translation system. It is partly based around examples using the [lamtram](http://github.com/neubig/lamtram) toolkit. Note that this will not cover the theory behind NMT in detail, nor is it a survey meant to cover all the work on neural MT, but it will show you how to use lamtram, and also demonstrate some things that you have to do in order to make a system that actually works well (focusing on ones that are implemented in my toolkit).

This tutorial will assume that you have already installed lamtram (and the [DyNet](http://github.com/clab/dynet) backend library that it depends on) on Linux or Mac. Then, use git to pull this tutorial and the corresponding data.

    git clone http://github.com/neubig/nmt-tips

The data in the `data/` directory is Japanese-English data that I have prepared doing some language-specific preprocessing (tokenization, lowercasing, etc.). Enter the `nmt-tips` directory

    cd nmt-tips

and make a link to the directory in which you installed lamtram:

    ln -s /full/path/to/lamtram/directory lamtram

## Machine Translation

Machine translation is a method for translating from a source sequence `F` with words `f_1, ..., f_J` to a target sequence `E` with words `e_1, ..., e_I`. This usually means that we translate between a sentence in a source language (e.g. Japanese) to a sentence in a target language (e.g. English). Machine translation can be used for other applications as well.

In recent years, the most prominent method is Statistical Machine Translation (SMT; Brown et al. (1993)), which builds a probabilistic model of the target sequence given the source sequence `P(E|F)`. This probabilistic model is trained using a large set of training data containing pairs of source and target sequences.

A good resource on machine translation in general, including a number of more traditional (non-Neural) methods is Koehn (2009)'s book "Statistical Machine Translation".

## Neural Machine Translation (NMT) and Encoder-decoder Models

Neural machine translation is a particular variety of SMT that learns the probabilistic model `P(E|F)` using neural networks. I will assume that readers already know basic concepts about neural networks: what a neural network is, particularly what a recurrent neural network is, and how they are trained. If you don't, a good tutorial is Goldberg (2015)'s primer on neural networks for natural language processing.

Encoder-decoder models (Kalchbrenner & Blunsom 2013, Sutskever et al. 2014) are the simplest version of NMT. The idea is relatively simple: we read in the words of a target sentence one-by-one using a recurrent neural network, then predict the words in the target sentence. 

First, we *encode* the source sentence. To do so, we convert the source word into a fixed-length word representation, parameterized by `Φ_wr`:

    wf_j = WORDREP(f_j; Φ_fwr)

Then,  we map this into a hidden state using a recurrent neural network, parameterized by `Φ_frnn`. We assume `h_0` is a zero vector.

    h_j = RNN(h_{j-1}, wf_j; Φ_frnn)

It is also common to generate `h_j` using bidirectional neural networks, where we run one forward RNN that reads from left-to-right, and another backward RNN that reads from right to left, then concatenate the representations for each word. This is the default setting in lamtram (specified by `--encoder_types "for|rev"`).

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

In addition to the standard `SGD_UPDATE` rule listed above, there are a myriad of additional ways to update the parameters, including "SGD With Momentum", "Adagrad", "Adadelta", "RMSProp", "Adam", and many others. Explaining these in detail is beyond the scope of this tutorial, but it suffices to say that these will more quickly find a good place in parameter space than the standard method above. My current favorite optimization method is "Adam" (Kingma et al. 2014), which can be run by setting `--trainer adam`. We'll also have to change the initial learning rate to `--learning_rate 0.001`, as a learning rate of 0.1 is too big when using Adam.

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

You'll probably find that the perplexity drops significantly faster than when using the standard SGD update (after the first epoch, I had a perplexity of 287 with standard SGD, and 233 with Adam).

### GPUs (Advanced)

If you have access to a machine with a GPU, this can make training much faster, particularly when training NMT systems with large vocabularies or large hidden layer sizes using minibatches. Running lamtram on GPUs is simple, you just need to compile the DyNet library using the `CUDA` backend, then link lamtram to it appropriately. However, in our case here we are using a small network and small training set, so training on CPU is sufficient for now.

## Attention

### Basic Concept

One of the major advances in NMT has been the introduction of attention (Bahdanau et al. 2015). The basic idea behind attention is that when we want to generate a particular target word `e_i`, that we will want to focus on a particular source word `f_j`, or a couple words. In order to express this, attention calculates a "context vector" `c_i` that is used as input to the softmax in addition to the decoder state:
 
    pe_i = SOFTMAX([g_{i-1}, c_i]; Φ_esm)

This context vector is defined as the sum of the input sequence vectors `h_j`, weighted by an attention vector `a_i` as follows:

    c_i = Σ_j a_{i,j} h_j

There are a number of ways to calculate the attention vector `a_i` (described in detail below), but all follow a basic pattern of calculating an attention score `α_{i,j}` for every word that is a function of `g_i` and `h_j`:

    α_{i,j} = ATTENTION(g_i, h_j; Φ_attn)

and then use a softmax function to convert score vector `α_i` into an attention vector `a_i` that adds to one.

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

There are several ways to calculate the attention scores `α_{i,j}`, such as those investigated by Luong et al. (2015). The following ones are implemented in lamtram, and can be changed using the `--attention_type TYPE` option as noted below.

* Dot Product: Calculate the dot product `α_{i,j} = g_i * transpose(wf_j)` (`--attention_type dot`).
* Bilinear: A bilinear model that puts a parameterized transform `Φ_bilin` between the two vectors `α_{i,j} = g_i * Φ_bilin * transpose(wf_j)` (`--attention_type bilin`).
* Multi-layer Perceptron: Input the two vectors into a multi-layer perceptron with a hidden layer of size `LAYERNODES`, `α_{i,j} = MLP([g_i, wf_j]; Φ_mlp)` (`--attention_type mlp:LAYERNODES`)

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
      > models/encdec.en

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
    paste data/train.{ja,en} | sed $'s/\t/ ||| /g' > lexicon/train.jaen
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
      > models/encatt-unk-stop-rep.en

This helped a little bit, raising the BLEU score from 2.58 to 2.63 for my model.

### Improving Translation Probabilities

Another way we can use lexicons is to use them to bootstrap translation probabilities (Arthur et al. 2016). This works by calculating a lexicon probability based on the attention weights `a_j`

    P_{lex}(e_i | F, a) = Σ_j a_j P(e_i | f_j)

This is then added as an additional information source when calculating the softmax probabilities over the output. The advantage of this method is that the lexicon is fast to train, and also contains information about what words can be translated into others in an efficient manner, making it easier for the MT system to learn correct translations, particularly of rare words.

This method can be applied by adding the `attention_lex` options as follows. "alpha" is a parameter to adjust the strength of the lexicon, where smaller indicates that more weight will be put on the lexicon probabilities:

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

In my running, this improves our perplexity from 57 to 37, and BLEU score from 2.48 to 8.83, nice!

## Search

### Beam Search

In the initial explanation of NMT, I explained that translations are generated by selecting the next word in the target sentence that maximizes the probability `pe_i`. However, while this gives us a locally optimal decision about the next word `e'_i`, this is a greedy search method that won't necessarily give us the sentence `E'` that maximizes the translation probability `P(E|F)`.

To improve search (and hopefully translation accuracy), we can use "beam search," which instead of considering the one best next word, considers the `k` best hypotheses at every time step `i`. If `k` is bigger, search will be more accurate but slower. `k` can be set with the `--beam` option during decoding, so let's try this here with our best model so far:

    lamtram/src/lamtram/lamtram \
      --operation gen \
      --models_in encatt=models/encatt-unk-stop-lex.mod \
      --src_in data/test.ja \
      --map_in lexicon/train.jaen.prob \
      --beam BEAM \
      > models/encatt-unk-stop-lex-beamBEAM.en

where we replace the two instances of `BEAM` above with values such as 1, 2, 3, 5.

Looking at the results

    BEAM=1:  BLEU = 8.83, 42.2/14.3/5.5/2.1 (BP=0.973, ratio=0.973, hyp_len=4564, ref_len=4690)
    BEAM=2:  BLEU = 9.23, 45.4/16.2/6.4/2.5 (BP=0.887, ratio=0.893, hyp_len=4186, ref_len=4690)
    BEAM=3:  BLEU = 9.66, 49.4/18.0/7.5/3.1 (BP=0.805, ratio=0.822, hyp_len=3855, ref_len=4690)
    BEAM=5:  BLEU = 9.66, 50.7/18.7/8.0/3.4 (BP=0.765, ratio=0.788, hyp_len=3698, ref_len=4690)
    BEAM=10: BLEU = 9.73, 51.7/19.2/8.5/3.8 (BP=0.726, ratio=0.758, hyp_len=3553, ref_len=4690)

we can see that by increasing the beam size, we can get a decent improvement in BLEU.

### Adjusting for Sentence Length

However, there is also something concerning about the previous result. "ratio=" is the ratio of "output length"/"reference length" and if this is less than 1, our sentences are too short. We can see that as we increase the beam size, our sentences are getting to be much shorter that the reference. The reason for this is that as sentences get longer, their probability tends to get lower, and when we increase the beam size we become more effective at finding these shorter sentences.

There are a number of ways to fix this problem, but the easiest is adding a "word penalty" `wp` which multiplies the probability of the sentence by the constant "e^{wp}" every time an additional word is added. This is equivalent to setting a prior probability on the length of the sentence that follows an exponential distribution. `wp` can be set using the `--word_pen` option of lamtram, so let's try setting a few different values and measure the BLEU score for beam width of 10:

    wp=0.0: BLEU = 9.73, 51.7/19.2/8.5/3.8 (BP=0.726, ratio=0.758, hyp_len=3553, ref_len=4690)
    wp=0.5: BLEU = 9.90, 50.5/18.6/8.0/3.5 (BP=0.775, ratio=0.797, hyp_len=3736, ref_len=4690)
    wp=1.0: BLEU = 10.00, 48.3/17.9/7.5/3.0 (BP=0.850, ratio=0.860, hyp_len=4033, ref_len=4690)
    wp=1.5: BLEU = 9.95, 44.1/16.1/6.5/2.5 (BP=0.963, ratio=0.963, hyp_len=4518, ref_len=4690)

We can see that as we increase the word penalty, this gives us more reasonably-lengthed hypotheses, which also improves the BLEU a little bit.

## Changing Network Structure

One thing that we have not considered so far is the size of the network that we're training. Currently the default for lamtram is that all recurrent networks have 100 hidden nodes (or when using forward/backward encoders, the encoders will be 50 and decoder will be 100). In addition, we're using only a single hidden layer, while many recent systems use deeper networks with 2-4 hidden layers. These can be changed using the `--layers` option of lamtram, which defaults to `lstm:100:1`, where the first option is using LSTM networks (which tend to work pretty well), the second option is the width, and third option is the depth. Let's try to train a wider network by setting `--layers lstm:200:1`.

One thing to note is that the DyNet toolkit has a default limit of using 512MB of memory, but once we start using larger networks this might not be sufficient. So we'll also increase the amount of memory to 1024MB by adding the `--dynet_mem 1024` parameter.

    lamtram/src/lamtram/lamtram-train \
      --dynet_mem 1024 \
      --model_type encatt \
      --train_src data/train.unk.ja \
      --train_trg data/train.unk.en \
      --dev_src data/dev.ja \
      --dev_trg data/dev.en \
      --attention_lex prior:file=lexicon/train.jaen.prob:alpha=0.0001 \
      --layers lstm:200:1 \
      --trainer adam \
      --learning_rate 0.001 \
      --minibatch_size 256 \
      --rate_decay 1.0 \
      --epochs 10 \
      --model_out models/encatt-unk-stop-lex-w200.mod

Note that this makes training significantly slower, because we need to do twice as many calculations in many of our matrix multiplications. Testing this model, the model with 200 nodes reduces perplexity from 37 to 33, and improves BLEU from 10.00 to 10.21. When using larger training data we'll get even bigger improvements by making the network bigger.

## Ensembling

One final technique that is useful for improving final results is "ensembling," or combining multiple models together. The way this works is that if we have two probability distributions `pe_i^{(1)}` and `pe_i^{(2)}` from multiple models, we can calculate the next probability by linearly interpolating them together:

    pe_i = (pe_i^{(1)} + pe_i^{(2)}) / 2

or log-linearly interpolating them together:

    pe_i = exp( (log(pe_i^{(1)}) + log(pe_i^{(2)})) / 2 )

Performing ensembling at test time in lamtram is simple: in `--models_in`, we simply add two different model options separated by a pipe, as follows. The default is linear interpolation, but you can also try log-linear interpolation by setting `--ensemble_op logsum`. Let's try ensembling our 100-node and 200-node models to measure perplexity:

    lamtram/src/lamtram/lamtram \
      --operation ppl \
      --models_in "encatt=models/encatt-unk-stop-lex.mod|encatt=models/encatt-unk-stop-lex-w200.mod" \
      --src_in data/test.ja \
      < data/test.en

This reduced the perplexity from 36/33 to 30 for the ensembled model, and resulted in a BLEU score of 10.99. Of course, we can probably improve this by ensembling even more models together. It's actually OK to just train several models of the same structure with different random seeds (if you set the `--seed` parameter of lamtram you can set a different seed, or by default a different one will be chosen randomly every time).

## Final Output

Because we're basically done, I'll also list up a few examples from the start of the test corpus, where the first line is the input, the second line is the correct translation, and the third line is generated translation.

    君 は １ 日 で それ が でき ま す か 。
    can you do it in one day ?
    you can do it on a day ?
    
    皮肉 な 笑い を 浮かべ て 彼 は 私 を 見つめ た 。
    he stared at me with a satirical smile .
    he stared at the irony of irony .
    
    私 たち の 出発 の 時間 が 差し迫 っ て い る 。
    it &apos;s time to leave .
    our start of our start is we .
    
    あなた は 午後 何 を し た い で す か 。
    what do you want to do in the afternoon ?
    what did you do for you this afternoon ?

Not great, but actually pretty good considering that we only have 10,000 sentences of training data, and that Japanese-English is a pretty difficult language pair to translate!

## More Advanced (but very useful!) Methods

The following are a few extra methods that can be pretty useful in some cases, but I won't be testing here:

### Regularization

As mentioned before, when dealing with small data we need to worry about overfitting, and some ways to fix this are ealy stopping and learning rate decay. In addition, we can also reduce the damage of overfitting by adding some variety of regularization.

One common way of regularizing neural networks is "dropout" (Srivastava et al. 2014) which consists of randomly disabling a set fraction of the units in the input network. This dropout rate can be set with the `--dropout RATE` option. Usually we use a rate of 0.5, which has nice theoretical properties. I tried this on this data set, and it reduced perplexity from 33 to 30 for the 200 node model, but didn't have a large effect on BLEU scores.

Another way to do this is using L2 regularization, which puts a penalty on the L2 norm of the parameter vectors in the model. This can be applied by adding `--dynet_l2 RATE` to the beginning of the option list. I've personally had little luck with getting this to work for neural networks, but it might be worth trying.

### Using Subword Units

One problem with neural network models is that as the vocabulary gets larger, training time increases, so it's often necessary to replace many of the words in the vocabulary with `<unk>` to ensure that training times remain reasonable. There are a number of ways that have been proposed to handle the problem of large vocabularies. One simple way to do so without sacrificing accuracy on low-frequency words (too much) is by splitting rare words into subword units. A method to do so by Sennrich et al. (2016) discovers good subword units using a method called "byte pair encoding", and is implemented in the [subword-nmt](http://github.com/rsennrich/subword-nmt) package. You can use this as an additional pre-processing/post-processing step before learning and using a model with lamtram.

### Training for Other Evaluation Measures

Finally, you may have noticed throughout this tutorial that we are training models to maximize the likelihood, but evaluating our models using BLEU score. There are a number of methods to resolve this mismatch between the training and testing criteria by directly optimizing NMT systems to improve translation accuracy. In lamtram, a method by Shen et al. (2016) can be used to optimize NMT systems for expected BLEU score (or in other words, minimize the risk). In particular, I've found that this does a good job of at least ensuring that the NMT system generates output that is of the appropriate length.

There are a number settings that should be changed when using the method:

* `--learning_criterion minrisk`: This will enable minimum-risk based training.
* `--model_in FILE`: Because this method is slow to train, it's better to first initialize the model using standard maximimum likelihood training, then fine-tune the model with BLEU-based training. This method can be used to read in an already-trained model.
* `--minrisk_num_samples NUM`: This method works by generating samples from the model, then evaluating these generated samples. Increasing NUM improves the stability of the training, but also reduces the training efficiency. A value 20-100 should be reasonable.
* `--minrisk_scaling`, `--minrisk_dedup`: Parameters of the algorithm including the scaling factors for probabilities, and whether to include the correct answer in the samples or not.
* `--trainer sgd --learning_rate 0.05`: I've found that using more advanced optimizers like Adam actually reduces stability in training, so using vanilla SGD might be a safer choice. Slightly lowering the learning rate is also sometimes necessary.
* `--eval_every 1000`: Training is a bit slower than standard NMT training, so we can evaluate more frequently than when we finish the whole corpus.

The final command will look like this:

    lamtram/src/lamtram/lamtram-train \
      --dynet_mem 1024 \
      --model_type encatt \
      --train_src data/train.unk.ja \
      --train_trg data/train.unk.en \
      --dev_src data/dev.ja \
      --dev_trg data/dev.en \
      --trainer sgd \
      --learning_criterion minrisk \
      --learning_rate 0.05 \
      --minrisk_num_samples 20 \
      --minrisk_scaling 0.005 \
      --minrisk_include_ref true \
      --rate_decay 1.0 \
      --epochs 10 \
      --eval_every 1000 \
      --model_in models/encatt-unk-stop-lex-w200.mod \
      --model_out models/encatt-unk-stop-lex-w200-minrisk.mod

## Preparing Data

### Data Size

Up until now, you have just been working with the small data set of 10,000 that I've provided. Having about 10,000 sentences makes training relatively fast, but having more data will make accuracy significantly higher. Fortunately, there is a larger data set of about 140,000 sentences called `train-big.ja` and `train-big.en`, which you can download by running the following commands.

    wget http://phontron.com/lamtram/download/data-big.tar.gz
    tar -xzf data-big.tar.gz

Try re-running experiments with this larger data set, and you will see that the accuracy gets significantly higher. In real NMT systems, it's common to use several million sentences (or more!) to achieve usable accuracies. Sometimes in these cases, you'll want to evaluate the accuracy of your system more frequently than when you reach the end of the corpus, so try specifying the `--eval_every NUM_SENTENCES` command, where `NUM_SENTENCES` is the number of sentences after which you'd like to evaluate on the dev set. Also, it's highly recommended that you use a GPU for training when scaling to larger data and networks.

### Preprocessing

Also note that up until now, we've taken it for granted that our data is split into words and lower-cased. When you build an actual system, this will not be the case, so you'll have to perform these processes yourself. Here, for tokenization we're using:

* English: [Moses](http://statmt.org/moses) (Koehn et al. 2007)
* Japanese: [KyTea](http://phontron.com/kytea/) (Neubig et al. 2011)

And for lowercasing we're using:

    perl -nle 'print lc'

Make sure that you do tokenization, and potentially lowercasing, before feeding your data into lamtram, or any MT toolkit. You can see an example of how we converted the Tanaka Corpus into the data used for the tutorial by looking at `scripts/create-data.sh` script.

## Final Word

Now, you know a few practical things about making an accurate neural MT system. Using the methods described here, we were able to improve a system trained on only 10,000 sentences from 1.83 BLEU to 10.99 BLEU. Switching over to larger data should result in much larger increases, and may even result in readable translations.

This is a very fast-moving field, so this guide might be obsolete in a few months from the writing (or even already!) but hopefully this helped you learn the basics to get started, start reading papers, and come up with your own methods/applications.

## References

* Philip Arthur, Graham Neubig, Satoshi Nakamura. Incorporating Discrete Translation Lexicons in Neural Machine Translation. EMNLP, 2016
* Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio. Neural Machine Translation by Jointly Learning to Align and Translate. ICLR, 2015.
* Peter F. Brown, Vincent J. Della Pietra, Stephen A. Della Pietra, Robert L. Mercer. The mathematics of statistical machine translation: Parameter estimation. Computational Linguistics, 1993.
* Yoav Goldberg. A primer on neural network models for natural language processing. ArXiv, 2015.
* Nal Kalchbrenner, Phil Blunsom. Recurrent Continuous Translation Models. EMNLP, 2013.
* Diederik Kingma, Jimmy Ba. Adam: A method for stochastic optimization. ArXiv, 2014.
* Philipp Koehn et al. Moses: Open source toolkit for statistical machine translation. ACL, 2007.
* Philipp Koehn. Statistical machine translation. Cambridge University Press, 2009.
* Minh-Thang Luong, Hieu Pham, Christopher D. Manning. Effective approaches to attention-based neural machine translation. EMNLP, 2015.
* Graham Neubig, Yosuke Nakata, Shinsuke Mori. Pointwise prediction for robust, adaptable Japanese morphological analysis. ACL, 2011.
* Rico Sennrich, Barry Haddow, Alexandra Birch. Neural machine translation of rare words with subword units. ACL, 2016.
* Nitish Srivastava, Geoffrey E. Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan R. Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. JMLR, 2014.
* Shiqi Shen, Yong Cheng, Zhongjun He, Wei He, Hua Wu, Maosong Sun, Yang Liu. Minimum risk training for neural machine translation. ACL, 2016.
* Ilya Sutskever, Oriol Vinyals, Quoc V. Le. Sequence to sequence learning with neural networks. NIPS, 2014.

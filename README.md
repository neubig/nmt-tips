# Tips on Building Neural Machine Translation Systems

by [Graham Neubig](http://phontron.com) (Nara Institute of Science and Technology/Carnegie Mellon University)

This tutorial will explain some pratical tips about how to train a neural machine translation system. It is partly based around examples using the [lamtram](http://github.com/neubig/lamtram) toolkit. Note that this will not cover the theory behind NMT in detail, nor is it a survey meant to cover all the work on neural MT, but it will show you how to use lamtram, and also demonstrate some things that you have to do in order to make a system that actually works well (focusing on ones that are implemented in my toolkit).

This tutorial will assume that you have installed lamtram already somewhere. Then, use git to pull this tutorial and the corresponding data.

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

Note that the various elements of the model explained in the previous model have parameters `Φ`. These need to be learned in order to generate high-quality translations. The standard way of training neural networks is by using maximum likelihood. This is done by maximizing the (log) likelihood of the training data:

	Φ' = ARGMAX_{Φ}( Σ_{E,F} log P(E|F;Φ) )

The standard way we maximize this likelihood is through *stochastic gradient descent*, where we calculate the gradient of the log probability for a single example `<F,E>`

	∇_{Φ} log P(E|F;Φ)

then update the parameters based on an update rule:

	Φ ← UPDATE(Φ, ∇_{Φ} log P(E|F;Φ))

Let's try to do this with lamtram. First make a directory to hold the model:

  mkdir models

then train the model with the following commands:

	lamtram/src/lamtram/lamtram \
	  --model_type encdec \
	  --train_src data/train.ja \
	  --train_trg data/train.en \
	  --trainer sgd \
	  --learning_rate 0.1 \
	  --rate_decay 1.0 \
	  --epochs 10 \
	  --model_out models/encdec.mod

Here, `model_type` indicates that we want to train an encoder-decoder, `train_src` and `train_trg` indicate the source and target training data files. `trainer`, `learning_rate`, and `rate_decay` specify parameters of the `UPDATE()` function, which I'll explain later. `epochs` is the number of passes over the training data, and `model_out` is the place where the model is written out to.

If training is going well, we will be able to see the following log output:

	TODO

`ppl` is reporting perplexity on the training set, which is equal to the exponent of the per-word negative log probability:

	PPL(Φ) = exp( -(Σ_{E,F} log P(E|F;Φ))/(Σ_E |E|) )

For this perplexity, lower is better, so if it's decreasing we're learning something.

## Attention

### Basic Concept

One of the major advances in NMT has been the introduction of attention (Bahdanau et al. 2015). The basic idea behind attention is that when we want to generate a particular target word `e_i`, that we will want to focus on a particular source word `f_j`, or a couple words.

TODO: Make this explanation more.

If you want to try to train an attentional model with lamtram, just change all mentions of `encdec` above to `encatt` (for encoder/attentional), and an attentional model will be trained for you. Try this, and I think you will see that perplexity decreases signficantly more quickly, demonstrating that we've made the learning problem a bit easier.

### Types of Attention

There are several ways to calculate attention, such as those investigate by Luong et al. (2015):

* Dot Product: TODO
* Bilinear: TODO
* Multi-layer Perceptron: TODO

In practice, I've found that dot product and multi-layer perceptron tend to perform well, but the results are different for different corpora. I'd recommend trying both and seeing which one works the best on your particular data set. 

## Testing NMT Systems

Now that we have a system, and know how good it is doing on the training set (according to perplexity), we will want to test to see how well it will do on data that is not included in training.

TODO: Command to measure perplexity.

TODO: Command to decode and measure BLEU.

Of course, these results are not yet satisfying, so we'll have to do some more work to get things working properly.

## Handling Unknown Words

### Thresholding Unknown Words

### Unknown Word Replacement

## Using a Development Set

### Early Stopping

### Rate Decay

## Minibatching

## Other Update Rules

TODO: Adam, etc.

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

### Using Subword Units

### Incorporating Discrete Lexicons

### Training for Other Evaluation Measures

## Preparing Data

### Data Size

Up until now, you have just been working with the small data set of 10,000 that I've provided. Having about 10,000 sentences makes training relatively fast, but having more data will make accuracy significantly higher. Fortunately, there is a larger data set of about 140,000 sentences called `train-big.ja` and `train-big.en`, which you can download by running the following commands.

  TODO

Try re-running experiments with this larger data set, and you will see that the accuracy gets significantly higher. In real NMT systems, it's common to use several million sentences (or more!) to achieve usable accuracies.

### Preprocessing

One thing to note is that up until now, we've taken it for granted that our data is split into words and lower-cased. When you build an actual system, this will not be the case, so you'll have to perform these processes yourself. Here, for tokenization we're using:

* English: [Moses](http://) (TODO et al.)
* Japanese: [KyTea](http://phontron.com/kytea/) (Neubig et al. 2011)

And for lowercasing we're using:

  perl -nle 'print lc'

Make sure that you do tokenization, and potentially lowercasing, before feeding your data into lamtram, or any MT toolkit. You can see an example of how we converted the Tanaka Corpus into the data used for the tutorial by looking at `data/create-data.sh` script.

## Final Word

Now, you know a few practical things about making an accurate neural MT system. This is a very fast-moving field, so this guide might be obsolete in a few months from the writing (or even already!) but hopefully this helped build the basics.

## References

* Brown et al. 1992
* Koehn 2010
* Neubig et al. 2011
* Kalchbrenner & Blunsom 2013
* Sutskever et al. 2014
* Goldberg 2015
* Bahdanau et al. 2015

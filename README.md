# Finetuning nanoGPT

This project is based on [nanoGPT](https://github.com/karpathy/nanoGPT) and [Stanford CS224N 2023 default project](https://github.com/gpoesia/minbert-default-final-project).

The [Stanford CS224N 2023 default project](https://github.com/gpoesia/minbert-default-final-project) finetunes [a Bert base model trained by Hugging face](https://huggingface.co/bert-base-uncased) which has 110M parameters. This project finetunes the nanoGPT small model which has 124M parameters.

The purpose is to practice finetuning from scratch, including data preparation, loss function, training and validation scripts etc.

Experiments environment is one single machine on
* Linux-6.5.0-14-generic-x86_64-with-glibc2.35
* Python 3.9.18
* one NVIDIA GeForce GTX 1080

Hyperparameters:
* learning_rate = 5e-5
* dropout = 0.2
* epochs = 20
* gradient_accumulation_steps = 32 

Block size is longer than >95% of training data, and batch size is adjusted to fit the memory.


## SST ([Stanford Sentiment Analysis](https://nlp.stanford.edu/sentiment/treebank.html))
Data from [Stanford CS224N 2023 default project](https://github.com/gpoesia/minbert-default-final-project)

Train data 8,544 sentences, validation 1101 sentences. 

There are 5 classes of sentiment, including negative 0, somewhat negative 1, neutral 2, somewhat positive 3, and positive 4.

A training data example:
`It 's a lovely film with lovely performances by Buy and Accorsi .	3`
where `3` is the label, meaning the sentiment is somewhat positive.

Best performance: 0.519. (cs224n project baseline 0.515).

Block size is set to 64, batch size 64.

## CFIMDB
Data from [Stanford CS224N 2023 default project](https://github.com/gpoesia/minbert-default-final-project)

Train has 1,707 sentences, and validation 245 sentences. 2 classes of sentiment, negative or positive.

A training data example:
`We know from other movies that the actors are good and they make the movie . Not at all a waste of time . The premise was not bad . One workable idea ( interaction between real bussiness men and Russian mafia ) is followed by an intelligent script	1`
where `1` is the label, meaning the sentiment is positive.

Best performance 0.984. (cs224n project baseline 0.966).

Block size 512, batch size 16.


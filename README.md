
Some work from 2023.

# Finetuning nanoGPT

This project is based on [nanoGPT](https://github.com/karpathy/nanoGPT) and [Stanford CS224N 2023 default project](https://github.com/gpoesia/minbert-default-final-project).

The [Stanford CS224N default project](https://github.com/gpoesia/minbert-default-final-project) finetunes [a Bert base model trained by Hugging face](https://huggingface.co/bert-base-uncased) which has 110M parameters. This project finetunes the nanoGPT small model which has 124M parameters.

The purpose is to experiment finetuning from scratch, starting from data preparation.

Experiments environment is one single machine on
* Linux-6.5.0-14-generic-x86_64-with-glibc2.35
* one RTX 3090
* Python 3.9.18
* Pytorch 2.1.2

Hyperparameters:
* learning_rate = 5e-5
* dropout = 0.2
* epochs = 20
* gradient_accumulation_steps = 32 

Block size is longer than >95% of the training sentences plus prompting question, and batch size is adjusted to fit the GPU.


## SST ([Stanford Sentiment Analysis](https://nlp.stanford.edu/sentiment/treebank.html))
Data from [Stanford CS224N 2023 default project](https://github.com/gpoesia/minbert-default-final-project)

Train data 8,544 sentences, validation 1101 sentences. 

There are 5 classes of sentiment, including negative 0, somewhat negative 1, neutral 2, somewhat positive 3, and positive 4.

Training sentence example:
`It 's a lovely film with lovely performances by Buy and Accorsi .	3`
where `3` is the label, meaning the sentiment is somewhat positive.

Training data is constructed by appending a question: "On a scale of 0 to 4, where 0 is negative, 2 neutral and 4 positive, rate sentence sentiment:"
For example:
```python
x = "It 's a lovely film with lovely performances by Buy and Accorsi . On a scale of 0 to 4, where 0 is negative, 2 neutral and 4 positive, rate sentence sentiment:"
y = 3
```
Followed by tokenization and padding. Details please see `data/sst/prepare.py`.

**Best performance: 0.527**. (cs224n project requirement 0.515).


## CFIMDB
Data from [Stanford CS224N 2023 default project](https://github.com/gpoesia/minbert-default-final-project)

Train has 1,707 sentences, and validation 245 sentences. 2 classes of sentiment, negative or positive.

Training and validation data are prepared in a similar way like SST, with the prompt "Is the sentence sentiment negative or positive:".
When computing loss, I use the tokens of "positive" and "negative", rather than 1 and 0.

Best performance 0.988. (cs224n project baseline 0.966).



# Finetuning nanoGPT

This project is based on [nanoGPT](https://github.com/karpathy/nanoGPT) and [Stanford CS224N 2023 default project](https://github.com/gpoesia/minbert-default-final-project).

The [Stanford CS224N 2023 default project](https://github.com/gpoesia/minbert-default-final-project) finetunes [a Bert base model trained by Hugging face](https://huggingface.co/bert-base-uncased) which has 110M parameters. This project finetunes the nanoGPT small model which has 124M parameters.

The purpose is to practice finetuning from scratch, including data preparation, loss function, training and validation scripts etc.


## SST ([Stanford Sentiment Analysis](https://nlp.stanford.edu/sentiment/treebank.html))
Data from [Stanford CS224N 2023 default project](https://github.com/gpoesia/minbert-default-final-project)

Train data 8,544 sentences, validation 1101 sentences.

A training data example:
`It 's a lovely film with lovely performances by Buy and Accorsi .	3`
where `3` is the label, meaning the sentiment is somewhat positive.

There are 5 classes of sentiment, including negative 0, somewhat negative 1, neutral 2, somewhat positive 3, and positive 4.

Best performance: 0.519. (project requirement 0.515)

Hyperparameters:
* learning_rate = 5e-5
* dropout = 0.2
* epochs = 20
* batch_size = 64
* gradient_accumulation_steps = 32 




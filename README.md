# AutoMLWrapper

# Training Details

All experiments are performed in a \textit{Jupyter} environment of the server discussed in the paper. 
The server is equipped with two Nvidia Quadro RTX 8000 graphics cards, each of which has 48GB of graphics memory available. 
The server also has two Intel Xeon Gold 6230R processors, each with 26 cores and 754 GB of RAM.
Training for all models is initially limited to one hour. In cases where this time limit is exceeded
time limit is exceeded, the question arises as to what effect an increase in the time limit has on the further training
of the AutoML process. For the AutoKeras library, for which no time limit can be defined, a division into 50 models for 50
epochs is used. In the first test for tabular data, this division resulted in an approximate time of
60 minutes. Except for one model, all LLMs in GPT4All are designed for the English language. The
Conversations with the LLMs are therefore uniformly conducted in English.

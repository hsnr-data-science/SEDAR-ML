# AutoMLWrapper

# Training Details

All experiments are performed in a Jupyter environment of the server discussed in the paper. 
The server is equipped with two Nvidia Quadro RTX 8000 graphics cards, each of which has 48GB of graphics memory available. 
The server also has two Intel Xeon Gold 6230R processors, each with 26 cores and 754 GB of RAM.
Training for all models is initially limited to one hour. For the AutoKeras library, for which no time limit can be defined, a division into 20 models for 30
epochs is used. Except for one model, all LLMs in GPT4All are designed for the English language. The
Conversations with the LLMs are therefore uniformly conducted in English.

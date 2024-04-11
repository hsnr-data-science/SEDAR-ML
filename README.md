# Supporting-Material

This repo contains supporting information, code and the datasets for our publication [Integrating AutoML and LLMs with the open-source data lake SEDAR](https://link.com).

# AutoMLWrapper
The AutoMLWrapper is a Python module acting as an interface for three open-source AutoML frameworks: AutoKeras, AutoGluon and AutoSklearn.
Special emphasis is placed on the standardized construction of AutoML jobs by configuring the underlying libraries using a common set of parameters.
Object-oriented programming for library-unspecific logic facilitates easy integration of new AutoML libraries without redesigning the entire interface.

# Notebooks
The folder notebooks contains the the Jupyter Notebooks to conduct the evaluation. They can be executed alongside the [SEDAR default image](https://hub.docker.com/r/mxibbls/gpu-jupyter-mlflow/) (stable tag). 

# Training Details

All experiments are performed in a Jupyter environment of the server discussed in the paper. 
The server is equipped with two Nvidia Quadro RTX 8000 graphics cards, each of which has 48GB of graphics memory available. 
The server also has two Intel Xeon Gold 6230R processors, each with 26 cores and 754 GB of RAM.
Training for all models is initially limited to one hour. For the AutoKeras library, for which no time limit can be defined, a division into 20 models for 30
epochs is used. Except for one model, all LLMs in GPT4All are designed for the English language. The
Conversations with the LLMs are therefore uniformly conducted in English.

## License

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png


## Acknowledgments
Sayed Hoseini and Gaoyuan Zhang equally contribute to this paper. 
The authors acknowledge gratefully the cooperation with the HIT Institute which made this work possible. 
This work has been sponsored by the German Federal Ministry of Education and Research, Germany in the funding program “Forschung an Fachhochschulen”, project IDACH (grant no. 13FH557KX0, [i2DACH](https://www.hs-niederrhein.de/i2dach) ).


## Citing
If you use this work in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:

```
@article{bibtex}
```

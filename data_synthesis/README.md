### Data Synthesis

Synthesizing NLI triplets from unlabelled sentences includes below consecutive parts:


#### Joint Training
Firstly, we train a unified generator/discriminator model with supervised NLI triplets (MNLI+SNLI):
```shell script
bash train.sh
```

#### NLI Triplets Generation
Then, we generate NLI triplets from unlabelled sentences with data generator:
```shell script
bash gen.sh
```

#### Noisy Triplets Discrimination
Lastly, we utilize the data discriminator to filter out noisy triplets with low confidence:
```shell script
bash filter.sh
```
In addition, we define several rules to further filter out identical sentences or short sentences. The final clean synthetic dataset will be in ``${OUTPUTDIR}/synli.csv``, which can be used in sentence representation learning.

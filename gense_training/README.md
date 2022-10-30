### GenSE Training & Evaluation


#### Training
Train GenSE on synthetic and human annotated NLI data:
```shell script
bash train.sh
```


#### Evaluation
To evaluate GenSE/GenSE+ on STS tasks:
```shell script
bash eval.sh
```
The results will be as follows:

| Model  | STS12 | STS13 | STS14 | STS15 | STS16 | STSb  | SICK-R | Avg.  |
| :---   | :---: | :---: | :---: | :---: | :---: | :---: | :---:  | :---: |
| GenSE  | 80.72 | 87.43 | 83.96 | 88.63 | 85.19 | 87.65 | 79.87  | 84.78 |
| GenSE+ | 80.65 | 88.18 | 84.69 | 89.03 | 85.82 | 87.88 | 80.10  | 85.19 |
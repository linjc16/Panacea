# [Click Here for Demo](https://03420227b3fe5e0d1c.gradio.live/)
# Panacea: A foundation model for clinical trial design, recruitment, search, and summarization
This repository is the official implementation of [Panacea: A foundation model for clinical trial design, recruitment, search, and summarization](https://arxiv.org/pdf/2407.11007). Model can be downloaded [here](https://huggingface.co/linjc16/Panacea-7B-Chat).

# Requirements and Installation
See `requirements.txt`.

# Get Started
Here we reproduced all eight tasks across different settings in our code base, including trial design, patient-trial matching, trial search, and trial summarization. 

## Data Download
* [TrialAlign](https://huggingface.co/datasets/linjc16/TrialAlign)
* [TrialInstruct](https://doi.org/10.6084/m9.figshare.25990090.v1)
* [TrialPanorama](https://doi.org/10.6084/m9.figshare.25990075)

## Alignment Step
We first use collected `TrialAlign` dataset to adapt Panacea to the vocabulary commonly used in clinical trials. Run the following
```[bash]
bash scripts/pretrain/run_pretrain_full.sh
```
## Instruction-tuning Step
Then, we conduct instruction-tuning step to enable Panacea to comprehend the user explanation of the task definition and the output requirement. Run
```[bash]
bash scripts/sft/sft.sh
```

## Evaluation
Take patient-trial matching as an example, just run
```[bash]
bash scripts/eval/matching/patient2trial/panacea-7b.sh
```
To calculate the metrics, run
```[bash]
bash scripts/eval/matching/patient2trial/metrics/cls.sh 
```
Evaluation of the other tasks is in the same way.



## Contact
Please feel free to submit a Github issue if you have any questions or find any bugs. We do not guarantee any support, but will do our best if we can help.

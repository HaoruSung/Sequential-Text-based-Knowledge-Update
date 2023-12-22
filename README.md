# Sequential Text-based Knowledge Update with Self-Supervised Learning for Generative Language Models
- This work proposes a new natural language processing (NLP) task to tackle the issue of multi-round, sequential text-based knowledge update.
- A dataset was also created for evaluation and results showed the effectiveness of our methodology. 


This is the code for the ACM CIKM 2023 paper :

[Sequential Text-based Knowledge Update with Self-Supervised Learning for Generative Language Models](https://dl.acm.org/doi/10.1145/3583780.3615188)   
[ [code](https://github.com/HaoruSung/Sequential-Text-based-Knowledge-Update) | [video](https://drive.google.com/file/d/1QPh00cUW4ySk0Pket6wq7R16E-sY9vAJ/view?usp=sharing) | [poster](https://github.com/HaoruSung/Sequential-Text-based-Knowledge-Update/blob/main/doc/CIKM2023_poster.pdf) | [slides](https://github.com/HaoruSung/Sequential-Text-based-Knowledge-Update/blob/main/doc/CIKM2023_slides.pdf) ]

## Setup

Clone the repo

```bash
git clone https://github.com/HaoruSung/Sequential-Text-based-Knowledge-Update.git
cd Sequential-Text-based-Knowledge-Update
```

install dependencies

```bash
pip install -r requirements.txt
```

You are all set! 🎉

## Finetune the model

```bash
python Finetune/fine_tuning_t5_model.py
```

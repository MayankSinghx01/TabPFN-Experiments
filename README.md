# TabPFN Experiments

Testing whether TabPFN, a Transformer pretrained only on synthetic tables, can match gradient boosting on real data with far fewer rows. TabPFN does no training on the new data: it reads the training rows as context and predicts in one pass.

- `richter.ipynb`: earthquake building damage
- `flushot.ipynb`: H1N1 and seasonal flu vaccine uptake
- `msedcl_forecast.ipynb`: monthly power-outage frequency (SAIFI) for 50 MSEDCL circles
- `baselines.py`: gradient boosting and simple baselines on the same splits

Results (TabPFN vs gradient boosting):

- Earthquake damage, F1: 0.72 with 10k rows vs 0.73 with 208k rows
- H1N1 vaccine, AUC: 0.87 vs 0.86 with the same 3k rows
- Seasonal vaccine, AUC: 0.88 vs 0.85 with the same 3k rows
- SAIFI forecast, error (WAPE, lower is better): 64% vs 67%

Each test set has 500 rows, so small differences are within noise.

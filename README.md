# fetal-birthweight-prediction

How to use the models to evaluate fetal birthweight?

## 1. Operating environment
Main dependencies: Python 3.8, PyTorch 1.7.1, scikit-learn 1.1.1

## 2. Create a Conda environment
```bash
conda env create -f environment.yml
conda activate birthweight # activate the environment
```

## 3.Models Download
[Download Trained Models](https://drive.google.com/drive/folders/10Pb_-1agBeV_iQDu6nrW0j0qw6xX6FBg?usp=sharing)

The whole zip package includes the models that had been trained. Please put them in model folder like this:

```text
├── model
|  └── forest.dat
|  └── mlp.model
|  └── ridge.dat
|  └── svr.dat
|  └── xgb.dat
├── modules
├── pred.py
├── README.md
```


## 4. Modify sample and get result

change the parameters in pred.py and run it! such as:
```python
data_dict = {
    'Preg_Days': 274, # Days of pregnancy, used to calculate gestational weeks
    'pre_weight': 69.0, #pre-pregnancy weight
    'maternal_weight_last': 81.3, #last birth weight
    'days_last_ul_to_delivery': 7, # Days from last ultrasound to delivery
    'pre_BMI': 23.7, #pre-pregnancy BMI
    'GWG': 12.3, #Weight gain during pregnancy, which can be calculated by
    'GA_last_inspect': 7, #Days from the last check-up to delivery
    'GDM': 0, # gestational diabetes，1 or 0
    'Parity': 1, #parity, integer
    'BPD': 95.0, # Biparietal diameter，mm
    'HC': 335.0, # head circumference，mm
    'FL': 70.0, # femur length，mm
    'HL': 60.0, # humerus length，mm
    'AC': 332.0, # abdominal circumference，mm
    'TTD': 106.0, # abdominal left and right diameter，mm
    'APTD': 106.0, # Abdominal anteroposterior diameter，mm
    'AFI_Sum': 115.0, # total amniotic fluid index
                }
```

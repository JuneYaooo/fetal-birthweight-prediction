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
    'Preg_Days': 274, # 怀孕天数，用于计算孕周
    'pre_weight': 69.0, #孕前体重
    'maternal_weight_last': 81.3, #末次产检体重
    'days_last_ul_to_delivery': 7, # 末次超声距分娩天数
    'pre_BMI': 23.7, #孕前BMI
    'GWG': 12.3, #孕期体重增加，可以通过计算得出
    'GA_last_inspect': 7, #末次产检距分娩天数
    'GDM': 0, # 妊娠期糖尿病，1，0P
    'Parity': 1, #产次，整数
    'BPD': 95.0, # 双顶径，单位mm
    'HC': 335.0, # 头围，单位mm
    'FL': 70.0, # 股骨长，单位mm
    'HL': 60.0, # 肱骨长，单位mm
    'AC': 332.0, # 腹围，单位mm
    'TTD': 106.0, # 腹左右径，单位mm
    'APTD': 106.0, # 腹前后径，单位mm
    'AFI_Sum': 115.0, # 羊水指数合计
                }
```

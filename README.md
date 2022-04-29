# Optimization of ShallowConvNet for Enhancing the Decoding Performance of Motor Imagery-based EEG Signals

This repository is the official implementation of M-ShallowConvNet in pytorch-lightning style.

- [M-ShallowConvNet paper]() - TBA

```
@article{
  TBA
}
```

## 1. Installation

## 2. Performance
### 2.1 BCI Competition IV 2a

|Subject No.|A1|A2|A3|A4|A5|A6|A7|A8|A9|Avg.|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|ShallowConvNet|0.833|0.517|0.931|0.743|0.747|0.625|0.816|0.847|0.823|0.765|
|M-ShallowConvNet|0.910|0.556|0.938|0.806|0.816|0.660|0.938|0.851|0.875|0.816|

### 2.2 BCI Competition IV 2b

|Subject No.|B1|B2|B3|B4|B5|B6|B7|B8|B9|Avg.|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|ShallowConvNet|0.759|0.689|0.762|0.963|0.997|0.841|0.925|0.916|0.844|0.855|
|M-ShallowConvNet|0.781|0.686|0.812|0.953|0.984|0.884|0.916|0.931|0.834|0.865|

## 3. Feature visualization on BCI Competition IV 2a

|t-SNE|Singular Value|
|:-:|:-:|
|<p align="center" width="100%"><img src="https://user-images.githubusercontent.com/46745325/165910596-05a2434f-5abd-430e-9f27-6629111914f6.png"></img></p>|<p align="center" width="100%"><img src="https://user-images.githubusercontent.com/46745325/165910710-76eeca18-33c1-42ca-9010-7fcb307581aa.png"></img></p>|

## 4. Pre-trained weight

- [BCI Competition IV 2a]()
- [BCI Competition IV 2a]()

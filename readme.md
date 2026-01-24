# DLOps Lab Assignment-1  
## Detailed Performance Analysis on MNIST & FashionMNIST

**Name:** Vidhan Savaliya  
**Roll Number:** M25CSA031  


---

##  Assignment Overview
This assignment presents a **systematic experimental evaluation** of deep learning and classical machine learning models for image classification using:

- **Datasets:** MNIST, FashionMNIST  
- **Deep Models:** ResNet-18, ResNet-50  
- **Classical Model:** Support Vector Machine (SVM)  
- **Total Experiments:** 128 controlled runs  

Hyperparameters varied across:
- Optimizers (SGD, Adam)
- Learning rates (0.001, 0.0001)
- Batch sizes (16, 32)
- Pin memory (True / False)
- CPU vs GPU execution

---

##  Best Performing Model — Q1(a)

**Best single-run configuration (MNIST):**

- **Model:** ResNet-18  
- **Optimizer:** SGD  
- **Learning Rate:** 0.001  
- **Batch Size:** 16  
- **Pin Memory:** True  
- **Epochs:** 5  
- **Test Accuracy:** **99.06%**

This was the **highest accuracy observed across all 128 experiments**, showing that for simpler datasets like MNIST, a **well-tuned SGD setup can outperform adaptive optimizers in peak performance**.

---

##  Aggregated Best Results per Configuration

| ID | Dataset | Model | Batch | Optimizer | LR | PinMem | Epochs | TestAcc(%) | Precision | Recall | F1 | TrainTime(s) |
|----|---------|-------|-------|-----------|----|--------|--------|------------|-----------|--------|----|--------------|
| 0 | MNIST | ResNet18 | 16 | SGD | 0.001 | FALSE | 2 | 98.55 | 0.985509 | 0.985604 | 0.985500 | 88.355937 |
| 1 | MNIST | ResNet18 | 16 | SGD | 0.001 | FALSE | 5 | 98.907143 | 0.989066 | 0.989106 | 0.989071 | 215.126785 |
| 2 | MNIST | ResNet18 | 16 | SGD | 0.001 | TRUE | 2 | 98.657143 | 0.986574 | 0.986648 | 0.986571 | 85.081275 |
| 3 | MNIST | ResNet18 | 16 | SGD | 0.001 | TRUE | 5 | 99.064286 | 0.990639 | 0.990657 | 0.990643 | 213.298061 |
| 4 | MNIST | ResNet18 | 16 | SGD | 0.0001 | FALSE | 2 | 97.807143 | 0.978071 | 0.978104 | 0.978071 | 84.992753 |
| 5 | MNIST | ResNet18 | 16 | SGD | 0.0001 | FALSE | 5 | 98.385714 | 0.983846 | 0.983884 | 0.983857 | 212.983056 |
| 6 | MNIST | ResNet18 | 16 | SGD | 0.0001 | TRUE | 2 | 97.885714 | 0.978862 | 0.978889 | 0.978857 | 85.207665 |
| 7 | MNIST | ResNet18 | 16 | SGD | 0.0001 | TRUE | 5 | 98.528571 | 0.985277 | 0.985317 | 0.985286 | 213.325875 |
| 8 | MNIST | ResNet18 | 16 | Adam | 0.001 | FALSE | 2 | 96.8 | 0.968001 | 0.970254 | 0.968000 | 97.750800 |
| 9 | MNIST | ResNet18 | 16 | Adam | 0.001 | FALSE | 5 | 98.757143 | 0.987575 | 0.987678 | 0.987571 | 243.239947 |
| 10 | MNIST | ResNet18 | 16 | Adam | 0.001 | TRUE | 2 | 98.764286 | 0.987635 | 0.987706 | 0.987643 | 98.805884 |
| 11 | MNIST | ResNet18 | 16 | Adam | 0.001 | TRUE | 5 | 98.535714 | 0.985329 | 0.985521 | 0.985357 | 245.804646 |
| 12 | MNIST | ResNet18 | 16 | Adam | 0.0001 | FALSE | 2 | 97.478571 | 0.974741 | 0.975715 | 0.974786 | 96.957254 |
| 13 | MNIST | ResNet18 | 16 | Adam | 0.0001 | FALSE | 5 | 98.857143 | 0.988573 | 0.988591 | 0.988571 | 245.238042 |
| 14 | MNIST | ResNet18 | 16 | Adam | 0.0001 | TRUE | 2 | 98.385714 | 0.983870 | 0.983981 | 0.983857 | 97.967685 |
| 15 | MNIST | ResNet18 | 16 | Adam | 0.0001 | TRUE | 5 | 98.064286 | 0.980628 | 0.980871 | 0.980643 | 243.655813 |
| 16 | MNIST | ResNet18 | 32 | SGD | 0.001 | FALSE | 2 | 98.392857 | 0.983932 | 0.984027 | 0.983929 | 44.594754 |
| 17 | MNIST | ResNet18 | 32 | SGD | 0.001 | FALSE | 5 | 98.978571 | 0.989786 | 0.989802 | 0.989786 | 113.459134 |
| 18 | MNIST | ResNet18 | 32 | SGD | 0.001 | TRUE | 2 | 98.485714 | 0.984855 | 0.984917 | 0.984857 | 44.992326 |
| 19 | MNIST | ResNet18 | 32 | SGD | 0.001 | TRUE | 5 | 98.735714 | 0.987366 | 0.987403 | 0.987357 | 110.067907 |
| 20 | MNIST | ResNet18 | 32 | SGD | 0.0001 | FALSE | 2 | 96.928571 | 0.969285 | 0.969380 | 0.969286 | 45.184097 |
| 21 | MNIST | ResNet18 | 32 | SGD | 0.0001 | FALSE | 5 | 97.85 | 0.978483 | 0.978568 | 0.978500 | 109.425615 |
| 22 | MNIST | ResNet18 | 32 | SGD | 0.0001 | TRUE | 2 | 96.971429 | 0.969694 | 0.969773 | 0.969714 | 44.669331 |
| 23 | MNIST | ResNet18 | 32 | SGD | 0.0001 | TRUE | 5 | 97.771429 | 0.977712 | 0.977723 | 0.977714 | 111.902050 |
| 24 | MNIST | ResNet18 | 32 | Adam | 0.001 | FALSE | 2 | 98.207143 | 0.982069 | 0.982232 | 0.982071 | 51.262735 |
| 25 | MNIST | ResNet18 | 32 | Adam | 0.001 | FALSE | 5 | 98.807143 | 0.988072 | 0.988101 | 0.988071 | 127.235181 |
| 26 | MNIST | ResNet18 | 32 | Adam | 0.001 | TRUE | 2 | 98.685714 | 0.986852 | 0.986961 | 0.986857 | 51.504911 |
| 27 | MNIST | ResNet18 | 32 | Adam | 0.001 | TRUE | 5 | 98.964286 | 0.989641 | 0.989678 | 0.989643 | 126.790282 |
| 28 | MNIST | ResNet18 | 32 | Adam | 0.0001 | FALSE | 2 | 97.671429 | 0.976759 | 0.977127 | 0.976714 | 50.303544 |
| 29 | MNIST | ResNet18 | 32 | Adam | 0.0001 | FALSE | 5 | 98.478571 | 0.984787 | 0.984922 | 0.984786 | 127.856656 |
| 30 | MNIST | ResNet18 | 32 | Adam | 0.0001 | TRUE | 2 | 98.014286 | 0.980150 | 0.980401 | 0.980143 | 49.268462 |
| 31 | MNIST | ResNet18 | 32 | Adam | 0.0001 | TRUE | 5 | 98.514286 | 0.985151 | 0.985286 | 0.985143 | 125.943760 |
| 32 | MNIST | ResNet50 | 16 | SGD | 0.001 | FALSE | 2 | 97.65 | 0.976595 | 0.977183 | 0.976500 | 177.990785 |
| 33 | MNIST | ResNet50 | 16 | SGD | 0.001 | FALSE | 5 | 98.45 | 0.984438 | 0.984597 | 0.984500 | 445.923378 |
| 34 | MNIST | ResNet50 | 16 | SGD | 0.001 | TRUE | 2 | 97.392857 | 0.973989 | 0.974807 | 0.973929 | 180.662423 |
| 35 | MNIST | ResNet50 | 16 | SGD | 0.001 | TRUE | 5 | 98.778571 | 0.987782 | 0.987829 | 0.987786 | 447.302748 |
| 36 | MNIST | ResNet50 | 16 | SGD | 0.0001 | FALSE | 2 | 95.964286 | 0.959641 | 0.959815 | 0.959643 | 178.711887 |
| 37 | MNIST | ResNet50 | 16 | SGD | 0.0001 | FALSE | 5 | 97.65 | 0.976474 | 0.976630 | 0.976500 | 446.592763 |
| 38 | MNIST | ResNet50 | 16 | SGD | 0.0001 | TRUE | 2 | 96.264286 | 0.962584 | 0.962870 | 0.962643 | 178.470820 |
| 39 | MNIST | ResNet50 | 16 | SGD | 0.0001 | TRUE | 5 | 97.721429 | 0.977199 | 0.977293 | 0.977214 | 446.752992 |
| 40 | MNIST | ResNet50 | 16 | Adam | 0.001 | FALSE | 2 | 96.607143 | 0.966201 | 0.967804 | 0.966071 | 203.562373 |
| 41 | MNIST | ResNet50 | 16 | Adam | 0.001 | FALSE | 5 | 98.471429 | 0.984720 | 0.984812 | 0.984714 | 512.928502 |
| 42 | MNIST | ResNet50 | 16 | Adam | 0.001 | TRUE | 2 | 89.314286 | 0.885004 | 0.922864 | 0.893143 | 205.003287 |
| 43 | MNIST | ResNet50 | 16 | Adam | 0.001 | TRUE | 5 | 98.321429 | 0.983189 | 0.983281 | 0.983214 | 510.412644 |
| 44 | MNIST | ResNet50 | 16 | Adam | 0.0001 | FALSE | 2 | 96.25 | 0.962444 | 0.962726 | 0.962500 | 203.987934 |
| 45 | MNIST | ResNet50 | 16 | Adam | 0.0001 | FALSE | 5 | 98 | 0.979963 | 0.980223 | 0.980000 | 512.751820 |
| 46 | MNIST | ResNet50 | 16 | Adam | 0.0001 | TRUE | 2 | 96.807143 | 0.968059 | 0.968200 | 0.968071 | 206.514813 |
| 47 | MNIST | ResNet50 | 16 | Adam | 0.0001 | TRUE | 5 | 97.807143 | 0.978046 | 0.978269 | 0.978071 | 512.405394 |
| 48 | MNIST | ResNet50 | 32 | SGD | 0.001 | FALSE | 2 | 97.792857 | 0.977928 | 0.978136 | 0.977929 | 92.675926 |
| 49 | MNIST | ResNet50 | 32 | SGD | 0.001 | FALSE | 5 | 98.35 | 0.983489 | 0.983576 | 0.983500 | 227.663362 |
| 50 | MNIST | ResNet50 | 32 | SGD | 0.001 | TRUE | 2 | 97.742857 | 0.977435 | 0.977619 | 0.977429 | 91.672861 |
| 51 | MNIST | ResNet50 | 32 | SGD | 0.001 | TRUE | 5 | 98.585714 | 0.985847 | 0.985897 | 0.985857 | 226.101295 |
| 52 | MNIST | ResNet50 | 32 | SGD | 0.0001 | FALSE | 2 | 93.528571 | 0.935370 | 0.935951 | 0.935286 | 91.415019 |
| 53 | MNIST | ResNet50 | 32 | SGD | 0.0001 | FALSE | 5 | 96.785714 | 0.967864 | 0.968011 | 0.967857 | 230.665051 |
| 54 | MNIST | ResNet50 | 32 | SGD | 0.0001 | TRUE | 2 | 93.842857 | 0.938476 | 0.938754 | 0.938429 | 91.797894 |
| 55 | MNIST | ResNet50 | 32 | SGD | 0.0001 | TRUE | 5 | 96.942857 | 0.969381 | 0.969524 | 0.969429 | 229.977396 |
| 56 | MNIST | ResNet50 | 32 | Adam | 0.001 | FALSE | 2 | 95.135714 | 0.951185 | 0.953578 | 0.951357 | 104.497362 |
| 57 | MNIST | ResNet50 | 32 | Adam | 0.001 | FALSE | 5 | 98.114286 | 0.981149 | 0.981189 | 0.981143 | 260.947787 |
| 58 | MNIST | ResNet50 | 32 | Adam | 0.001 | TRUE | 2 | 97.571429 | 0.975743 | 0.976150 | 0.975714 | 104.484349 |
| 59 | MNIST | ResNet50 | 32 | Adam | 0.001 | TRUE | 5 | 98.2 | 0.982005 | 0.982207 | 0.982000 | 263.053784 |
| 60 | MNIST | ResNet50 | 32 | Adam | 0.0001 | FALSE | 2 | 94.907143 | 0.949162 | 0.950157 | 0.949071 | 104.630257 |
| 61 | MNIST | ResNet50 | 32 | Adam | 0.0001 | FALSE | 5 | 97.157143 | 0.971575 | 0.971786 | 0.971571 | 261.196265 |
| 62 | MNIST | ResNet50 | 32 | Adam | 0.0001 | TRUE | 2 | 94.664286 | 0.946602 | 0.947255 | 0.946643 | 104.800598 |
| 63 | MNIST | ResNet50 | 32 | Adam | 0.0001 | TRUE | 5 | 97.2 | 0.971953 | 0.972271 | 0.972000 | 260.498787 |
| 64 | FashionMNIST | ResNet18 | 16 | SGD | 0.001 | FALSE | 2 | 89.014286 | 0.890595 | 0.892746 | 0.890143 | 86.763352 |
| 65 | FashionMNIST | ResNet18 | 16 | SGD | 0.001 | FALSE | 5 | 90.1 | 0.900651 | 0.901831 | 0.901000 | 215.805686 |
| 66 | FashionMNIST | ResNet18 | 16 | SGD | 0.001 | TRUE | 2 | 86.207143 | 0.862285 | 0.878694 | 0.862071 | 85.805088 |
| 67 | FashionMNIST | ResNet18 | 16 | SGD | 0.001 | TRUE | 5 | 90.107143 | 0.901197 | 0.901722 | 0.901071 | 210.814127 |
| 68 | FashionMNIST | ResNet18 | 16 | SGD | 0.0001 | FALSE | 2 | 85.885714 | 0.859825 | 0.864911 | 0.858857 | 85.963984 |
| 69 | FashionMNIST | ResNet18 | 16 | SGD | 0.0001 | FALSE | 5 | 88.292857 | 0.881451 | 0.881495 | 0.882929 | 210.690578 |
| 70 | FashionMNIST | ResNet18 | 16 | SGD | 0.0001 | TRUE | 2 | 86.95 | 0.869519 | 0.870524 | 0.869500 | 84.357115 |
| 71 | FashionMNIST | ResNet18 | 16 | SGD | 0.0001 | TRUE | 5 | 88.757143 | 0.887898 | 0.888529 | 0.887571 | 212.108158 |
| 72 | FashionMNIST | ResNet18 | 16 | Adam | 0.001 | FALSE | 2 | 86.492857 | 0.864866 | 0.883850 | 0.864929 | 97.469665 |
| 73 | FashionMNIST | ResNet18 | 16 | Adam | 0.001 | FALSE | 5 | 89.421429 | 0.894652 | 0.896283 | 0.894214 | 242.255827 |
| 74 | FashionMNIST | ResNet18 | 16 | Adam | 0.001 | TRUE | 2 | 88.321429 | 0.882973 | 0.884709 | 0.883214 | 97.243505 |
| 75 | FashionMNIST | ResNet18 | 16 | Adam | 0.001 | TRUE | 5 | 90.671429 | 0.906165 | 0.906961 | 0.906714 | 244.008739 |
| 76 | FashionMNIST | ResNet18 | 16 | Adam | 0.0001 | FALSE | 2 | 88.564286 | 0.886322 | 0.888889 | 0.885643 | 97.594784 |
| 77 | FashionMNIST | ResNet18 | 16 | Adam | 0.0001 | FALSE | 5 | 89.757143 | 0.898469 | 0.900516 | 0.897571 | 243.527458 |
| 78 | FashionMNIST | ResNet18 | 16 | Adam | 0.0001 | TRUE | 2 | 86.842857 | 0.868519 | 0.873505 | 0.868429 | 97.837528 |
| 79 | FashionMNIST | ResNet18 | 16 | Adam | 0.0001 | TRUE | 5 | 89.5 | 0.895556 | 0.897169 | 0.895000 | 245.785283 |
| 80 | FashionMNIST | ResNet18 | 32 | SGD | 0.001 | FALSE | 2 | 88.178571 | 0.877873 | 0.884152 | 0.881786 | 45.180971 |
| 81 | FashionMNIST | ResNet18 | 32 | SGD | 0.001 | FALSE | 5 | 89.085714 | 0.890645 | 0.893143 | 0.890857 | 111.566961 |
| 82 | FashionMNIST | ResNet18 | 32 | SGD | 0.001 | TRUE | 2 | 88.078571 | 0.878949 | 0.881449 | 0.880786 | 44.684100 |
| 83 | FashionMNIST | ResNet18 | 32 | SGD | 0.001 | TRUE | 5 | 89.985714 | 0.899444 | 0.900392 | 0.899857 | 111.173840 |
| 84 | FashionMNIST | ResNet18 | 32 | SGD | 0.0001 | FALSE | 2 | 85.721429 | 0.855151 | 0.855246 | 0.857214 | 45.357002 |
| 85 | FashionMNIST | ResNet18 | 32 | SGD | 0.0001 | FALSE | 5 | 87.492857 | 0.874886 | 0.875352 | 0.874929 | 109.205193 |
| 86 | FashionMNIST | ResNet18 | 32 | SGD | 0.0001 | TRUE | 2 | 85.871429 | 0.859238 | 0.860107 | 0.858714 | 44.162372 |
| 87 | FashionMNIST | ResNet18 | 32 | SGD | 0.0001 | TRUE | 5 | 87.592857 | 0.874567 | 0.875356 | 0.875929 | 107.962632 |
| 88 | FashionMNIST | ResNet18 | 32 | Adam | 0.001 | FALSE | 2 | 88.078571 | 0.877974 | 0.883024 | 0.880786 | 50.531666 |
| 89 | FashionMNIST | ResNet18 | 32 | Adam | 0.001 | FALSE | 5 | 89.95 | 0.900190 | 0.902525 | 0.899500 | 127.590689 |
| 90 | FashionMNIST | ResNet18 | 32 | Adam | 0.001 | TRUE | 2 | 88.264286 | 0.880603 | 0.883966 | 0.882643 | 50.564523 |
| 91 | FashionMNIST | ResNet18 | 32 | Adam | 0.001 | TRUE | 5 | 89.9 | 0.897369 | 0.900779 | 0.899000 | 126.643501 |
| 92 | FashionMNIST | ResNet18 | 32 | Adam | 0.0001 | FALSE | 2 | 88.521429 | 0.885051 | 0.885129 | 0.885214 | 50.737906 |
| 93 | FashionMNIST | ResNet18 | 32 | Adam | 0.0001 | FALSE | 5 | 89.1 | 0.890097 | 0.889944 | 0.891000 | 125.910042 |
| 94 | FashionMNIST | ResNet18 | 32 | Adam | 0.0001 | TRUE | 2 | 87.421429 | 0.874717 | 0.878961 | 0.874214 | 50.269625 |
| 95 | FashionMNIST | ResNet18 | 32 | Adam | 0.0001 | TRUE | 5 | 88.364286 | 0.883803 | 0.887289 | 0.883643 | 125.667137 |
| 96 | FashionMNIST | ResNet50 | 16 | SGD | 0.001 | FALSE | 2 | 85.528571 | 0.852506 | 0.860420 | 0.855286 | 177.343883 |
| 97 | FashionMNIST | ResNet50 | 16 | SGD | 0.001 | FALSE | 5 | 88.285714 | 0.879218 | 0.887530 | 0.882857 | 446.852117 |
| 98 | FashionMNIST | ResNet50 | 16 | SGD | 0.001 | TRUE | 2 | 84.985714 | 0.844172 | 0.850956 | 0.849857 | 177.685039 |
| 99 | FashionMNIST | ResNet50 | 16 | SGD | 0.001 | TRUE | 5 | 88.792857 | 0.888014 | 0.889429 | 0.887929 | 445.948294 |
| 100 | FashionMNIST | ResNet50 | 16 | SGD | 0.0001 | FALSE | 2 | 80.25 | 0.799518 | 0.800097 | 0.802500 | 177.617096 |
| 101 | FashionMNIST | ResNet50 | 16 | SGD | 0.0001 | FALSE | 5 | 85.178571 | 0.848700 | 0.852158 | 0.851786 | 445.172010 |
| 102 | FashionMNIST | ResNet50 | 16 | SGD | 0.0001 | TRUE | 2 | 79.028571 | 0.784646 | 0.804791 | 0.790286 | 186.665335 |
| 103 | FashionMNIST | ResNet50 | 16 | SGD | 0.0001 | TRUE | 5 | 85.25 | 0.847295 | 0.853289 | 0.852500 | 475.011566 |
| 104 | FashionMNIST | ResNet50 | 16 | Adam | 0.001 | FALSE | 2 | 84.107143 | 0.841161 | 0.846155 | 0.841071 | 217.895180 |
| 105 | FashionMNIST | ResNet50 | 16 | Adam | 0.001 | FALSE | 5 | 88.15 | 0.879071 | 0.881524 | 0.881500 | 524.347786 |
| 106 | FashionMNIST | ResNet50 | 16 | Adam | 0.001 | TRUE | 2 | 84.342857 | 0.840620 | 0.844573 | 0.843429 | 216.216160 |
| 107 | FashionMNIST | ResNet50 | 16 | Adam | 0.001 | TRUE | 5 | 87.714286 | 0.877766 | 0.881610 | 0.877143 | 543.674725 |
| 108 | FashionMNIST | ResNet50 | 16 | Adam | 0.0001 | FALSE | 2 | 84.664286 | 0.844365 | 0.848958 | 0.846643 | 239.594875 |
| 109 | FashionMNIST | ResNet50 | 16 | Adam | 0.0001 | FALSE | 5 | 87.064286 | 0.870475 | 0.876092 | 0.870643 | 588.738320 |
| 110 | FashionMNIST | ResNet50 | 16 | Adam | 0.0001 | TRUE | 2 | 83.771429 | 0.834945 | 0.837573 | 0.837714 | 236.793452 |
| 111 | FashionMNIST | ResNet50 | 16 | Adam | 0.0001 | TRUE | 5 | 87.25 | 0.871345 | 0.873811 | 0.872500 | 585.165637 |
| 112 | FashionMNIST | ResNet50 | 32 | SGD | 0.001 | FALSE | 2 | 84.814286 | 0.848040 | 0.852046 | 0.848143 | 97.687922 |
| 113 | FashionMNIST | ResNet50 | 32 | SGD | 0.001 | FALSE | 5 | 88.85 | 0.887725 | 0.888682 | 0.888500 | 230.841173 |
| 114 | FashionMNIST | ResNet50 | 32 | SGD | 0.001 | TRUE | 2 | 84.928571 | 0.851537 | 0.859346 | 0.849286 | 92.505578 |
| 115 | FashionMNIST | ResNet50 | 32 | SGD | 0.001 | TRUE | 5 | 88.207143 | 0.879484 | 0.882073 | 0.882071 | 227.150103 |
| 116 | FashionMNIST | ResNet50 | 32 | SGD | 0.0001 | FALSE | 2 | 78.65 | 0.778616 | 0.784123 | 0.786500 | 93.213138 |
| 117 | FashionMNIST | ResNet50 | 32 | SGD | 0.0001 | FALSE | 5 | 83.814286 | 0.834582 | 0.835835 | 0.838143 | 229.116314 |
| 118 | FashionMNIST | ResNet50 | 32 | SGD | 0.0001 | TRUE | 2 | 79.2 | 0.788242 | 0.787145 | 0.792000 | 95.240226 |
| 119 | FashionMNIST | ResNet50 | 32 | SGD | 0.0001 | TRUE | 5 | 83.535714 | 0.832785 | 0.835335 | 0.835357 | 240.495728 |
| 120 | FashionMNIST | ResNet50 | 32 | Adam | 0.001 | FALSE | 2 | 84.778571 | 0.846139 | 0.857323 | 0.847786 | 110.994563 |
| 121 | FashionMNIST | ResNet50 | 32 | Adam | 0.001 | FALSE | 5 | 81.15 | 0.806625 | 0.835053 | 0.811500 | 275.709854 |
| 122 | FashionMNIST | ResNet50 | 32 | Adam | 0.001 | TRUE | 2 | 84.242857 | 0.839880 | 0.845941 | 0.842429 | 110.702560 |
| 123 | FashionMNIST | ResNet50 | 32 | Adam | 0.001 | TRUE | 5 | 84.471429 | 0.839909 | 0.850925 | 0.844714 | 266.255402 |
| 124 | FashionMNIST | ResNet50 | 32 | Adam | 0.0001 | FALSE | 2 | 83.557143 | 0.835672 | 0.838939 | 0.835571 | 109.311506 |
| 125 | FashionMNIST | ResNet50 | 32 | Adam | 0.0001 | FALSE | 5 | 86.5 | 0.864428 | 0.868729 | 0.865000 | 271.624363 |
| 126 | FashionMNIST | ResNet50 | 32 | Adam | 0.0001 | TRUE | 2 | 82.207143 | 0.817505 | 0.825799 | 0.822071 | 113.818326 |
| 127 | FashionMNIST | ResNet50 | 32 | Adam | 0.0001 | TRUE | 5 | 86.657143 | 0.866063 | 0.866098 | 0.866571 | 269.739542 |





## 🧠 Analytical Performance Summary

| Aspect | MNIST Best Model | FashionMNIST Best Model |
|------|-----------------|------------------------|
| Configuration | ResNet-18 + SGD | ResNet-18 + Adam |
| Peak Accuracy | 99.06% | 90.67% |
| Convergence | Moderate & stable | Fast & early saturation |
| Training Stability | High | Very High |
| Overfitting | Minimal | Minimal–Moderate |
| Generalization | Excellent | Strong |

**Reality check:**  
- ResNet-50 **does not justify its depth** for MNIST-level complexity.  
- Adam is more robust on FashionMNIST, but **SGD still wins peak MNIST accuracy**.

---

## 🧮 SVM Classification Results

### SVM Performance Summary (MNIST & FashionMNIST)

| Dataset | Kernel | Hyperparameters | Test Accuracy (%) | Train Time (ms) |
|--------|--------|-----------------|------------------:|----------------:|
| MNIST | Poly | Degree=4, C=1.0 | **95.86** | 6426 |
| MNIST | RBF | C=10, γ=0.001 | 95.20 | 8167 |
| FashionMNIST | Poly | Degree=3, C=1.0 | 86.48 | 5809 |
| FashionMNIST | RBF | C=10 | **86.51** | 8156 |

**Key observation:**  
SVMs perform competitively but **do not scale well**. Training time explodes with kernel complexity and poor hyperparameter choices, reinforcing why CNNs dominate image tasks.

---

##  CPU vs GPU Performance Analysis

### Performance Comparison (FashionMNIST)

| Compute | Optimizer | Acc R18 (%) | Acc R50 (%) | Time R18 (ms) | Time R50 (ms) | FLOPs R18 | FLOPs R50 |
|--------|-----------|-------------|-------------|---------------|---------------|-----------|-----------|
| CPU | SGD | 89.86 | 86.77 | 3,038,957 | 5,482,543 | 66.11 | 156.30 |
| CPU | Adam | 89.95 | 87.27 | 3,659,143 | 6,603,654 | 66.11 | 156.30 |
| GPU | SGD | 89.78 | 87.14 | 240,042 | 487,784 | 66.11 | 156.30 |
| GPU | Adam | **90.07** | 86.57 | 270,296 | 559,413 | 66.11 | 156.30 |

---

## Best Model - CPU vs GPU



**ResNet-18 + Adam (LR=0.001, Batch=16) on GPU**

- **Accuracy:** 90.07%
- **Training Time:** >10× faster than CPU
- **FLOPs:** Less than half of ResNet-50

 **Deeper ≠ Better**. Efficient models + GPU matter more.

---

## 📈 Training Curves
- Training & validation loss
- Accuracy curves
- Confusion matrix  

(See figures in the PDF report)

---

##  Final Conclusion

- ResNet-18 consistently **matches or outperforms ResNet-50**
- SGD can outperform Adam **when tuned correctly**
- SVMs are limited by scalability and training cost
- GPU acceleration is **non-negotiable** for deep learning
- Model depth must match dataset complexity

**Big takeaway:**  
> Efficiency, optimizer choice, and hardware matter more than blindly increasing model depth.

---

##  Contact
**Vidhan Savaliya**  
Roll No: M25CSA031  
Email: m25csa031@iitj.ac.in


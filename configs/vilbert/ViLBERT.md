# ViLBERT

## Introduction

```
@InProceedings{Lu_2020_CVPR,
author = {Lu, Jiasen and Goswami, Vedanuj and Rohrbach, Marcus and Parikh, Devi and Lee, Stefan},
title = {12-in-1: Multi-Task Vision and Language Representation Learning},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}

@inproceedings{lu2019vilbert,
  title={Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks},
  author={Lu, Jiasen and Batra, Dhruv and Parikh, Devi and Lee, Stefan},
  booktitle={Advances in Neural Information Processing Systems},
  pages={13--23},
  year={2019}
}
```

## Results and Models

|       Task        |  Style  |    Accuracy    |                            Config                            |                           Download                           |
| :---------------: | :-----: | :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|        VQA        | pytorch | 71.43%(71.18%) | [config](https://mega.nz/file/MCYDESxb#rHPsVgUJ0qheWD0LqzQ0wC5UB4fLlp_aEdN6USYjOGo) | [model](https://mega.nz/file/kSQBCSbZ#grdUQc8xhx4A14FueNSmJetGM2yP8PgXkQe88wrtFBo) &#124; [log](https://mega.nz/file/xCIDRKaQ#ZrISgzAtCECYOda5eLKM9o_FVTjdjoNrCSuuaiLvVZA) |
|        GQA        | pytorch | 67.90%(67.90%) | [config](https://mega.nz/file/RGRBGYBS#z0coqrWpAoUx1Q-tgJpGH1j0IasVnXHqpyMC0sQemIE) | [model](https://mega.nz/file/IPJ3RIyI#8PYAHeyDLn02KdWRgsof43lHX8LAj9EhFpPuGyVhFJg) &#124; [log](https://mega.nz/file/0PAzQahZ#gknzrTuCtE8JVNd0UyEt2WOtEGTtkwlWwFRMVHQJ_ro) |
|      refcoco      | pytorch | 85.56%(85.44%) | [config](https://mega.nz/file/4KJxiKpa#5FyX7GTk-kiLKBRaNZpM4sV80JXHjmlXAUlGd13JYYA) | [model](https://mega.nz/file/RGBDlI6K#Iplw4Q_Bv-iL10WiJBuzT1YkOgu50Y3idZ7zXH0pKeY) &#124; [log](https://mega.nz/file/dfQl2K7b#9SskUFWFj7y7QTg0tT76ZE3peQD0P5blSqID5DgefaM) |
|     refcoco+      | pytorch | 78.30%(78.26%) | [config](https://mega.nz/file/cPIzwALJ#mhvdQ6WNn4VuC6SFIXQK_UWQBMQKVT5EsRCmWCZXy9Q) | [model](https://mega.nz/file/cOIDHaoQ#YbW7NfIeDk_KfLMyG_jNBuXmW-OsYvnX6nNterBXDP0) &#124; [log](https://mega.nz/file/BGRDjSQS#ycTta_5Fl17EDMTaZ6HJlhT7ZbRh47N-oQykN8yxb6I) |
|     refcocog      | pytorch | 80.76%(80.84%) | [config](https://mega.nz/file/RSBVRaCb#2jn2Zmey5DLMS6Czmq4wcM9MQAm7Zz7J5tFeHcP4xT4) | [model](https://mega.nz/file/YHJTnYyT#tRGyyyc70PiiusJgg1qFoS5AUGZPw68-yevHA2FAwLg) &#124; [log](https://mega.nz/file/NSJRxY4Z#UQyYHtqX5okekDljqURBHBR1QkKd_0-y8cAe1STfSgA) |
|   Retrivalcoco    | pytorch | 99.52%(99.58%) | [config](https://mega.nz/file/cWAznarC#yuL_kKDakuY2vwseq-rCLuDNjyN8qg9Co4g9t5giU94) | [model](https://mega.nz/file/gPhnVQyC#cSUHhDErBS5hiZlJKDHgT4xwXVNGjGgYKgOA_2v2HLA) &#124; [log](https://mega.nz/file/sXBVAKqI#2gfAKep4GXbbi3oAzmicRt9RSjoqWVx1157BPj4PaGc) |
|     GenomeQA      | pytorch | 35.88%(35.86%) | [config](https://mega.nz/file/keYBVKQQ#Q-EptUg0eXUaaCRp-HYLMeurOwqyteoioe639VkUmM4) | [model](https://mega.nz/file/wSQhHC4I#H85OTmPj7_FsWFGsIoyIJg0Q-Z0zmZWKD0C9eVg4gcI) &#124; [log](https://mega.nz/file/AaYzHCrY#xmiPR4LqoPU3F3Mgri-VgiL99DMYLfJGSO3Z8Uq8wx0) |
| RetrivalFlickr30k | pytorch | 99.46%(99.42%) | [config](https://mega.nz/file/QOJ3SQLB#XPqh4Z0Af8qVVhxbXdm0wWU_GljV5ZS5Ua7ra0voerw) | [model](https://mega.nz/file/pCJHSCRB#IQxE8uqwOh4eiLQGtlzTHOKUEmzsZ9TuV7d_83x5N_c) &#124; [log](https://mega.nz/file/5fBHVYIJ#rkTz5t5IxgCqi3bx2rho3Rh4kel-s2Dpfrt-BU-rOn4) |
|     Visual7w      | pytorch | 83.42%(83.32%) | [config](https://mega.nz/file/MDZ2VZqA#4PanT72EibFWCZhbRK_C9hY2Z5XtA9IooPflq4zew-s) | [model](https://mega.nz/file/gSAiEbhY#X_irazghGJmz8zugZhpHDQiYpJcsW1kM8dOxlw6xoPY) &#124; [log](https://mega.nz/file/cHYmEZgQ#RHY5r2EL3e9L7aKghotOk5PMKt6wFQChEeXcLSsbYd0) |
|     GussWhat      | pytorch | 63.36%(62.92%) | [config](https://mega.nz/file/4WAU0JzB#b_uW4sxHBtrqc3Qn4-jhkzzHQGIjBtKNjrLGmExhoyo) | [model](https://mega.nz/file/sWAkRbRZ#Efs9O9dvsMAY64Lrj5l1zScvTqy3rWQgGKcTzYHRzDE) &#124; [log](https://mega.nz/file/BbQQSRTA#XtjKqZ0pnbn_gCl_gS5XGJR7Q7J526w03dKS3QGAWHI) |
| VisualEntailment  | pytorch | 77.08%(77.07%) | [config](https://mega.nz/file/9WBQXJLJ#slz1ClqCcp1SLfM2LaAtZDishxKG19VufieCsZ0bOns) | [model](https://mega.nz/file/RCQUWRRL#b7nVeQpl8MvY1mryz0G9sNgL42uQ958GTRR2kDrNpi0) &#124; [log](https://mega.nz/file/hLJWiRQJ#g8rxcUyM3ZWUU3xyhfSz3pAejwsPHNFjBzW6I9CwUWs) |
| GussWhatPointing  | pytorch | 65.78%(65.95%) | [config](https://mega.nz/file/ULQ0SZBY#e6EBOzF9eQOiqWQdJmfWWmjos1IIHl2SQ1IhWn0Rigo) | [model](https://mega.nz/file/NbQiyb4a#SfbXp3H9BjanSoCYb_hwmWi5ikpNAaefA7NBdpfo7Vo) &#124; [log](https://mega.nz/file/EaJ0jbra#REjnny8C8ADti2vAIrtsLl-pJParHy_cOvRZW6nXk0A) |

**Notes:**

- The accuracy values in the brackets represent those reported in the source code of https://github.com/facebookresearch/vilbert-multi-task.

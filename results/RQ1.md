# Detailed results for RQ1

## HDFS dataset


### Message Count Vector

|   Models       |                     | Metrics   |        |          |
| :------------- | :-----------------: | :-------: | :----: | :------: |
|                |                     | Precision | Recall | F1-Score |
| Traditional    | SVM                 | 0\.999    | 0\.917 | 0\.956   |
|                | Decision Tree       | 1\.000    | 0\.998 | 0\.999   |
|                | Logistic Regression | 1\.000    | 0\.996 | 0\.998   |
|                | Random Forest       | 0\.998    | 1\.000 | 0\.999   |
| Deep           | MLP                 | 0\.999    | 0\.999 | 0\.999   |


### Event Template ID based TF-IDF

|   Models       |                     | Metrics   |        |          |
| :------------- | :-----------------: | :-------: | :----: | :------: |
|                |                     | Precision | Recall | F1-Score |
| Traditional    | SVM                 | 0\.999    | 0\.999 | 0\.999   |
|                | Decision Tree       | 1\.000    | 0\.998 | 0\.999   |
|                | Logistic Regression | 0\.999    | 0\.997 | 0\.998   |
|                | Random Forest       | 0\.999    | 1\.000 | 0\.999   |
| Deep           | MLP                 | 0\.911    | 1\.000 | 0\.953   |


### Template Text based TF-IDF

|   Models       |                     | Metrics   |        |          |
| :------------- | :-----------------: | :-------: | :----: | :------: |
|                |                     | Precision | Recall | F1-Score |
| Traditional    | SVM                 | 0\.999    | 0\.979 | 0\.989   |
|                | Decision Tree       | 0\.985    | 0\.999 | 0\.992   |
|                | Logistic Regression | 1\.000    | 0\.900 | 0\.947   |
|                | Random Forest       | 0\.997    | 0\.999 | 0\.998   |
| Deep           | MLP                 | 0\.987    | 0\.999 | 0\.993   |
|                | CNN                 | 0\.982    | 0\.922 | 0\.951   |
|                | LSTM                | 0\.991    | 0\.922 | 0\.955   |

### Word2Vec

|   Models       |                     | Metrics   |        |          |
| :------------- | :-----------------: | :-------: | :----: | :------: |
|                |                     | Precision | Recall | F1-Score |
| Traditional    | SVM                 | 0\.998    | 0\.998 | 0\.998   |
|                | Decision Tree       | 0\.985    | 0\.998 | 0\.992   |
|                | Logistic Regression | 0\.999    | 0\.901 | 0\.948   |
|                | Random Forest       | 0\.999    | 0\.985 | 0\.992   |
| Deep           | MLP                 | 0\.911    | 0\.999 | 0\.953   |
|                | CNN                 | 0\.985    | 0\.923 | 0\.953   |
|                | LSTM                | 0\.997    | 0\.921 | 0\.958   |

### FastText

|   Models       |                     | Metrics   |        |          |
| :------------- | :-----------------: | :-------: | :----: | :------: |
|                |                     | Precision | Recall | F1-Score |
| Traditional    | SVM                 | 0\.998    | 0\.998 | 0\.998   |
|                | Decision Tree       | 0\.985    | 0\.998 | 0\.992   |
|                | Logistic Regression | 1\.000    | 0\.884 | 0\.938   |
|                | Random Forest       | 0\.999    | 0\.985 | 0\.992   |
| Deep           | MLP                 | 0\.911    | 1\.000 | 0\.954   |
|                | CNN                 | 0\.990    | 0\.922 | 0\.955   |
|                | LSTM                | 0\.993    | 0\.920 | 0\.955   |

### BERT

|   Models       |                     | Metrics   |        |          |
| :------------- | :-----------------: | :-------: | :----: | :------: |
|                |                     | Precision | Recall | F1-Score |
| Traditional    | SVM                 | 0\.998    | 0\.998 | 0\.998   |
|                | Decision Tree       | 0\.985    | 0\.998 | 0\.992   |
|                | Logistic Regression | 0\.999    | 0\.999 | 0\.999   |
|                | Random Forest       | 0\.998    | 1\.000 | 0\.999   |
| Deep           | MLP                 | 0\.911    | 0\.999 | 0\.953   |
|                | CNN                 | 0\.992    | 0\.921 | 0\.955   |
|                | LSTM                | 0\.998    | 0\.923 | 0\.959   |

## BGL dataset

### Message Count Vector

|   Models       |                     | Metrics   |        |          |
| :------------- | :-----------------: | :-------: | :----: | :------: |
|                |                     | Precision | Recall | F1-Score |
| Traditional    | SVM                 | 0\.958    | 0\.840 | 0\.895   |
|                | Decision Tree       | 0\.959    | 0\.921 | 0\.939   |
|                | Logistic Regression | 0\.947    | 0\.889 | 0\.917   |
|                | Random Forest       | 0\.830    | 0\.963 | 0\.891   |
| Deep           | MLP                 | 0\.958    | 0\.840 | 0\.895   |


### Event Template ID based TF-IDF

|   Models       |                     | Metrics   |        |          |
| :------------- | :-----------------: | :-------: | :----: | :------: |
|                |                     | Precision | Recall | F1-Score |
| Traditional    | SVM                 | 0\.828    | 0\.654 | 0\.731   |
|                | Decision Tree       | 0\.959    | 0\.919 | 0\.938   |
|                | Logistic Regression | 0\.882    | 0\.741 | 0\.805   |
|                | Random Forest       | 0\.810    | 0\.951 | 0\.875   |
| Deep           | MLP                 | 0\.951    | 0\.951 | 0\.951   |

### Template Text based TF-IDF

|   Models       |                     | Metrics   |        |          |
| :------------- | :-----------------: | :-------: | :----: | :------: |
|                |                     | Precision | Recall | F1-Score |
| Traditional    | SVM                 | 0\.855    | 0\.728 | 0\.787   |
|                | Decision Tree       | 0\.971    | 0\.963 | 0\.967   |
|                | Logistic Regression | 0\.868    | 0\.728 | 0\.792   |
|                | Random Forest       | 0\.872    | 0\.946 | 0\.907   |
| Deep           | MLP                 | 0\.927    | 0\.938 | 0\.933   |
|                | CNN                 | 0\.900    | 1\.000 | 0\.947   |
|                | LSTM                | 0\.866    | 0\.877 | 0\.871   |
### Word2Vec


|   Models       |                     | Metrics   |        |          |
| :------------- | :-----------------: | :-------: | :----: | :------: |
|                |                     | Precision | Recall | F1-Score |
| Traditional    | SVM                 | 0\.853    | 0\.716 | 0\.779   |
|                | Decision Tree       | 0\.781    | 0\.701 | 0\.739   |
|                | Logistic Regression | 0\.871    | 0\.753 | 0\.808   |
|                | Random Forest       | 0\.667    | 0\.783 | 0\.720   |
| Deep           | MLP                 | 0\.895    | 0\.840 | 0\.866   |
|                | CNN                 | 0\.868    | 0\.975 | 0\.919   |
|                | LSTM                | 0\.755    | 0\.988 | 0\.856   |

### FastText

|   Models       |                     | Metrics   |        |          |
| :------------- | :-----------------: | :-------: | :----: | :------: |
|                |                     | Precision | Recall | F1-Score |
| Traditional    | SVM                 | 0\.869    | 0\.654 | 0\.746   |
|                | Decision Tree       | 0\.734    | 0\.654 | 0\.692   |
|                | Logistic Regression | 0\.844    | 0\.667 | 0\.745   |
|                | Random Forest       | 0\.681    | 0\.808 | 0\.738   |
| Deep           | MLP                 | 0\.868    | 0\.815 | 0\.841   |
|                | CNN                 | 0\.857    | 0\.963 | 0\.907   |
|                | LSTM                | 0\.822    | 0\.914 | 0\.865   |


### BERT

|   Models       |                     | Metrics   |        |          |
| :------------- | :-----------------: | :-------: | :----: | :------: |
|                |                     | Precision | Recall | F1-Score |
| Traditional    | SVM                 | 0\.871    | 0\.667 | 0\.746   |
|                | Decision Tree       | 0\.812    | 0\.701 | 0\.752   |
|                | Logistic Regression | 0\.886    | 0\.765 | 0\.821   |
|                | Random Forest       | 0\.694    | 0\.806 | 0\.745   |
| Deep           | MLP                 | 0\.910    | 0\.877 | 0\.893   |
|                | CNN                 | 0\.939    | 0\.951 | 0\.945   |
|                | LSTM                | 0\.871    | 1\.000 | 0\.931   |
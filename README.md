# WB_Task2
A simple model for labeling fake reviews on russian marketplace. 
## Research
During research multiple articles were found, most used neural networks and behavioural features unable, both are not compatible with offered dataset.  
It was decided to generate new features based on available features and use conventional ML methods.
## Baseline
Simple array labeling all reviews as fake was chosen as baseline.
Baseline metrics:  
Precision:  0.294  
Recall:  1.0  
F1:  0.4544049459041731  
Confusion matrix for baseline:  
![image](https://github.com/AVPankov0/WB_Task2/assets/133259896/525e2572-1595-4451-8eeb-f273abbe05b0)
## Model
CatBoostClassifier was chosen as model based on it's results compared to other candidates. 
After tuning, following results were achieved:  
Precision:  0.75  
Recall:  0.24489795918367346  
F1:  0.369230769230769  
Confusion matrix for CatBoostClassifier:  
![image](https://github.com/AVPankov0/WB_Task2/assets/133259896/d9808e99-33d9-4089-9ae4-70d11927da04)
## Structure
### Configuration
Adjusting model parameters and changing work mode for model is done through editing *conf.py* file.
### Data
Data format and used dataset is stored in *data* folder. It is possible to use more numerical features as long as they are between *text* and *label* columns.
Model is saved in *models* folders
### Preprocessing
Preprocessing of data done throug files in *preprocess* folder:  
* *preprocess.py* generates new features for inputted dataset.
* *feature_imporatnce.py* selects most prominent features based on their influence on precision and f1 scores.
List of important features is stored in *data* folder.
### Notebooks
Folder *notebooks* contains notebooks used for EDA and selecting classifier model and models used in them. 

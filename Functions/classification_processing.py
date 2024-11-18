# Required packages
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score



def explore_estimators(X, y, estimators, voting='hard'):
    """
    Evaluates multiple classifiers using cross-validation and displays confusion matrices and reports.
    Additionally, it creates a hard or soft (depending on parameter `voting`) voting classifier with the estimators passed.
    """        
    names = ['Sane', 'Infected']

    all_estimators = estimators.copy()
    all_estimators.append(('Voting', VotingClassifier(estimators=estimators, voting=voting)))

    # Evaluate each estimator
    for name, estimator in all_estimators:
        predictions = cross_val_predict(estimator, X, y, cv=5)  # Cross-validated predictions. 5 folds
        
        # Display confusion matrix
        print(f'CLASSIFIER USED: {name}')
        cm = confusion_matrix(y, predictions)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names).plot(cmap='Blues')
        plt.title(f"Confusion Matrix - {name}")
        plt.show()

        # Display classification report
        print("Classification Report:")
        print(classification_report(y, predictions, target_names=names))
        print('\n' + '='*50 + '\n')



def inference_pass(patient_data, X, y, metadata, estimator, train=False, patch_level=False):
    """
    Performs inference with a given estimator and saves those inferences to the patient_data dictionary.
    `metadata` is used to keep track of the origin of the patches or patients.
    If `train=True` the estimator is fitted and the function returns a trained model.
    If `patch_level` is set to `True`, the function works at the patch-level. Otherwise, it works at the patient-level.
    """
    # Train estimator if labels are provided
    if train:
        estimator.fit(X, y)
    
    # Make predictions
    predictions = estimator.predict(X)
    
    # Store predictions in patient_data
    if patch_level:
        for (patient_id, patch_idx), prediction in zip(metadata, predictions):
            patient_data[patient_id]['images'][patch_idx]['prediction'] = prediction  

    else:
        for patient_id, prediction in zip(metadata, predictions):            
            patient_data[patient_id]['patient_prediction'] = prediction

    return estimator if train else None


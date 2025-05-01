from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from monai.data import DataLoader, Dataset, CacheDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier, VotingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from ML.classification.classifiers.config.transforms_selector import transforms_selector
from ML.classification.classifiers.config.model_selector import model_selector
from ML.classification.classifiers.dataset.ClassificationDataset import ClassificationDataset
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR  # instead of SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np


def plot_confusion_matrix(true_labels, predicted_labels, predictor):
    CKD_stages = np.arange(1,6)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=CKD_stages, yticklabels=CKD_stages)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for validation set using {predictor}")
    plt.show()
    
    return fig
        

def train(model, 
            loss_function, 
            train_dataloader, 
            device, 
            optimizer):
    training_losses = []
    training_accuracy = []
    model.train()

    training_labels = []
    training_predictions = []
    
    correct = 0
    total = 0

    X_train, y_train = [], []
    for batch_data in tqdm(train_dataloader):
        images, labels = batch_data["image"].to(device), batch_data["label"].to(device, dtype=torch.long)
        #Labels should be 1 index
        labels = labels - 1

        optimizer.zero_grad()

        outputs, features = model(images)

        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()
        training_losses.append(loss.item())

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        training_labels.extend(labels.cpu().numpy())
        training_predictions.extend(predicted.cpu().numpy())

        X_train.append(features.detach().cpu().numpy())  # Convert tensor to numpy and store
        y_train.append(labels.cpu().numpy())

    y_train = np.concatenate(y_train, axis=0)

    training_accuracy.append(correct / total) # SKAL PÅ INNSIDEN?

    return np.mean(training_losses), np.mean(training_accuracy), X_train, y_train

def validate(model, loss_function, val_dataloader, device, optimizer, epoch, epochs_to_save=10): # FIX epochs to save
    correct = 0
    total = 0
    validation_labels = []
    validation_predictions = []
    validation_losses = []
    validation_accuracy = []

    model.eval() # ER DETTE RIKTIG EGT? BURDE DET VÆRE model.model.eval()
    with torch.no_grad():
        X_val, y_val = [], []
        for batch in tqdm(val_dataloader):
                images, labels = batch["image"].to(device), batch["label"].to(device, dtype=torch.long)
                
                labels = labels - 1

                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs, features = outputs
                else:
                    features = None

                loss = loss_function(outputs, labels)
                validation_losses.append(loss.item())

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                validation_labels.extend(labels.cpu().numpy())
                validation_predictions.extend(predicted.cpu().numpy())
        
        #if (epoch + 1) % epochs_to_save == 0:
        #    writer.add_figure("validation confusion matrix",
        #        plot_confusion_matrix(
        #            true_labels = np.array(validation_labels) + 1,
        #            predicted_labels = np.array(validation_predictions) + 1,
        #            epoch = epoch)
        #        ,global_step = epoch)
        #
        #    #Save checkpoint
        #    torch.save({
        #        'epoch': epoch,
        #        'model_state_dict': model.state_dict(),
        #        'optimizer_state_dict': optimizer.state_dict(),
        #        'loss': loss
        #        },f"classification_models/checkpoint_{model_name}.pth")
    
               

    validation_accuracy.append(correct / total)
    return np.mean(validation_losses), np.mean(validation_accuracy), validation_predictions

def train_loop(model, 
               epochs: int, 
               train_dataloader: DataLoader, 
               val_dataloader: DataLoader, 
               device: torch.device,
               writer: SummaryWriter,
               epochs_to_save: int,
               model_name: str):

    loss_function = torch.nn.CrossEntropyLoss() # CAN USE LABEL SMOOTHING 
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01) # With regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
            
        training_loss, training_accuracy = train(model, loss_function, train_dataloader, device, optimizer)

        validation_loss, validation_accuracy = validate(model, loss_function, val_dataloader, device, optimizer, epoch)

        print(f"""Epoch {epoch+1}, Average Training Loss: {training_loss},
                Average Validation Loss: {validation_loss}
                Average Accuracy: {validation_accuracy},
                Average Training Accuracy: {training_accuracy}%""")
        
        writer.add_scalar("Average training loss", training_loss, epoch)
        writer.add_scalar("Average validation loss", validation_loss, epoch)
        writer.add_scalar("Average accuracy", validation_accuracy, epoch)
        writer.add_scalar("Average Training accuracy", training_accuracy, epoch)
    
    writer.flush()

def train_model(model, dataloader, optimizer, loss_function, device):
    model.train()
    for batch in tqdm(dataloader):
        images, labels, noisy_label = batch["image"].to(device), batch["label"].to(device, dtype=torch.long), batch["noisy_label"].to(device, dtype=torch.float32)
        optimizer.zero_grad()
        labels = labels - 1
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs, features = outputs

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

def k_fold_validation(model_name,
                      dataset, 
                      epochs:int, 
                      batch_size: int, 
                      device: torch.device,
                      writer: SummaryWriter,
                      transforms_name: str,
                      num_workers: int,
                      splits: int = 5):
    
    
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

    nn_conf_matr_labels = []
    nn_conf_matr_pred = []
    rf_conf_matr_labels = []
    rf_conf_matr_pred = []
    svm_conf_matr_labels = []
    svm_conf_matr_pred = []

    train_transforms, val_transforms = transforms_selector(transforms_name)
    train_transforms_baseline, val_transforms_baseline = transforms_selector("pretrained")
    

    labels = np.array([sample["label"] for sample in dataset])
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    
        
        ## MODELS TO TEST
        model = model_selector(model_name, device)
        model_baseline = model_selector("resnet18", device)

        loss_function = torch.nn.CrossEntropyLoss() 
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
        optimizer_baseline = torch.optim.Adam(model_baseline.parameters(), lr=0.001)

        print(f"Fold {fold+1}/{splits}")

        train_set = [dataset[i] for i in train_idx]
        val_set = [dataset[i] for i in val_idx]
        
        train_ds = ClassificationDataset(data_list=train_set,
                                            start_frame=0,
                                            end_frame=None,
                                            agg="time_series",
                                            cache=False,
                                            transforms=train_transforms,
                                            radiomics=True,
                                            train=True
                                            )
    
        feature_ds = ClassificationDataset(data_list=train_set, 
                                            start_frame=0, 
                                            end_frame=None,
                                            agg="time_series", 
                                            cache=False,
                                            transforms=val_transforms,
                                            radiomics=True,
                                            train=True
                                            )

        val_ds = ClassificationDataset(data_list=val_set, 
                                            start_frame=0, 
                                            end_frame=None, 
                                            agg="time_series", 
                                            cache=None,
                                            transforms=val_transforms, 
                                            radiomics=True,
                                            train=False
                                            )

        train_ds_baseline = ClassificationDataset(data_list=train_set, 
                                            start_frame=6, 
                                            end_frame=18,
                                            agg="mean", 
                                            cache=False,
                                            transforms=train_transforms_baseline,
                                            radiomics=False,
                                            train=True
                                            )
        
        val_ds_baseline = ClassificationDataset(data_list=val_set, 
                                            start_frame=6, 
                                            end_frame=18,
                                            agg="mean", 
                                            cache=False,
                                            transforms=val_transforms_baseline,
                                            radiomics=False,
                                            train=False
                                            )
        
        val_ds.scaler, val_ds.imputer, val_ds.nan_cols = train_ds.get_objects()
        feature_ds.scaler, feature_ds.imputer, feature_ds.nan_cols = train_ds.get_objects()

        train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers) 
        feature_dataloader = DataLoader(feature_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        train_dataloader_baseline = DataLoader(train_ds_baseline, batch_size=batch_size, shuffle=True, num_workers=num_workers) 
        val_dataloader_baseline = DataLoader(val_ds_baseline, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        ## TRAIN BASELINE
        for epoch in range(5): # HVOR MANGE EPOKER?
            train_model(model_baseline, train_dataloader_baseline, optimizer_baseline, loss_function, device)
        
        ## EVAL BASELINE
        model_baseline.eval()
        _, accuracy_baseline, _ = validate(model_baseline, loss_function, val_dataloader_baseline, device, optimizer_baseline, epoch)    
        
        
        X_train, y_train = [], []
        X_val, y_val = [], []
        
        X_train_radiomics, X_val_radiomics = [], []

        for epoch in range(epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{epochs}")

            if epoch != epochs - 1:
                # TRAIN 3D CNN
                # TIME_SERIES FOR 3D CNN
                train_model(model, train_dataloader, optimizer, loss_function, device)

            else:
                # EVAL

                ## EVAL MODELS
                model.eval()
                model.backbone.eval()
                for param in model.backbone.parameters():
                    param.requires_grad = False
                with torch.no_grad():
                    # TRAIN FEATURES
                    for batch in tqdm(feature_dataloader):
                        images, labels, noisy_label, radiomics = batch["image"].to(device), batch["label"].to(device), batch["noisy_label"].to(device, dtype=torch.float32), batch["radiomics"]
                        labels = labels - 1
                        outputs, features = model(images)

                        features = features.detach().cpu().numpy() 
                        radiomics = radiomics.detach().cpu().numpy() if torch.is_tensor(radiomics) else radiomics 
                        combined_features = np.concatenate([features, radiomics], axis=1)  
                
                        X_train.append(features)
                        y_train.append(labels.cpu().numpy())  
                        X_train_radiomics.append(combined_features)
                    
                    #VAL FEATURES
                    total = 0
                    correct = 0
                    for batch in tqdm(val_dataloader):
                        img, label, noisy_label, radiomics = batch["image"].to(device), batch["label"].to(device), batch["noisy_label"].to(device, dtype=torch.long), batch["radiomics"]
                        label = label - 1
                        outputs, features = model(img)

                        # FOR NEURAL NETWORK PREDS
                        _, predicted = torch.max(outputs.data, 1)
                        nn_conf_matr_pred.extend(predicted.tolist())
                        nn_conf_matr_labels.extend(label.tolist())

                        total += label.size(0)
                        correct += (predicted == label).sum().item()

                        features = features.detach().cpu().numpy() 
                        radiomics = radiomics.detach().cpu().numpy() if torch.is_tensor(radiomics) else radiomics 
                        combined_features = np.concatenate([features, radiomics], axis=1)  

                        X_val.append(features)  
                        y_val.append(label.cpu().numpy())  
                        X_val_radiomics.append(combined_features)
                
                neural_network_acc = correct / total

                X_train = np.concatenate(X_train, axis=0)  
                y_train = np.concatenate(y_train, axis=0) 
                
                X_val = np.concatenate(X_val, axis=0)  
                y_val = np.concatenate(y_val, axis=0) 

                X_train_radiomics = np.concatenate(X_train_radiomics, axis=0) 
                X_val_radiomics = np.concatenate(X_val_radiomics, axis=0)  
                

                scaler = StandardScaler()

                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)


                scaler2 = StandardScaler()
                X_train_radiomics = scaler2.fit_transform(X_train_radiomics) # SCALER radiomics to ganger nå
                X_val_radiomics = scaler2.transform(X_val_radiomics)

                rf_validation_accuracy, rf_pred, svm_validation_accuracy, svm_pred = train_models(X_train, y_train, X_val, y_val)
                
                rf_conf_matr_labels.extend(y_val)
                rf_conf_matr_pred.extend(rf_pred)

                svm_conf_matr_labels.extend(y_val)
                svm_conf_matr_pred.extend(svm_pred)

                #CONFUSION MATRIX RADIOMICS

                print(f"BASELINE acc: {accuracy_baseline}")
                print(f"NN acc: {neural_network_acc}")
                print(f"RF acc: {rf_validation_accuracy}")
                print(f"SVM acc: {svm_validation_accuracy}")

                print("")
                print("WITH RADIOMICS")
                rf_validation_accuracy_radiomics, rf_pred_radiomics, svm_validation_accuracy_radiomics, svm_pred_radiomics = train_models(X_train_radiomics, y_train, X_val_radiomics, y_val)

                print(f"RF radiomics acc: {rf_validation_accuracy_radiomics}")
                print(f"SVM radiomics acc: {svm_validation_accuracy_radiomics}")

    #Report all labels and predictions across folds
    print(f"NN validation labels across all folds: {nn_conf_matr_labels}")
    print(f"NN validation predictions across all folds: {nn_conf_matr_pred}")
    print("\n")
    print(f"RF validation labels across all folds: {nn_conf_matr_labels}")
    print(f"RF validation predictions across all folds: {nn_conf_matr_pred}")
    print("\n")
    print(f"SVM validation labels across all folds: {nn_conf_matr_labels}")
    print(f"SVM validation predictions across all folds: {nn_conf_matr_pred}")

    #Plot confusion matrixs
    plot_confusion_matrix(nn_conf_matr_labels, nn_conf_matr_pred, "3D CNN + NN classifier")
    plot_confusion_matrix(rf_conf_matr_labels, rf_conf_matr_pred, "3D CNN + RF classifier")
    plot_confusion_matrix(svm_conf_matr_labels, svm_conf_matr_pred, "3D CNN + SVM classifier")

                


def train_models(X_train, y_train, X_val, y_val):

    rf_model = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_val)
    rf_validation_accuracy = accuracy_score(y_val, y_pred_rf)

    et_model = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=2 , max_features=1.0, bootstrap=True, random_state=42)
    et_model.fit(X_train, y_train)
    y_pred_et = et_model.predict(X_val)
    et_validation_accuracy = accuracy_score(y_val, y_pred_et)

    svm = SVC(kernel='linear', C=0.001, random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_val)
    svm_validation_accuracy = accuracy_score(y_val, y_pred_svm)

    knn_model = KNeighborsClassifier(n_neighbors=7)
    ensemble_knn = BaggingClassifier(
        estimator=knn_model,
        n_estimators=500,              
        max_samples=1.0,              
        max_features=1.0,             
        bootstrap=False,               
        bootstrap_features=True,      
        random_state=42,
        n_jobs=-1                     
    )
    ensemble_knn.fit(X_train, y_train)
    y_pred_knn = ensemble_knn.predict(X_val)
    knn_validation_accuracy = accuracy_score(y_val, y_pred_knn)

    logreg_model = LogisticRegression(solver='liblinear', max_iter=500, C=0.01) # use elasticnet penalty
    logreg_model.fit(X_train, y_train)
    y_pred_logreg = logreg_model.predict(X_val)
    logreg_validation_accuracy = accuracy_score(y_val, y_pred_logreg)

    print(f"KNN {knn_validation_accuracy}")
    print(f"Logreg {logreg_validation_accuracy}")
    print(f"ET {et_validation_accuracy}")

    svm_model_prob = SVC(kernel="linear", C=0.001, probability=True) 

    voting_clf = VotingClassifier(
        estimators=[
            ('svc', svm_model_prob),
            ('knn', ensemble_knn),
            ('logreg', logreg_model),
            ('rf', rf_model),
            ('et', et_model),
        ],
        voting='soft'
    )

    voting_clf.fit(X_train, y_train)
    y_pred_ensemble = voting_clf.predict(X_val)
    ensemble_validation_accuracy = accuracy_score(y_val, y_pred_ensemble)
    print(f"Ensemble {ensemble_validation_accuracy}")

    return rf_validation_accuracy, y_pred_rf, svm_validation_accuracy, y_pred_svm
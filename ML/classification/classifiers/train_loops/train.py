from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from monai.data import DataLoader, Dataset, CacheDataset
from torch.utils.tensorboard import SummaryWriter
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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import numpy as np


def plot_confusion_matrix(true_labels, predicted_labels, predictor):
    CKD_stages = np.arange(1,6)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=CKD_stages, yticklabels=CKD_stages)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for {predictor}")
    
    return fig

        

def train(model, 
            loss_function, 
            train_dataloader, 
            device, 
            optimizer,
            radiomics = False):
    training_losses = []
    training_accuracy = []
    model.train()

    training_labels = []
    training_predictions = []
    
    correct = 0
    total = 0

    for batch_data in tqdm(train_dataloader):
        images, labels, radiomic_feats = batch_data["image"].to(device), batch_data["label"].to(device, dtype=torch.long), batch_data["radiomics"]
        #Labels should be 1 index

        optimizer.zero_grad()

        if radiomics:
            outputs = model(images, radiomic_feats)
        else:
            outputs = model(images)

        if isinstance(outputs, tuple):
            outputs, features = outputs


        loss = loss_function(outputs, labels)

        # L1 REGULARIZATION

        #l1_lambda = 1e-2 for baseline eller -3
        l1_lambda = 1e-3
        l1_norm = sum(p.abs().sum() for p in model.fc.parameters())
        loss = loss + l1_lambda * l1_norm

        loss.backward()
        optimizer.step()
        training_losses.append(loss.item())

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        training_labels.extend(labels.cpu().numpy())
        training_predictions.extend(predicted.cpu().numpy())

    training_accuracy.append(correct / total) # SKAL PÅ INNSIDEN?

    return np.mean(training_losses), np.mean(training_accuracy)

def validate(model, loss_function, val_dataloader, device, optimizer, epoch, epochs_to_save=10, radiomics = False): # FIX epochs to save
    correct = 0
    total = 0
    validation_labels = []
    validation_predictions = []
    validation_losses = []
    validation_accuracy = []

    model.eval() # ER DETTE RIKTIG EGT? BURDE DET VÆRE model.model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
                images, labels, radiomic_feats = batch["image"].to(device), batch["label"].to(device, dtype=torch.long), batch["radiomics"].to(device)
                
                if radiomics:
                    outputs = model(images, radiomic_feats)
                else:
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
    precision = precision_score(validation_labels, validation_predictions, average=None)
    recall = recall_score(validation_labels, validation_predictions, average=None)
    return np.mean(validation_losses), np.mean(validation_accuracy), validation_predictions, precision, recall

def train_loop(model, 
               epochs: int, 
               train_dataloader: DataLoader, 
               val_dataloader: DataLoader, 
               device: torch.device,
               writer: SummaryWriter,
               radiomics: bool,
               epochs_to_save: int,
               model_name: str):

    loss_function = torch.nn.CrossEntropyLoss() # CAN USE LABEL SMOOTHING 
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01) # With regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
            
        training_loss, training_accuracy = train(model, loss_function, train_dataloader, device, optimizer, radiomics=radiomics)

        validation_loss, validation_accuracy, _, _, _ = validate(model, loss_function, val_dataloader, device, optimizer, epoch, radiomics=radiomics)

        print(f"""Epoch {epoch+1}, Average Training Loss: {training_loss},
                Average Validation Loss: {validation_loss}
                Average Accuracy: {validation_accuracy},
                Average Training Accuracy: {training_accuracy}%""")
        
        writer.add_scalar("Average training loss", training_loss, epoch)
        writer.add_scalar("Average validation loss", validation_loss, epoch)
        writer.add_scalar("Average accuracy", validation_accuracy, epoch)
        writer.add_scalar("Average Training accuracy", training_accuracy, epoch)
    
    writer.flush()

def train_model(model, dataloader, optimizer, loss_function, device, l1_lambda = 1e-3, radiomics = False):
    model.train()

    saved = False

    for batch in tqdm(dataloader):
        images, labels, noisy_label, radiomic_feats = batch["image"].to(device), batch["label"].to(device, dtype=torch.long), batch["noisy_label"].to(device, dtype=torch.float32), batch["radiomics"].to(device)
        optimizer.zero_grad()

        #Used to check the orientation of images to ensure correct permutation transform
        #Change permutation config in transforms to get it correct
        #Code for plotting generated by Chat GPT
        if not saved:
            if images.ndim == 4 :
                plt.imshow(images[0, 0, :, :].cpu().numpy(), cmap='gray')
                plt.axis('off')
                plt.savefig('control_image.png', bbox_inches='tight', pad_inches=0)
                saved = True

            if images.ndim == 5 :
                plt.imshow(images[0,0, 100, :, :].cpu().numpy(), cmap='gray')
                plt.axis('off')
                plt.savefig('control_image.png', bbox_inches='tight', pad_inches=0)
                saved == True




        if radiomics:
            outputs = model(images, radiomic_feats)
        else:
            outputs = model(images)

        if isinstance(outputs, tuple):
            outputs, features = outputs

        loss = loss_function(outputs, labels)

        # L1 REGULARIZATION
        l1_norm = sum(p.abs().sum() for p in model.fc.parameters())
        loss = loss + l1_lambda * l1_norm
        
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
                      splits: int = 10):
    
    
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

    nn_conf_matr_labels = []
    conf_matr_labels = []

    baseline_matr_pred = []
    nn_conf_matr_pred = []
    rf_conf_matr_pred = []
    svm_conf_matr_pred = []
    logreg_conf_matr_pred = []
    ensemble_conf_matr_pred = []
    et_conf_matr_pred = []

    baseline_matr_pred_rad = []
    nn_conf_matr_pred_rad = []
    rf_conf_matr_pred_rad = []
    svm_conf_matr_pred_rad = []
    logreg_conf_matr_pred_rad = []
    ensemble_conf_matr_pred_rad = []
    et_conf_matr_pred_rad = []

    train_transforms, val_transforms = transforms_selector(transforms_name)
    train_transforms_baseline, val_transforms_baseline = transforms_selector("pretrained")
    
    labels = np.array([sample["label"] for sample in dataset])

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    
        
        ## MODELS TO TEST
        model = model_selector(model_name, device)
        model_weak = model_selector(model_name, device)
        model_radiomics = model_selector("cnnweakradiomics", device)
        model_baseline = model_selector("resnet18", device)
        model_baseline_radiomics = model_selector("resnet18radiomics", device)

        loss_function = torch.nn.CrossEntropyLoss() 
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
        optimizer_weak = torch.optim.Adam(model_weak.parameters(), lr=0.001) 
        optimizer_radiomics = torch.optim.Adam(model_radiomics.parameters(), lr=0.001) 
        optimizer_baseline = torch.optim.Adam(model_baseline.parameters(), lr=0.001)
        optimizer_baseline_radiomics = torch.optim.Adam(model_baseline_radiomics.parameters(), lr=0.001)

        print(f"Fold {fold+1}/{splits}")

        train_set = [dataset[i] for i in train_idx]
        val_set = [dataset[i] for i in val_idx]
        
        train_ds = ClassificationDataset(data_list=train_set,
                                            transforms=train_transforms,
                                            radiomics=True,
                                            train=True
                                            )
    
        feature_ds = ClassificationDataset(data_list=train_set, 
                                            transforms=val_transforms,
                                            radiomics=True,
                                            train=True
                                            )

        val_ds = ClassificationDataset(data_list=val_set, 
                                            transforms=val_transforms, 
                                            radiomics=True,
                                            train=False
                                            )

        train_ds_baseline = ClassificationDataset(data_list=train_set, 
                                            start_frame=6, 
                                            end_frame=18,
                                            agg="mean", 
                                            transforms=train_transforms_baseline,
                                            radiomics=True,
                                            train=True
                                            )
        
        val_ds_baseline = ClassificationDataset(data_list=val_set, 
                                            start_frame=6, 
                                            end_frame=18,
                                            agg="mean", 
                                            transforms=val_transforms_baseline,
                                            radiomics=True,
                                            train=False
                                            )
        
        ## GET SCALER, IMPUTER AND NAN COLS FROM TRAIN SET TO VAL SET FOR RADIOMICS
        val_ds.scaler, val_ds.imputer, val_ds.nan_cols = train_ds.get_objects()
        feature_ds.scaler, feature_ds.imputer, feature_ds.nan_cols = train_ds.get_objects()

        val_ds_baseline.scaler, val_ds_baseline.imputer, val_ds_baseline.nan_cols = train_ds_baseline.get_objects()

        # INITIALIZE DATALOADERS
        train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers) 
        feature_dataloader = DataLoader(feature_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        train_dataloader_baseline = DataLoader(train_ds_baseline, batch_size=batch_size, shuffle=True, num_workers=num_workers) 
        val_dataloader_baseline = DataLoader(val_ds_baseline, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        ## TRAIN BASELINE
        print("\n TRAINING BASELINE")
        for epoch in range(10): # HVOR MANGE EPOKER?
            train_model(model_baseline, train_dataloader_baseline, optimizer_baseline, loss_function, device)
        
        ## EVAL BASELINE
        print("\n EVALUATE BASELINE")
        model_baseline.eval()
        _, accuracy_baseline, baseline_pred, precision_baseline, recall_baseline = validate(model_baseline, loss_function, val_dataloader_baseline, device, optimizer_baseline, epoch)    
        
        ## TRAIN BASELINE WITH RADIOMICS
        print("\n TRAINING BASELINE WITH RADIOMICS")
        for epoch in range(10):
            train_model(model_baseline_radiomics, train_dataloader_baseline, optimizer_baseline_radiomics, loss_function, device, radiomics=True)
        
        ## EVAL BASELINE
        print("\n EVALUATE BASELINE RADIOMICS")
        model_baseline_radiomics.eval()
        _, accuracy_baseline_radiomics, baseline_pred_rad, precision_baseline_radiomics, recall_baseline_radiomics = validate(model_baseline_radiomics, loss_function, val_dataloader_baseline, device, optimizer_baseline_radiomics, epoch, radiomics = True) 

        print("\n TRAINING 3D CNN + NN")
        for epoch in range(10):
            train_model(model, train_dataloader, optimizer, loss_function, device)

        print("\n EVALUATE 3D CNN + NN")
        model.eval()
        _, accuracy_3d_nn, nn_pred, precision_3d_nn, recall_3d_nn = validate(model, loss_function, val_dataloader, device, optimizer, epoch) 

        print("\n TRAINING 3D CNN + NN RADIOMICS")
        for epoch in range(10):
            train_model(model_radiomics, train_dataloader, optimizer_radiomics, loss_function, device, radiomics=True)

        print("\n EVALUATE 3D CNN + NN WITH RADIOMICS")
        model_radiomics.eval()
        _, accuracy_3d_nn_radiomics, nn_pred_rad, precision_3d_nn_radiomics, recall_3d_nn_radiomics = validate(model_radiomics, loss_function, val_dataloader, device, optimizer_radiomics, epoch, radiomics=True) 

        print("\n TRAINING 3D CNN")
        for epoch in range(epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{epochs}")
            X_train, y_train = [], []
            X_val, y_val = [], []
            
            X_train_radiomics, X_val_radiomics = [], []

            if epoch != epochs - 1:
                # TRAIN 3D CNN
                train_model(model_weak, train_dataloader, optimizer_weak, loss_function, device, l1_lambda=0)
            else:
                # EVAL

                print("\n EVALUATE 3D BASELINE WEAK LEARNERS")
                ## EVAL MODELS
                model_weak.eval()
                model_weak.backbone.eval()
                for param in model_weak.backbone.parameters():
                    param.requires_grad = False
                with torch.no_grad():
                    # TRAIN FEATURES
                    for batch in tqdm(feature_dataloader):
                        images, labels, noisy_label, radiomics = batch["image"].to(device), batch["label"].to(device, dtype=torch.long), batch["noisy_label"].to(device, dtype=torch.float32), batch["radiomics"]
                        outputs, features = model_weak(images)

                        features = features.detach().cpu().numpy() 
                        radiomics = radiomics.detach().cpu().numpy() if torch.is_tensor(radiomics) else radiomics 

                        X_train_radiomics.append(radiomics)
                        X_train.append(features)
                        y_train.append(labels.cpu().numpy())  
                        #X_train_radiomics.append(combined_features)
                    
                    #VAL FEATURES
                    for batch in tqdm(val_dataloader):
                        img, label, noisy_label, radiomics = batch["image"].to(device), batch["label"].to(device, dtype=torch.long), batch["noisy_label"].to(device, dtype=torch.long), batch["radiomics"]
                        outputs, features = model_weak(img)

                        features = features.detach().cpu().numpy() 
                        radiomics = radiomics.detach().cpu().numpy() if torch.is_tensor(radiomics) else radiomics 

                        X_val.append(features)  
                        y_val.append(label.cpu().numpy())  
                        X_val_radiomics.append(radiomics)
                

                X_train = np.concatenate(X_train, axis=0)  
                y_train = np.concatenate(y_train, axis=0) 
                
                X_val = np.concatenate(X_val, axis=0)  
                y_val = np.concatenate(y_val, axis=0) 

                X_train_radiomics = np.concatenate(X_train_radiomics, axis=0)  
                X_val_radiomics = np.concatenate(X_val_radiomics, axis=0)
                
                scaler = StandardScaler()

                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

                X_train_radiomics = np.concatenate([X_train, X_train_radiomics], axis=1) 
                X_val_radiomics = np.concatenate([X_val, X_val_radiomics], axis=1) 

                rf_validation_accuracy, rf_pred, svm_validation_accuracy, svm_pred, y_pred_logreg, y_pred_ensemble, logreg_validation_accuracy, et_validation_accuracy, y_pred_et, ensemble_validation_accuracy = train_models(X_train, y_train, X_val, y_val)
                
                conf_matr_labels.extend(y_val)
                baseline_matr_pred.extend(baseline_pred)
                nn_conf_matr_pred.extend(nn_pred)
                rf_conf_matr_pred.extend(rf_pred)
                svm_conf_matr_pred.extend(svm_pred)
                logreg_conf_matr_pred.extend(y_pred_logreg)
                ensemble_conf_matr_pred.extend(y_pred_ensemble)
                et_conf_matr_pred.extend(y_pred_et)


                #Eval scores
                print(f"BASELINE acc: {accuracy_baseline}")
                writer.add_scalar("BASELINE acc", accuracy_baseline, fold + 1)
                print(f"Baseline precision: {precision_baseline}")
                writer.add_text("Baseline precision", str(precision_baseline), fold + 1)
                print(f"Baseline recall: {precision_baseline}")
                writer.add_text("Baseline recall", str(recall_baseline), fold + 1)

                print("\n")
                print(f"3D CNN + NN acc: {accuracy_3d_nn}")
                writer.add_scalar("3D CNN + NN acc", accuracy_3d_nn, fold + 1)
                print(f"3D CNN + NN precision: {precision_3d_nn}")
                writer.add_text("3D CNN + NN  precision", str(precision_3d_nn), fold + 1)
                print(f"3D CNN + NN  recall: {recall_3d_nn}")
                writer.add_text("B3D CNN + NN recall", str(recall_3d_nn), fold + 1)

                print("\n")
                print(f"Logreg acc {logreg_validation_accuracy}")
                writer.add_scalar("Logreg acc", logreg_validation_accuracy, fold + 1)
                print(f"Logreg precision: {precision_score(y_val, y_pred_logreg, average=None)}")
                writer.add_text("Logreg precision", str(precision_score(y_val, y_pred_logreg, average=None)), fold + 1)
                print(f"Logreg recall: {recall_score(y_val, y_pred_logreg, average=None)}")
                writer.add_text("Logreg recall", str(recall_score(y_val, y_pred_logreg, average=None)), fold + 1)


                print("\n")
                print(f"ET acc {et_validation_accuracy}")
                writer.add_scalar("ET acc", et_validation_accuracy, fold + 1)
                print(f"ET precision: {precision_score(y_val, y_pred_et, average=None)}")
                writer.add_text("ET precision", str(precision_score(y_val, y_pred_et, average=None)), fold + 1)
                print(f"ET recall: {recall_score(y_val, y_pred_et, average=None)}")
                writer.add_text("ET recall", str(recall_score(y_val, y_pred_et, average=None)), fold + 1)

                print("\n")
                print(f"RF acc: {rf_validation_accuracy}")
                writer.add_scalar("RF acc", rf_validation_accuracy, fold + 1)
                print(f"RF precision: {precision_score(y_val, rf_pred, average=None)}")
                writer.add_text("RF precision", str(precision_score(y_val, rf_pred, average=None)), fold + 1)
                print(f"RF recall: {recall_score(y_val, rf_pred, average=None)}")
                writer.add_text("RF recall", str(recall_score(y_val, rf_pred, average=None)), fold + 1)

                print("\n")
                print(f"SVM acc: {svm_validation_accuracy}")
                writer.add_scalar("SVM acc", svm_validation_accuracy, fold + 1)
                print(f"SVM precision: {precision_score(y_val, svm_pred, average=None)}")
                writer.add_text("SVM precision", str(precision_score(y_val, svm_pred, average=None)), fold + 1)
                print(f"SVM recall: {recall_score(y_val, svm_pred, average=None)}")
                writer.add_text("SVM recall", str(recall_score(y_val, svm_pred, average=None)), fold + 1)

                print("\n")
                print(f"Ensemble acc: {ensemble_validation_accuracy}")
                writer.add_scalar("Ensemble acc", ensemble_validation_accuracy, fold + 1)
                print(f"Ensemble precision: {precision_score(y_val, y_pred_ensemble, average=None)}")
                writer.add_text("Ensemble precision", str(precision_score(y_val, y_pred_ensemble, average=None)), fold + 1)
                print(f"Ensemble recall: {recall_score(y_val, y_pred_ensemble, average=None)}")
                writer.add_text("Ensemble recall", str(recall_score(y_val, y_pred_ensemble, average=None)), fold + 1)

                rf_validation_accuracy_rad, rf_pred_rad, svm_validation_accuracy_rad, svm_pred_rad, y_pred_logreg_rad, y_pred_ensemble_rad, logreg_validation_accuracy_rad, et_validation_accuracy_rad, y_pred_et_rad, ensemble_validation_accuracy_rad = train_models(X_train_radiomics, y_train, X_val_radiomics, y_val)
                baseline_matr_pred_rad.extend(baseline_matr_pred_rad)
                nn_conf_matr_pred_rad.extend(nn_pred_rad)
                rf_conf_matr_pred_rad.extend(rf_pred_rad)
                svm_conf_matr_pred_rad.extend(svm_pred_rad)
                logreg_conf_matr_pred_rad.extend(y_pred_logreg_rad)
                ensemble_conf_matr_pred_rad.extend(y_pred_ensemble_rad)
                et_conf_matr_pred_rad.extend(y_pred_et_rad)
                
                print("\n")
                print("WITH RADIOMICS")
                print("\n")
                print(f"BASELINE RADIOMICS acc: {accuracy_baseline_radiomics}")
                writer.add_scalar("BASELINE RADIOMICS acc", accuracy_baseline_radiomics, fold + 1)
                print(f"Baseline RADIOMICS precision: {precision_baseline_radiomics}")
                writer.add_text("Baseline RADIOMICS precision", str(precision_baseline_radiomics), fold + 1)
                print(f"Baseline RADIOMICS recall: {precision_baseline_radiomics}")
                writer.add_text("Baseline RADIOMICS recall", str(recall_baseline_radiomics), fold + 1)

                print("\n")
                print(f"3D CNN + NN RADIOMICS acc: {accuracy_3d_nn_radiomics}")
                writer.add_scalar("3D CNN + NN  RADIOMICS acc", accuracy_3d_nn_radiomics, fold + 1)
                print(f"3D CNN + NN  RADIOMICS precision: {precision_3d_nn_radiomics}")
                writer.add_text("B3D CNN + NN  RADIOMICS precision", str(precision_3d_nn_radiomics), fold + 1)
                print(f"3D CNN + NN  RADIOMICS recall: {recall_3d_nn_radiomics}")
                writer.add_text("3D CNN + NN RADIOMICS recall", str(recall_3d_nn_radiomics), fold + 1)

                print("\n")
                print(f"Logreg acc rad: {logreg_validation_accuracy_rad}")
                writer.add_scalar("Logreg acc rad", logreg_validation_accuracy_rad, fold + 1)
                print(f"Logreg precision rad: {precision_score(y_val, y_pred_logreg_rad, average=None)}")
                writer.add_text("Logreg precision rad", str(precision_score(y_val, y_pred_logreg_rad, average=None)), fold + 1)
                print(f"Logreg recall rad: {recall_score(y_val, y_pred_logreg_rad, average=None)}")
                writer.add_text("Logreg recall rad", str(recall_score(y_val, y_pred_logreg_rad, average=None)), fold + 1)

                print("\n")
                print(f"ET acc rad: {et_validation_accuracy_rad}")
                writer.add_scalar("ET acc rad", et_validation_accuracy_rad, fold + 1)
                print(f"ET precision rad: {precision_score(y_val, y_pred_et_rad, average=None)}")
                writer.add_text("ET precision rad", str(precision_score(y_val, y_pred_et_rad, average=None)), fold + 1)
                print(f"ET recall rad: {recall_score(y_val, y_pred_et_rad, average=None)}")
                writer.add_text("ET recall rad", str(recall_score(y_val, y_pred_et_rad, average=None)), fold + 1)

                print("\n")
                print(f"RF acc rad: {rf_validation_accuracy_rad}")
                writer.add_scalar("RF acc rad", rf_validation_accuracy_rad, fold + 1)
                print(f"RF precision rad: {precision_score(y_val, rf_pred_rad, average=None)}")
                writer.add_text("RF precision rad", str(precision_score(y_val, rf_pred_rad, average=None)), fold + 1)
                print(f"RF recall rad: {recall_score(y_val, rf_pred_rad, average=None)}")
                writer.add_text("RF recall rad", str(recall_score(y_val, rf_pred_rad, average=None)), fold + 1)

                print("\n")
                print(f"SVM acc rad: {svm_validation_accuracy_rad}")
                writer.add_scalar("SVM acc rad", svm_validation_accuracy_rad, fold + 1)
                print(f"SVM precision rad: {precision_score(y_val, svm_pred_rad, average=None)}")
                writer.add_text("SVM precision rad", str(precision_score(y_val, svm_pred_rad, average=None)), fold + 1)
                print(f"SVM recall rad: {recall_score(y_val, svm_pred_rad, average=None)}")
                writer.add_text("SVM recall rad", str(recall_score(y_val, svm_pred_rad, average=None)), fold + 1)

                print("\n")
                print(f"Ensemble acc rad: {ensemble_validation_accuracy_rad}")
                writer.add_scalar("Ensemble acc rad", ensemble_validation_accuracy_rad, fold + 1)
                print(f"Ensemble precision rad: {precision_score(y_val, y_pred_ensemble_rad, average=None)}")
                writer.add_text("Ensemble precision rad", str(precision_score(y_val, y_pred_ensemble_rad, average=None)), fold + 1)
                print(f"Ensemble recall rad: {recall_score(y_val, y_pred_ensemble_rad, average=None)}")
                writer.add_text("Ensemble recall rad", str(recall_score(y_val, y_pred_ensemble_rad, average=None)), fold + 1)


    #Plot confusion matrix
    #Without rad features
    #baseline
    writer.add_figure("BASELINE", plot_confusion_matrix(conf_matr_labels, baseline_pred, "BASELINE"))
    writer.add_figure("3D CNN Feature Extractor + NN Classifier", plot_confusion_matrix(conf_matr_labels, nn_conf_matr_pred, "3D CNN Feature Extractor + NN Classifier"))
    writer.add_figure("3D CNN Feature Extractor + Logistic Regression Classifier", plot_confusion_matrix(conf_matr_labels, logreg_conf_matr_pred, "3D CNN Feature Extractor + Logistic Regression Classifier"))
    writer.add_figure("3D CNN Feature Extractor +  Extra Trees Classifier",  plot_confusion_matrix(conf_matr_labels, et_conf_matr_pred, "3D CNN Feature Extractor + Extra Trees Classifier"))
    writer.add_figure("3D CNN Feature Extractor + RF classifier",  plot_confusion_matrix(conf_matr_labels, rf_conf_matr_pred,  "3D CNN Feature Extractor + RF classifier"))
    writer.add_figure("3D CNN Feature Extractor + SVM classifier",  plot_confusion_matrix(conf_matr_labels, svm_conf_matr_pred,  "3D CNN Feature Extractor + SVM Classifier"))
    writer.add_figure("3D CNN Feature Extractor + Ensemble classifier",  plot_confusion_matrix(conf_matr_labels, ensemble_conf_matr_pred,  "3D CNN Feature Extractor + Ensemble Classifier"))

    #With rad features
    #baseline
    writer.add_figure("BASELINE", plot_confusion_matrix(conf_matr_labels, baseline_pred_rad, "BASELINE"))
    writer.add_figure("3D CNN Feature Extractor With Radiomic Features + NN Classifier", plot_confusion_matrix(conf_matr_labels, nn_conf_matr_pred_rad, "3D CNN Feature Extractor With Radiomic Features + NN Classifier"))
    writer.add_figure("3D CNN Feature Extractor With Radiomic Features + Logistic Regression Classifier", plot_confusion_matrix(conf_matr_labels, logreg_conf_matr_pred_rad, "3D CNN Feature Extractor With Radiomic Features + Logistic Regression Classifier"))
    writer.add_figure("3D CNN Feature Extractor With Radiomic Features +  Extra Trees Classifier",  plot_confusion_matrix(conf_matr_labels, et_conf_matr_pred_rad, "3D CNN Feature Extractor With Radiomic Features + Extra Trees Classifier"))
    writer.add_figure("3D CNN Feature Extractor With Radiomic Features + RF classifier",  plot_confusion_matrix(conf_matr_labels, rf_conf_matr_pred_rad,  "3D CNN Feature Extractor With Radiomic Features + RF classifier"))
    writer.add_figure("3D CNN Feature Extractor With Radiomic Features + SVM classifier",  plot_confusion_matrix(conf_matr_labels, svm_conf_matr_pred_rad,  "3D CNN Feature Extractor With Radiomic Features + SVM Classifier"))
    writer.add_figure("3D CNN Feature Extractor With Radiomic Features + Ensemble classifier",  plot_confusion_matrix(conf_matr_labels, ensemble_conf_matr_pred_rad,  "3D CNN Feature Extractor With Radiomic Features + Ensemble Classifier"))


                


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

    logreg_model = LogisticRegression(solver='liblinear', max_iter=500, C=0.01, random_state=42) # use elasticnet penalty
    logreg_model.fit(X_train, y_train)
    y_pred_logreg = logreg_model.predict(X_val)
    logreg_validation_accuracy = accuracy_score(y_val, y_pred_logreg)


    svm_model_prob = SVC(kernel="linear", C=0.001, probability=True, random_state=42) 

    voting_clf = VotingClassifier(
        estimators=[
            ('svc', svm_model_prob),
            ('logreg', logreg_model),
            ('rf', rf_model),
            ('et', et_model),
        ],
        voting='soft'
    )

    voting_clf.fit(X_train, y_train)
    y_pred_ensemble = voting_clf.predict(X_val)
    ensemble_validation_accuracy = accuracy_score(y_val, y_pred_ensemble)

    return rf_validation_accuracy, y_pred_rf, svm_validation_accuracy, y_pred_svm, y_pred_logreg, y_pred_ensemble, logreg_validation_accuracy, et_validation_accuracy, y_pred_et, ensemble_validation_accuracy

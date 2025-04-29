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

    model.eval()
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
    
    all_train_loss = np.zeros((splits, epochs))
    all_val_loss = np.zeros((splits, epochs))
    all_train_accuracy = np.zeros((splits, epochs))
    all_val_accuracy = np.zeros((splits, epochs))

    labels = np.array([sample["label"] for sample in dataset])
    rf_acc_per_epoch = defaultdict(list)
    rf_acc_per_epoch2 = defaultdict(list)
    rf_acc_per_epoch3 = defaultdict(list)
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(len(train_idx)) # FIX dil
        print(len(val_idx))
        train_loss = []
        val_loss = []
        train_accuracy = []
        val_accuracy = []
        #Need to reinitalize the model every time
        model = model_selector(model_name, device)
        model_baseline = model_selector("resnet18", device)

        loss_function = torch.nn.CrossEntropyLoss() # CAN USE LABEL SMOOTHING 
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01) # With regularization
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001) ## ADAM W
        optimizer_baseline = torch.optim.Adam(model.parameters(), lr=0.001)

        print(f"Fold {fold+1}/{splits}")

        train_set = [dataset[i] for i in train_idx]
        val_set = [dataset[i] for i in val_idx]
        
        train_ds = ClassificationDataset(data_list=train_set,
                                            start_frame=0,
                                            end_frame=None,
                                            agg="time_series",
                                            cache=False,
                                            transforms=train_transforms,
                                            radiomics=False,
                                            train=True
                                            )
    
        feature_ds = ClassificationDataset(data_list=train_set, 
                                            start_frame=0, 
                                            end_frame=None,
                                            agg="time_series", 
                                            cache=False,
                                            transforms=val_transforms,
                                            radiomics=False,
                                            train=True
                                            )

        val_ds = ClassificationDataset(data_list=val_set, 
                                            start_frame=0, 
                                            end_frame=None, 
                                            agg="time_series", 
                                            cache=None,
                                            transforms=val_transforms, 
                                            radiomics=False,
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

        train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers) # SHUFFLE
        feature_dataloader = DataLoader(feature_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        train_dataloader_baseline = DataLoader(train_ds_baseline, batch_size=batch_size, shuffle=False, num_workers=num_workers) # SHUFFLE
        val_dataloader_baseline = DataLoader(val_ds_baseline, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        for epoch in range(epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{epochs}")

            X_train, y_train = [], []
            X_val, y_val = [], []

            if epoch != epochs - 1:
                # TRAIN 3D CNN
                # TIME_SERIES FOR 3D CNN
                for param in model.backbone.layer4.parameters(): 
                    param.requires_grad = True
                for param in model.backbone.layer3.parameters():  
                    param.requires_grad = True
                model.backbone.train()
                train_model(model, train_dataloader, optimizer, loss_function, device)

                # TRAIN BASELINE
                # MEAN FOR 2D CNN
                #train_model(model_baseline, train_dataloader_baseline, optimizer_baseline, loss_function, device) # PASSE PÅ RIKTIG SHAPE, TRENGER IKKE TRENES LIK EPOCHS 
            else:
                # EVAL

                ## EVAL BASELINE
                #_, accuracy_baseline, _ = validate(model_baseline, loss_function, val_dataloader_baseline, device, optimizer_baseline, epoch)

                ## EVAL MODELS
                model.eval()
                model.backbone.eval()
                for param in model.backbone.parameters():
                    param.requires_grad = False
                with torch.no_grad():
                    # TRAIN FEATURES
                    for batch in tqdm(feature_dataloader):
                        images, labels, noisy_label = batch["image"].to(device), batch["label"].to(device, dtype=torch.long), batch["noisy_label"].to(device, dtype=torch.float32)
                        labels = labels - 1
                        outputs, features = model(images)
                        X_train.append(features.detach().cpu().numpy())
                        y_train.append(labels.cpu().numpy())  
                    
                    #VAL FEATURES
                    total = 0
                    correct = 0
                    for batch in tqdm(val_dataloader):
                        img, label, noisy_label = batch["image"].to(device), batch["label"].to(device), batch["noisy_label"].to(device, dtype=torch.long)
                        label = label - 1
                        outputs, features = model(img)

                        # FOR NEURAL NETWORK PREDS
                        _, predicted = torch.max(outputs.data, 1)
                        nn_conf_matr_pred.extend(predicted.tolist())
                        nn_conf_matr_labels.extend(label.tolist())

                        total += label.size(0)
                        correct += (predicted == label).sum().item()

                        X_val.append(features.detach().cpu().numpy())  
                        y_val.append(label.cpu().numpy())  
                
                neural_network_acc = correct / total

                X_train = np.concatenate(X_train, axis=0)  
                y_train = np.concatenate(y_train, axis=0) 

                X_val = np.concatenate(X_val, axis=0)  
                y_val = np.concatenate(y_val, axis=0) 

                scaler = StandardScaler()

                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

                rf_validation_accuracy, rf_pred, svm_validation_accuracy, svm_pred = train_models(X_train, y_train, X_val, y_val)

                rf_conf_matr_labels.extend(y_val)
                rf_conf_matr_pred.extend(rf_pred)

                svm_conf_matr_labels.extend(y_val)
                svm_conf_matr_pred.extend(svm_pred)

                #print(accuracy_baseline)
                print(f"NN acc: {neural_network_acc}")
                print(f"RF acc: {rf_validation_accuracy}")
                print(f"SVM acc: {svm_validation_accuracy}")

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

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_val)
    rf_validation_accuracy = accuracy_score(y_val, y_pred_rf)

    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_val)
    svm_validation_accuracy = accuracy_score(y_val, y_pred_svm)

    ensemble_svm = BaggingClassifier(
        estimator=svm,
        n_estimators=10,             
        max_samples=0.67,             
        max_features=0.1,             
        bootstrap=True,               
        bootstrap_features=True,      
        random_state=42,
        n_jobs=-1                    
    )

    return rf_validation_accuracy, y_pred_rf, svm_validation_accuracy, y_pred_svm



def train_weak(X_train, y_train, X_val, y_val, fold, writer, epoch):


    print(X_train.shape)
    print(X_val.shape)
    X_val_selected = X_val


    pca = PCA(n_components=0.99, random_state=42)  # You choose # components
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)


    svm_model = SVC(kernel="linear", C=0.1) # value of C?
    svm_model_l1 = LinearSVC(penalty="l1", C=0.1) # value of C?
    svm_model_prob = SVC(kernel="linear", C=0.1, probability=True) # value of C?
    knn_model = KNeighborsClassifier(n_neighbors=3)
    logreg_model = LogisticRegression(solver='lbfgs', max_iter=1000, C=0.1) # use elasticnet penalty

    lgbm = LGBMClassifier(
        objective='multiclass',
        num_class=5,                  
        learning_rate=0.03,           
        n_estimators=200,             
        max_depth=3,                  
        num_leaves=15,                
        subsample=0.8,                
        colsample_bytree=0.6,         
        reg_alpha=0.1,                
        reg_lambda=1.0,               
        random_state=42,
        verbosity=-1
    )

    ensemble_knn = BaggingClassifier(
        estimator=knn_model,
        n_estimators=100,              
        max_samples=0.67,              
        max_features=0.1,             
        bootstrap=True,               
        bootstrap_features=True,      
        random_state=42,
        n_jobs=-1                     
    )

    ensemble_svm_l1 = BaggingClassifier(
        estimator=svm_model_l1,
        n_estimators=100,             
        max_samples=0.67,             
        max_features=0.1,             # % of features per model (columns)
        bootstrap=True,               # Sample rows with replacement
        bootstrap_features=True,      # Sample features with replacement
        random_state=42,
        n_jobs=-1                     # Parallel training
    )

    ensemble_svm = BaggingClassifier(
        estimator=svm_model,
        n_estimators=100,             
        max_samples=0.67,             
        max_features=0.1,             
        bootstrap=True,               
        bootstrap_features=True,      
        random_state=42,
        n_jobs=-1                    
    )

    ensemble_logreg = BaggingClassifier(
        estimator=logreg_model,
        n_estimators=100,              
        max_samples=0.67,              
        max_features=0.1,             
        bootstrap=True,               
        bootstrap_features=True,      
        random_state=42,
        n_jobs=-1                     
    )
    rf_model2 = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, max_features="sqrt")
    et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    voting_clf = VotingClassifier(
        estimators=[
            ('svc', ensemble_svm),
            ('knn', ensemble_knn),
            ('logreg', ensemble_logreg),
            ('rf', rf_model2),
            ('et', et_model),
            ('xgb', xgb)    
        ],
        voting='soft'  # or 'hard'
    )

    voting_clf_no_bag = VotingClassifier(
        estimators=[
            ('svc', svm_model_prob),
            ('knn', knn_model),
            ('logreg', logreg_model),
            ('rf', rf_model2),
            ('et', et_model)    
        ],
        voting='soft'  # or 'hard'
    )

    voting_clf.fit(X_train, y_train)

    # Predict on validation/test data
    y_pred = voting_clf.predict(X_val)
    accuracy_stacked_vote = accuracy_score(y_val, y_pred)

    voting_clf_no_bag.fit(X_train, y_train)

    # Predict on validation/test data
    y_pred = voting_clf_no_bag.predict(X_val)
    accuracy_stacked_vote_no_bag = accuracy_score(y_val, y_pred)

    #stacked.fit(X_train, y_train)
    #y_pred = stacked.predict(X_val_selected)
    #accuracy_stacked = accuracy_score(y_val, y_pred)

    ensemble_svm.fit(X_train, y_train)
    ensemble_knn.fit(X_train, y_train)
    ensemble_logreg.fit(X_train, y_train)
    ensemble_svm_l1.fit(X_train, y_train)

    y_pred = ensemble_svm.predict(X_val)
    accuracy_ens_svm = accuracy_score(y_val, y_pred)

    y_pred = ensemble_knn.predict(X_val)
    accuracy_ens_knn = accuracy_score(y_val, y_pred)

    y_pred = ensemble_logreg.predict(X_val)
    accuracy_ens_logreg = accuracy_score(y_val, y_pred)

    y_pred = ensemble_logreg.predict(X_val)
    accuracy_ens_svm_l1 = accuracy_score(y_val, y_pred)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_val)
    rf_validation_accuracy = accuracy_score(y_val, y_pred)

    rf_model2 = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, max_features="sqrt")
    rf_model2.fit(X_train, y_train)
    y_pred = rf_model2.predict(X_val)
    rf_validation_accuracy2 = accuracy_score(y_val, y_pred)

    
    rf_model2 = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_model2.fit(X_train, y_train)
    y_pred = rf_model2.predict(X_val)

    class_labels = np.array([0, 1, 2, 3, 4])
    final_predictions = class_labels[np.argmin(np.abs(y_pred[:, None] - class_labels[None, :]), axis=1)]

    rf_validation_accuracy3 = accuracy_score(y_val, final_predictions)

    et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    et_model.fit(X_train, y_train)
    y_pred = et_model.predict(X_val)

    et_accuracy = accuracy_score(y_val, y_pred)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    print(f"📊 KNN Accuracy: {accuracy:.4f}")

    logreg_model = LogisticRegression(solver='lbfgs', max_iter=1000) # use elasticnet penalty
    logreg_model.fit(X_train, y_train)
    y_pred = logreg_model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)


    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_val)

    accuracy_xgb = accuracy_score(y_val, y_pred)

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_val)
    accuracy_bayes = accuracy_score(y_val, y_pred)

    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_val)
    accuracy_lightgbm = accuracy_score(y_val, y_pred)

    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_val)
    svm_validation_accuracy = accuracy_score(y_val, y_pred)

    print(f"📊 LogReg Accuracy: {accuracy:.4f}")
    print(f"📊 XGB Accuracy: {accuracy_xgb:.4f}")
    print(f"📊 Bayes Accuracy: {accuracy_bayes:.4f}")
    print(f"📊 LightGBM Accuracy: {accuracy_lightgbm:.4f}")
    print(f"RF ACCURACY: {rf_validation_accuracy}")
    print(f"RF2 ACCURACY: {rf_validation_accuracy2}")
    print(f"RF3 ACCURACY: {rf_validation_accuracy3}")
    print(f"SVM ACCURACY: {svm_validation_accuracy}")
    print(f"ET ACCURACY: {et_accuracy}")
    print(f"ACC ensemble svm {accuracy_ens_svm}")
    print(f"ACC ensemble svm l1 {accuracy_ens_svm_l1}")
    print(f"ACC ensemble knn {accuracy_ens_knn}")
    print(f"ACC ensemble logreg {accuracy_ens_logreg}")
    print(f"ACC stack no bag {accuracy_stacked_vote_no_bag}")
    print(f"ACC stack {accuracy_stacked_vote}")


    svm_model = SVC(kernel='linear', random_state=42)

    # Using BaggingClassifier with SVM as base learner
    bagging_svm = BaggingClassifier(estimator=svm_model, n_estimators=100, random_state=42)

    # Train the model
    bagging_svm.fit(X_train, y_train)

    # Predict on the test set
    y_pred_svm = bagging_svm.predict(X_val)

    # Evaluate the model
    accuracy_svm = accuracy_score(y_val, y_pred_svm)
    print(f"Accuracy with Bagging + SVM: {accuracy_svm:.4f}")

    knn_model = KNeighborsClassifier(n_neighbors=3)

    # Using BaggingClassifier with SVM as base learner
    bagging_logreg = BaggingClassifier(estimator=knn_model, n_estimators=50, random_state=42)

    # Train the model
    bagging_logreg.fit(X_train, y_train)

    # Predict on the test set
    y_pred_logreg = bagging_logreg.predict(X_val)

    # Evaluate the model
    accuracy_logreg = accuracy_score(y_val, y_pred_logreg)
    print(f"Accuracy with Bagging + KNN: {accuracy_logreg:.4f}")

    preds, models = custom_bagging_with_sampled_labels(X_train, y_train, X_val, n_estimators=50)
    
    accuracy_logreg = accuracy_score(y_val, preds)
    print(f"Accuracy with Custom: {accuracy_logreg:.4f}")
    # Make predictions (Ensemble Averaging)
    #y_preds = np.array([model.predict(X_val[:, f]) for model, f in zip(models, selected_features)])
    #y_final = np.round(y_preds.mean(axis=0))  # Majority Voting
    #accuracy = accuracy_score(y_val, y_final)
    #print("Ensemble Accuracy:", accuracy)
    #writer.add_scalars(f"Validation Accuracy/Fold {fold+1}", {
    #    "RF": rf_validation_accuracy,
    #    "SVM": svm_validation_accuracy,
    #    "ET": et_accuracy
    #}, epoch)
    return svm_validation_accuracy, rf_validation_accuracy, accuracy_stacked_vote_no_bag

def custom_bagging_with_sampled_labels(X_train, y_train, X_test, n_estimators=100, sample_std=0.3):
    # Initialize an empty list to store trained models
    models = []
    
    # Store predictions from each model
    predictions = []
    
    # Train n_estimators models
    for i in range(n_estimators):
        # Generate a bootstrap sample from the features
        bootstrap_indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_bootstrap = X_train[bootstrap_indices]
        
        # Sample labels from a normal distribution for the bootstrap sample
        # (mean = sample_mean, std = sample_std, size = len(X_bootstrap))
        y_bootstrap = np.array([np.random.normal(loc=y_train[i], scale=sample_std) for i in bootstrap_indices])

        # Train a model on the bootstrap sample
        model = SVR(kernel='linear')
        model.fit(X_bootstrap, y_bootstrap)
        models.append(model)
        
        # Store predictions from each model
        predictions.append(model.predict(X_test))
    
    # Aggregate the predictions from all the models (majority voting for classification)
    class_labels = np.array([0, 1, 2, 3, 4])
    predictions = np.array(predictions)
    averaged_preds  = np.round(predictions.mean(axis=0))  # Majority voting
    final_predictions = class_labels[np.argmin(np.abs(averaged_preds[:, None] - class_labels[None, :]), axis=1)]

    print(final_predictions)
    return final_predictions, models

def custom_bagging_random_subspace(X_train, y_train, X_test, n_models=10, sample_std=0.3):
    # Initialize an empty list to store trained models
    models = []
    
    # Store predictions from each model
    predictions = []
    
    #for i in range(n_models):
        
    # Train n_estimators models
    for i in range(n_models):
        # Generate a bootstrap sample from the features
        bootstrap_indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_bootstrap = X_train[bootstrap_indices]
        
        # Sample labels from a normal distribution for the bootstrap sample
        # (mean = sample_mean, std = sample_std, size = len(X_bootstrap))
        y_bootstrap = np.array([np.random.normal(loc=y_train[i], scale=sample_std) for i in bootstrap_indices])

        # Train a model on the bootstrap sample
        model = SVR(kernel='linear')
        model.fit(X_bootstrap, y_bootstrap)
        models.append(model)
        
        # Store predictions from each model
        predictions.append(model.predict(X_test))
    
    # Aggregate the predictions from all the models (majority voting for classification)
    class_labels = np.array([0, 1, 2, 3, 4])
    predictions = np.array(predictions)
    averaged_preds  = np.round(predictions.mean(axis=0))  # Majority voting
    final_predictions = class_labels[np.argmin(np.abs(averaged_preds[:, None] - class_labels[None, :]), axis=1)]

    print(final_predictions)
    return final_predictions, models
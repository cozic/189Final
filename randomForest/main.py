import mne
import numpy as np
import pandas as pd
from mne.decoding import CSP
from preProcess import preProcess
import matplotlib.pyplot as plt
from RF import RF
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    '''
    test
    '''
    # check if the path exists.
    participants_info = pd.read_csv('participant.tsv', sep='\t')
    participants = participants_info.participant_id.tolist()
    files = []
    paths = []
    for sub in participants:
        sub_path = 'new_dataset\\' + 'sub-001' + '\\eeg\\'
        part_files = [fn for fn in os.listdir(sub_path) if fn.endswith('set')
                      & (fn.find('task-motion_run-2') == -1) & (fn.find('task-motion_run-1') == -1) & (fn.find('task-motion_run-3') == -1) &
                      (fn.find('task-motion_run-5') == -1) & (fn.find('task-motion_run-7') == -1) & (fn.find('task-motion_run-9') == -1)&
                      (fn.find('task-motion_run-11') == -1) & (fn.find('task-motion_run-13') == -1)]
        files.append(part_files)
        paths.append(sub_path)
        break
    total_data = []
    total_label = []
    total_raw = []
    step = 0
    for i in range(len(files)):
        for file in files[i]:
            path = 'new_dataset\\' + participants[i] + '\\eeg\\'
            EEG = preProcess(path+file)
            # # filter the raw data with FFT and bandpass filter.
            # # bandwidth is Mu(8-12hz) and Beta(12-30hz).
            print(EEG.raw_data.shape)
            EEG.raw.notch_filter(freqs=[60, 79], method='fir')
            EEG.raw_data = EEG.filter(6, 30, EEG.raw_data)

            epoch, target_event, event_dict = EEG.epoch()
            # # construct CSP filter
            # CSP = CSP(epoch_data, target_event)

            if step == 0:
                total_data = epoch
                total_raw = EEG.raw
            else:
                total_data = mne.concatenate_epochs([total_data, epoch])
                total_raw = mne.concatenate_raws([total_raw, EEG.raw])
            label = EEG.labeling(event_dict, target_event)
    #         shape = np.array(epoch_data).shape
    #         epoch_data = epoch_data.reshape(shape[0], shape[1]*shape[2])
    #         epoch_data = epoch_data.reshape(epoch_data.shape[1], epoch_data.shape[0])
    #         # concatenate data altogether
            if total_label == []:
    #             total_data = epoch_data
                total_label = label
            else:
    #           total_data = np.concatenate((total_data, epoch_data), axis=1)
                total_label = total_label+label
            #print(f"event:{epoch.events[:, -1]}")
            step = 1
    # Initialize ICA object
    ica = mne.preprocessing.ICA(n_components=10, random_state=90)

    # Fit ICA on the epochs
    ica.fit(total_data)

    # Plot the independent components
    ica.plot_components()
    ica_exclude = [0, 3]  # Replace with the indices of the artifact-related components
    ica.exclude = ica_exclude

    # Apply the ICA to the epochs
    total_data = ica.apply(total_data.copy())
    # total_data = np.rot90(total_data)
    data = total_data.get_data()
    total_label = list(total_label)
    unique_event = list(set(total_label))
    total_event_dict = {event_name: i+1 for i, event_name in enumerate(unique_event)}
    label = [total_event_dict[event] for event in total_label]
    label = np.array(label)
    print(f"total label shape:{label}")
    # label = total_data.events[:, -1]
    csp = CSP(n_components=70, reg=None, log=True, norm_trace=False)
    rf = RandomForestClassifier(n_estimators=2000, random_state=42)
    clf = Pipeline([('CSP', csp), ('RF', rf)])
    cv = StratifiedKFold(n_splits=10,  shuffle=True, random_state=42)
    accuracy = []
    for train_idx, test_idx in cv.split(data, label):
        data_train, data_test = data[train_idx], data[test_idx]
        label_train, label_test = label[train_idx], label[test_idx]

        clf.fit(data_train, label_train)
        prediction = clf.predict(data_test)
        accuracy.append(accuracy_score(label_test, prediction))
    mean_accuracy = np.mean(accuracy)
    print(f"label size:{len(set(total_label))}")
    print(f"mean accuracy: {mean_accuracy:.4f}")
    #print(np.array(total_label).shape)
    #print(np.array(total_data).shape)
    # concatenate epoch data with labels
    #print(label.shape, epoch_data.shape)
    # one hot encoding
    #onehot_encoding = EEG.one_hot_encoding(label)
    #print(onehot_encoding)
    #  shuffle and training_testing dataset.
    # training, testing = EEG.shuffle_and_split(total_data, total_label, 0.8)
    # model = RF(training, testing)
    # model.train()
    # model.test()
    # print(model.loss_and_accuracy())


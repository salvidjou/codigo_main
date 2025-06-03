%% ========================================================================
%% Wavelet Time Scattering for ECG Signal Classification
%
% https://www.mathworks.com/help/wavelet/ug/ecg-signal-classification-using-wavelet-time-scattering.html
% https://www.mathworks.com/videos/ai-techniques-for-ecg-classification-part-2-ecg-classification-using-machine-learning-1611911281849.html
%% ========================================================================
clear all;
close all;
disp('Start...')

%##########################################################################
%% Read dataset
%##########################################################################
% In total, there are 96 recordings from persons with arrhythmia, 30 
% recordings from persons with congestive heart failure, and 36 recordings 
% from persons with normal sinus rhythms. The goal is to train a classifier 
% to distinguish between arrhythmia (ARR), congestive heart failure (CHF), 
% and normal sinus rhythm (NSR).

load("ECGData.mat");


%##########################################################################
%% Plotar exemplos dos sinais
%##########################################################################
% arrhythmia (ARR) - arritmia (ARR)
% congestive heart failure (CHF) - Insuficiência cardíaca congestiva (ICC)
% normal sinus rhythm (NSR) - ritmo sinusal normal (RSN)

figure (1)
tiledlayout(1,3,"TileSpacing","tight");

% RSN - linha 127
% -------------------------------------------------------------------------
nexttile(1);

plot(ECGData.Data(127,1:1000))
title('RSN')
xlabel ('Amostras')
ylabel ('Voltagem [V]')

% ARR - linha 1
% -------------------------------------------------------------------------
nexttile(2);

plot(ECGData.Data(1,1:1000))
title('ARR')
xlabel ('Amostras')
ylabel ('Voltagem [V]')

% ICC - linha 97
% -------------------------------------------------------------------------
nexttile(3);
plot(ECGData.Data(97,1:1000))
title('ICC')
xlabel ('Amostras')
ylabel ('Voltagem [V]')


%##########################################################################
%% Create Training and Test Data
%##########################################################################

% 70% train, 30% test
percent_train = 70;
[trainData,testData,trainLabels,testLabels] = ...
helperRandomSplit(percent_train,ECGData);


%##########################################################################
%% Extração de características- Wavelet Time Scattering
%##########################################################################

% Wavelet Scattering
N = size(ECGData.Data,2);
sn = waveletScattering(SignalLength=N,InvarianceScale=150, ...
    SamplingFrequency=128);

% Train set
scat_features_train = featureMatrix(sn,trainData');

Nwin = size(scat_features_train,2);
scat_features_train = permute(scat_features_train,[2 3 1]);
scat_features_train = reshape(scat_features_train, ...
    size(scat_features_train,1)*size(scat_features_train,2),[]);

% Test set
scat_features_test = featureMatrix(sn,testData');

scat_features_test = permute(scat_features_test,[2 3 1]);
scat_features_test = reshape(scat_features_test, ...
    size(scat_features_test,1)*size(scat_features_test,2),[]);

% Create labels to match the number of windows
[sequence_labels_train,sequence_labels_test] = ...
createSequenceLabels(Nwin,trainLabels,testLabels);


%##########################################################################
%% Criar dataset (table) para ML
%##########################################################################

% Train set
var = scat_features_train;
train_set = array2table(var);
train_set.Labels = categorical(sequence_labels_train,["ARR" "CHF" "NSR"],["ARR" "ICC" "RSN"]);

% Test set
var = scat_features_test;
test_set = array2table(var);
test_set.Labels = categorical(sequence_labels_test,["ARR" "CHF" "NSR"],["ARR" "ICC" "RSN"]);




%% ========================================================================
disp('End!!!')



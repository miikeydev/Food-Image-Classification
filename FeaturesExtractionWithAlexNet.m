% Part ii: AlexNet Feature Extraction & Classification

% Load pretrained AlexNet
net = alexnet;

% Prepare data
datasetPath = 'Food-11';
subsets = {'training','validation','evaluation'};
numTrain = 280; 
numVal = 80; 
numTest = 80;
imageFormats = {'.jpg','.jpeg','.png','.bmp','.gif'};
trainImages = []; 
trainLabels = [];
valImages = [];   
valLabels = [];
testImages = [];  
testLabels = [];

for i = 1:length(subsets)
    subset = subsets{i};
    subsetPath = fullfile(datasetPath, subset);
    imds = imageDatastore(subsetPath, 'IncludeSubfolders', true,...
        'FileExtensions', imageFormats, 'LabelSource','foldernames');
    classes = unique(imds.Labels);
    for c = 1:numel(classes)
        idx = find(imds.Labels == classes(c));
        if numel(idx) < (strcmp(subset, 'training')*numTrain + ...
                strcmp(subset, 'validation')*numVal + ...
                strcmp(subset, 'evaluation')*numTest)
            warning('Not enough images for class %s in subset %s.',...
                char(classes(c)), subset);
        end
        idx = idx(randperm(numel(idx)));
        switch subset
            case 'training'
                sel = idx(1:min(numTrain,numel(idx)));
                trainImages = [trainImages; imds.Files(sel)];
                trainLabels = [trainLabels; repmat(classes(c), numel(sel), 1)];
            case 'validation'
                sel = idx(1:min(numVal,numel(idx)));
                valImages = [valImages; imds.Files(sel)];
                valLabels = [valLabels; repmat(classes(c), numel(sel), 1)];
            case 'evaluation'
                sel = idx(1:min(numTest,numel(idx)));
                testImages = [testImages; imds.Files(sel)];
                testLabels = [testLabels; repmat(classes(c), numel(sel), 1)];
        end
    end
end

imdsTrain = imageDatastore(trainImages, 'Labels', trainLabels);
imdsVal   = imageDatastore(valImages,   'Labels', valLabels);
imdsTest  = imageDatastore(testImages,  'Labels', testLabels);

% Extract features from AlexNet fc7
inputSize = net.Layers(1).InputSize(1:2);
augTrain = augmentedImageDatastore(inputSize, imdsTrain);
augVal   = augmentedImageDatastore(inputSize, imdsVal);
augTest  = augmentedImageDatastore(inputSize, imdsTest);

featuresTrain = activations(net, augTrain, 'fc7', 'OutputAs','columns');
featuresVal   = activations(net, augVal,   'fc7', 'OutputAs','columns');
featuresTest  = activations(net, augTest,  'fc7', 'OutputAs','columns');

labelsTrain   = categorical(imdsTrain.Labels);
labelsVal     = categorical(imdsVal.Labels);
labelsTest    = categorical(imdsTest.Labels);

% SVM
svmModel  = fitcecoc(featuresTrain', labelsTrain);
predSVMTr = predict(svmModel, featuresTrain'); 
trainAccSVM = mean(predSVMTr == labelsTrain)*100;
predSVMVa = predict(svmModel, featuresVal');   
valAccSVM = mean(predSVMVa == labelsVal)*100;
predSVMTe = predict(svmModel, featuresTest');  
testAccSVM = mean(predSVMTe == labelsTest)*100;

% Confusion matrices (SVM)
[confMatTrSVM, ~] = confusionmat(labelsTrain, predSVMTr);
[confMatVaSVM, ~] = confusionmat(labelsVal, predSVMVa);
[confMatTeSVM, ~] = confusionmat(labelsTest, predSVMTe);

% Compute Precision / Recall / F1 for SVM (Train)
TP = diag(confMatTrSVM);
FP = sum(confMatTrSVM,1)' - TP;
FN = sum(confMatTrSVM,2) - TP;
precisionTrSVM = TP ./ (TP + FP);
recallTrSVM    = TP ./ (TP + FN);
f1TrSVM        = 2 * (precisionTrSVM .* recallTrSVM) ./ (precisionTrSVM + recallTrSVM);
trainPrecisionSVM = mean(precisionTrSVM)*100;
trainRecallSVM    = mean(recallTrSVM)*100;
trainF1SVM        = mean(f1TrSVM)*100;

% (Val)
TP = diag(confMatVaSVM);
FP = sum(confMatVaSVM,1)' - TP;
FN = sum(confMatVaSVM,2) - TP;
precisionVaSVM = TP ./ (TP + FP);
recallVaSVM    = TP ./ (TP + FN);
f1VaSVM        = 2 * (precisionVaSVM .* recallVaSVM) ./ (precisionVaSVM + recallVaSVM);
valPrecisionSVM = mean(precisionVaSVM)*100;
valRecallSVM    = mean(recallVaSVM)*100;
valF1SVM        = mean(f1VaSVM)*100;

% (Test)
TP = diag(confMatTeSVM);
FP = sum(confMatTeSVM,1)' - TP;
FN = sum(confMatTeSVM,2) - TP;
precisionTeSVM = TP ./ (TP + FP);
recallTeSVM    = TP ./ (TP + FN);
f1TeSVM        = 2 * (precisionTeSVM .* recallTeSVM) ./ (precisionTeSVM + recallTeSVM);
testPrecisionSVM = mean(precisionTeSVM)*100;
testRecallSVM    = mean(recallTeSVM)*100;
testF1SVM        = mean(f1TeSVM)*100;

% KNN
knnModel  = fitcknn(featuresTrain', labelsTrain, 'NumNeighbors',5);
predKNNTr = predict(knnModel, featuresTrain'); 
trainAccKNN = mean(predKNNTr == labelsTrain)*100;
predKNNVa = predict(knnModel, featuresVal');
valAccKNN = mean(predKNNVa == labelsVal)*100;
predKNNTe = predict(knnModel, featuresTest');
testAccKNN = mean(predKNNTe == labelsTest)*100;

% Confusion matrices (KNN)
[confMatTrKNN, ~] = confusionmat(labelsTrain, predKNNTr);
[confMatVaKNN, ~] = confusionmat(labelsVal, predKNNVa);
[confMatTeKNN, ~] = confusionmat(labelsTest, predKNNTe);

% Compute Precision / Recall / F1 for KNN (Train)
TP = diag(confMatTrKNN);
FP = sum(confMatTrKNN,1)' - TP;
FN = sum(confMatTrKNN,2) - TP;
precisionTrKNN = TP ./ (TP + FP);
recallTrKNN    = TP ./ (TP + FN);
f1TrKNN        = 2 * (precisionTrKNN .* recallTrKNN) ./ (precisionTrKNN + recallTrKNN);
trainPrecisionKNN = mean(precisionTrKNN)*100;
trainRecallKNN    = mean(recallTrKNN)*100;
trainF1KNN        = mean(f1TrKNN)*100;

% (Val)
TP = diag(confMatVaKNN);
FP = sum(confMatVaKNN,1)' - TP;
FN = sum(confMatVaKNN,2) - TP;
precisionVaKNN = TP ./ (TP + FP);
recallVaKNN    = TP ./ (TP + FN);
f1VaKNN        = 2 * (precisionVaKNN .* recallVaKNN) ./ (precisionVaKNN + recallVaKNN);
valPrecisionKNN = mean(precisionVaKNN)*100;
valRecallKNN    = mean(recallVaKNN)*100;
valF1KNN        = mean(f1VaKNN)*100;

% (Test)
TP = diag(confMatTeKNN);
FP = sum(confMatTeKNN,1)' - TP;
FN = sum(confMatTeKNN,2) - TP;
precisionTeKNN = TP ./ (TP + FP);
recallTeKNN    = TP ./ (TP + FN);
f1TeKNN        = 2 * (precisionTeKNN .* recallTeKNN) ./ (precisionTeKNN + recallTeKNN);
testPrecisionKNN = mean(precisionTeKNN)*100;
testRecallKNN    = mean(recallTeKNN)*100;
testF1KNN        = mean(f1TeKNN)*100;

% Simple NN
numClasses = numel(categories(labelsTrain));
YTrainIdx = grp2idx(labelsTrain)';
YTrainOneHot = full(ind2vec(YTrainIdx, numClasses));
netNN = patternnet([100, 50]);
netNN.trainParam.epochs = 100;
netNN.trainParam.showWindow = false;
[netNN, ~] = train(netNN, featuresTrain, YTrainOneHot);

predNNTraw = netNN(featuresTrain);
[~, iTr] = max(predNNTraw,[],1);
predNNT = categorical(iTr, 1:numClasses, categories(labelsTrain))';
trainAccNN = mean(predNNT == labelsTrain)*100;

predNNVraw = netNN(featuresVal);
[~, iVa] = max(predNNVraw,[],1);
predNNV = categorical(iVa, 1:numClasses, categories(labelsVal))';
valAccNN = mean(predNNV == labelsVal)*100;

predNNTeraw = netNN(featuresTest);
[~, iTe] = max(predNNTeraw,[],1);
predNNTe = categorical(iTe, 1:numClasses, categories(labelsTest))';
testAccNN = mean(predNNTe == labelsTest)*100;

% Confusion matrices (NN)
[confMatTrNN, ~] = confusionmat(labelsTrain, predNNT);
[confMatVaNN, ~] = confusionmat(labelsVal, predNNV);
[confMatTeNN, ~] = confusionmat(labelsTest, predNNTe);

% Compute Precision / Recall / F1 for NN (Train)
TP = diag(confMatTrNN);
FP = sum(confMatTrNN,1)' - TP;
FN = sum(confMatTrNN,2) - TP;
precisionTrNN = TP ./ (TP + FP);
recallTrNN    = TP ./ (TP + FN);
f1TrNN        = 2 * (precisionTrNN .* recallTrNN) ./ (precisionTrNN + recallTrNN);
trainPrecisionNN = mean(precisionTrNN)*100;
trainRecallNN    = mean(recallTrNN)*100;
trainF1NN        = mean(f1TrNN)*100;

% (Val)
TP = diag(confMatVaNN);
FP = sum(confMatVaNN,1)' - TP;
FN = sum(confMatVaNN,2) - TP;
precisionVaNN = TP ./ (TP + FP);
recallVaNN    = TP ./ (TP + FN);
f1VaNN        = 2 * (precisionVaNN .* recallVaNN) ./ (precisionVaNN + recallVaNN);
valPrecisionNN = mean(precisionVaNN)*100;
valRecallNN    = mean(recallVaNN)*100;
valF1NN        = mean(f1VaNN)*100;

% (Test)
TP = diag(confMatTeNN);
FP = sum(confMatTeNN,1)' - TP;
FN = sum(confMatTeNN,2) - TP;
precisionTeNN = TP ./ (TP + FP);
recallTeNN    = TP ./ (TP + FN);
f1TeNN        = 2 * (precisionTeNN .* recallTeNN) ./ (precisionTeNN + recallTeNN);
testPrecisionNN = mean(precisionTeNN)*100;
testRecallNN    = mean(recallTeNN)*100;
testF1NN        = mean(f1TeNN)*100;

% Blending by majority vote
predBlendTe = mode([predSVMTe, predKNNTe, predNNTe],2);
testAccBlend = mean(predBlendTe == labelsTest)*100;

% Confusion matrix (Blended, Test only)
[confMatTeBlend, ~] = confusionmat(labelsTest, predBlendTe);
TP = diag(confMatTeBlend);
FP = sum(confMatTeBlend,1)' - TP;
FN = sum(confMatTeBlend,2) - TP;
precisionTeBlend = TP ./ (TP + FP);
recallTeBlend    = TP ./ (TP + FN);
f1TeBlend        = 2 * (precisionTeBlend .* recallTeBlend) ./ (precisionTeBlend + recallTeBlend);
testPrecisionBlend = mean(precisionTeBlend)*100;
testRecallBlend    = mean(recallTeBlend)*100;
testF1Blend        = mean(f1TeBlend)*100;

% Prepare results for table
names      = {'SVM'; 'KNN'; 'NeuralNet'; 'Blended'};
trainAccs  = [trainAccSVM;     trainAccKNN;     trainAccNN;     NaN];
valAccs    = [valAccSVM;       valAccKNN;       valAccNN;       NaN];
testAccs   = [testAccSVM;      testAccKNN;      testAccNN;      testAccBlend];
trainPrec  = [trainPrecisionSVM; trainPrecisionKNN; trainPrecisionNN; NaN];
valPrec    = [valPrecisionSVM;   valPrecisionKNN;   valPrecisionNN;   NaN];
testPrec   = [testPrecisionSVM;  testPrecisionKNN;  testPrecisionNN;  testPrecisionBlend];
trainRec   = [trainRecallSVM;  trainRecallKNN;  trainRecallNN;  NaN];
valRec     = [valRecallSVM;    valRecallKNN;    valRecallNN;    NaN];
testRec    = [testRecallSVM;   testRecallKNN;   testRecallNN;   testRecallBlend];
trainF1s   = [trainF1SVM;      trainF1KNN;      trainF1NN;      NaN];
valF1s     = [valF1SVM;        valF1KNN;        valF1NN;        NaN];
testF1s    = [testF1SVM;       testF1KNN;       testF1NN;       testF1Blend];

% Final summary table with all metrics
T = table(names, ...
    trainAccs, valAccs, testAccs, ...
    trainPrec, valPrec, testPrec, ...
    trainRec, valRec, testRec, ...
    trainF1s, valF1s, testF1s, ...
    'VariableNames', { ...
    'Classifier', ...
    'TrainAcc', 'ValAcc', 'TestAcc', ...
    'TrainPrecision', 'ValPrecision', 'TestPrecision', ...
    'TrainRecall', 'ValRecall', 'TestRecall', ...
    'TrainF1', 'ValF1', 'TestF1'});
disp(T);

% Display a few examples
for i = 1:5
    disp(['Example ', num2str(i), ': True=', char(labelsTest(i)), ...
         ', SVM=', char(predSVMTe(i)), ...
         ', KNN=', char(predKNNTe(i)), ...
         ', NN=',  char(predNNTe(i)), ...
         ', Blend=', char(predBlendTe(i))]);
end

figure;
confusionchart(labelsTest, predBlendTe);
title('Confusion Matrix Blended Model');

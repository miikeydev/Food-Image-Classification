% Part iv.1: Feature Extraction with GoogLeNet

if ~exist('googlenet', 'file')
    error('GoogLeNet is not installed.');
end

net = googlenet;
disp('GoogLeNet loaded.');
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
    imds = imageDatastore(subsetPath, 'IncludeSubfolders', true, ...
        'FileExtensions', imageFormats, 'LabelSource','foldernames');
    classes = unique(imds.Labels);
    for c = 1:length(classes)
        class = classes(c);
        classIdx = find(imds.Labels == class);
        
        if strcmp(subset, 'training') && length(classIdx)<numTrain
            warning('Not enough training images for class %s.', char(class));
        end
        if strcmp(subset, 'validation') && length(classIdx)<numVal
            warning('Not enough validation images for class %s.', char(class));
        end
        if strcmp(subset, 'evaluation') && length(classIdx)<numTest
            warning('Not enough test images for class %s.', char(class));
        end
        
        rng('default');
        shuffledIdx = classIdx(randperm(length(classIdx)));
        
        switch subset
            case 'training'
                sel = shuffledIdx(1:min(numTrain, length(shuffledIdx)));
                trainImages = [trainImages; imds.Files(sel)];
                trainLabels = [trainLabels; repmat(class, length(sel),1)];
            case 'validation'
                sel = shuffledIdx(1:min(numVal, length(shuffledIdx)));
                valImages = [valImages; imds.Files(sel)];
                valLabels = [valLabels; repmat(class, length(sel),1)];
            case 'evaluation'
                sel = shuffledIdx(1:min(numTest, length(shuffledIdx)));
                testImages = [testImages; imds.Files(sel)];
                testLabels = [testLabels; repmat(class, length(sel),1)];
        end
    end
    fprintf('Images selected for subset: %s\n', subset);
end

imdsTrain = imageDatastore(trainImages, 'Labels', trainLabels);
imdsVal   = imageDatastore(valImages,   'Labels', valLabels);
imdsTest  = imageDatastore(testImages,  'Labels', testLabels);

inputSize = net.Layers(1).InputSize(1:2);
augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain);
augimdsVal   = augmentedImageDatastore(inputSize, imdsVal);
augimdsTest  = augmentedImageDatastore(inputSize, imdsTest);

featureLayer = 'pool5-drop_7x7_s1';
featuresTrain = activations(net, augimdsTrain, featureLayer, 'MiniBatchSize',32, 'OutputAs','columns');
featuresVal   = activations(net, augimdsVal,   featureLayer, 'MiniBatchSize',32, 'OutputAs','columns');
featuresTest  = activations(net, augimdsTest,  featureLayer, 'MiniBatchSize',32, 'OutputAs','columns');

labelsTrain = categorical(imdsTrain.Labels);
labelsVal   = categorical(imdsVal.Labels);
labelsTest  = categorical(imdsTest.Labels);

% SVM
svmModel  = fitcecoc(featuresTrain', labelsTrain);
predTrainSVM = predict(svmModel, featuresTrain');
trainAccSVM = mean(predTrainSVM == labelsTrain)*100;
predValSVM   = predict(svmModel, featuresVal');
valAccSVM    = mean(predValSVM == labelsVal)*100;
predTestSVM  = predict(svmModel, featuresTest');
testAccSVM   = mean(predTestSVM == labelsTest)*100;

[confMatTrSVM, ~] = confusionmat(labelsTrain, predTrainSVM);
[confMatValSVM, ~] = confusionmat(labelsVal, predValSVM);
[confMatTestSVM, ~] = confusionmat(labelsTest, predTestSVM);

TP = diag(confMatTrSVM);
FP = sum(confMatTrSVM,1)' - TP;
FN = sum(confMatTrSVM,2) - TP;
precisionTrSVM = TP ./ (TP + FP);
recallTrSVM    = TP ./ (TP + FN);
f1TrSVM        = 2 * (precisionTrSVM .* recallTrSVM) ./ (precisionTrSVM + recallTrSVM);
trainPrecisionSVM = mean(precisionTrSVM)*100;
trainRecallSVM    = mean(recallTrSVM)*100;
trainF1SVM        = mean(f1TrSVM)*100;

TP = diag(confMatValSVM);
FP = sum(confMatValSVM,1)' - TP;
FN = sum(confMatValSVM,2) - TP;
precisionValSVM = TP ./ (TP + FP);
recallValSVM    = TP ./ (TP + FN);
f1ValSVM        = 2 * (precisionValSVM .* recallValSVM) ./ (precisionValSVM + recallValSVM);
valPrecisionSVM = mean(precisionValSVM)*100;
valRecallSVM    = mean(recallValSVM)*100;
valF1SVM        = mean(f1ValSVM)*100;

TP = diag(confMatTestSVM);
FP = sum(confMatTestSVM,1)' - TP;
FN = sum(confMatTestSVM,2) - TP;
precisionTestSVM = TP ./ (TP + FP);
recallTestSVM    = TP ./ (TP + FN);
f1TestSVM        = 2 * (precisionTestSVM .* recallTestSVM) ./ (precisionTestSVM + recallTestSVM);
testPrecisionSVM = mean(precisionTestSVM)*100;
testRecallSVM    = mean(recallTestSVM)*100;
testF1SVM        = mean(f1TestSVM)*100;

fprintf('SVM - Train: %.2f%%, Val: %.2f%%, Test: %.2f%%\n', trainAccSVM, valAccSVM, testAccSVM);
figure;
confusionchart(labelsTest, predTestSVM);
title('Confusion Matrix SVM');

% KNN
knnModel = fitcknn(featuresTrain', labelsTrain, 'NumNeighbors',5);
predTrainKNN = predict(knnModel, featuresTrain');
trainAccKNN = mean(predTrainKNN == labelsTrain)*100;
predValKNN = predict(knnModel, featuresVal');
valAccKNN = mean(predValKNN == labelsVal)*100;
predTestKNN = predict(knnModel, featuresTest');
testAccKNN = mean(predTestKNN == labelsTest)*100;

[confMatTrKNN, ~] = confusionmat(labelsTrain, predTrainKNN);
[confMatValKNN, ~] = confusionmat(labelsVal, predValKNN);
[confMatTestKNN, ~] = confusionmat(labelsTest, predTestKNN);

TP = diag(confMatTrKNN);
FP = sum(confMatTrKNN,1)' - TP;
FN = sum(confMatTrKNN,2) - TP;
precisionTrKNN = TP ./ (TP + FP);
recallTrKNN    = TP ./ (TP + FN);
f1TrKNN        = 2 * (precisionTrKNN .* recallTrKNN) ./ (precisionTrKNN + recallTrKNN);
trainPrecisionKNN = mean(precisionTrKNN)*100;
trainRecallKNN    = mean(recallTrKNN)*100;
trainF1KNN        = mean(f1TrKNN)*100;

TP = diag(confMatValKNN);
FP = sum(confMatValKNN,1)' - TP;
FN = sum(confMatValKNN,2) - TP;
precisionValKNN = TP ./ (TP + FP);
recallValKNN    = TP ./ (TP + FN);
f1ValKNN        = 2 * (precisionValKNN .* recallValKNN) ./ (precisionValKNN + recallValKNN);
valPrecisionKNN = mean(precisionValKNN)*100;
valRecallKNN    = mean(recallValKNN)*100;
valF1KNN        = mean(f1ValKNN)*100;

TP = diag(confMatTestKNN);
FP = sum(confMatTestKNN,1)' - TP;
FN = sum(confMatTestKNN,2) - TP;
precisionTestKNN = TP ./ (TP + FP);
recallTestKNN    = TP ./ (TP + FN);
f1TestKNN        = 2 * (precisionTestKNN .* recallTestKNN) ./ (precisionTestKNN + recallTestKNN);
testPrecisionKNN = mean(precisionTestKNN)*100;
testRecallKNN    = mean(recallTestKNN)*100;
testF1KNN        = mean(f1TestKNN)*100;

fprintf('KNN - Train: %.2f%%, Val: %.2f%%, Test: %.2f%%\n', trainAccKNN, valAccKNN, testAccKNN);
figure;
confusionchart(labelsTest, predTestKNN);
title('Confusion Matrix KNN');

% Simple NN
numClasses = numel(categories(labelsTrain));
YTrainIdx = grp2idx(labelsTrain)';
YTrainOneHot = full(ind2vec(YTrainIdx, numClasses));
netNN = patternnet([100, 50]);
netNN.trainParam.epochs = 100;
netNN.trainParam.showWindow = false;
[netNN, ~] = train(netNN, featuresTrain, YTrainOneHot);

predNNTraw = netNN(featuresTrain);
[~, iTr] = max(predNNTraw, [], 1);
predNNT = categorical(iTr, 1:numClasses, categories(labelsTrain))';
trainAccNN = mean(predNNT == labelsTrain)*100;

predNNVraw = netNN(featuresVal);
[~, iVa] = max(predNNVraw, [], 1);
predNNV = categorical(iVa, 1:numClasses, categories(labelsVal))';
valAccNN = mean(predNNV == labelsVal)*100;

predNNTeraw = netNN(featuresTest);
[~, iTe] = max(predNNTeraw, [], 1);
predNNTe = categorical(iTe, 1:numClasses, categories(labelsTest))';
testAccNN = mean(predNNTe == labelsTest)*100;

[confMatTrNN, ~] = confusionmat(labelsTrain, predNNT);
[confMatValNN, ~] = confusionmat(labelsVal, predNNV);
[confMatTestNN, ~] = confusionmat(labelsTest, predNNTe);

TP = diag(confMatTrNN);
FP = sum(confMatTrNN,1)' - TP;
FN = sum(confMatTrNN,2) - TP;
precisionTrNN = TP ./ (TP + FP);
recallTrNN    = TP ./ (TP + FN);
f1TrNN        = 2 * (precisionTrNN .* recallTrNN) ./ (precisionTrNN + recallTrNN);
trainPrecisionNN = mean(precisionTrNN)*100;
trainRecallNN    = mean(recallTrNN)*100;
trainF1NN        = mean(f1TrNN)*100;

TP = diag(confMatValNN);
FP = sum(confMatValNN,1)' - TP;
FN = sum(confMatValNN,2) - TP;
precisionValNN = TP ./ (TP + FP);
recallValNN    = TP ./ (TP + FN);
f1ValNN        = 2 * (precisionValNN .* recallValNN) ./ (precisionValNN + recallValNN);
valPrecisionNN = mean(precisionValNN)*100;
valRecallNN    = mean(recallValNN)*100;
valF1NN        = mean(f1ValNN)*100;

TP = diag(confMatTestNN);
FP = sum(confMatTestNN,1)' - TP;
FN = sum(confMatTestNN,2) - TP;
precisionTestNN = TP ./ (TP + FP);
recallTestNN    = TP ./ (TP + FN);
f1TestNN        = 2 * (precisionTestNN .* recallTestNN) ./ (precisionTestNN + recallTestNN);
testPrecisionNN = mean(precisionTestNN)*100;
testRecallNN    = mean(recallTestNN)*100;
testF1NN        = mean(f1TestNN)*100;

fprintf('NeuralNet - Train: %.2f%%, Val: %.2f%%, Test: %.2f%%\n', trainAccNN, valAccNN, testAccNN);
figure;
confusionchart(labelsTest, predNNTe);
title('Confusion Matrix NN');

% Blended
predBlendTe = mode([predTestSVM, predTestKNN, predNNTe],2);
testAccBlend = mean(predBlendTe == labelsTest)*100;

[confMatTestBlend, ~] = confusionmat(labelsTest, predBlendTe);
TP = diag(confMatTestBlend);
FP = sum(confMatTestBlend,1)' - TP;
FN = sum(confMatTestBlend,2) - TP;
precisionTestBlend = TP ./ (TP + FP);
recallTestBlend    = TP ./ (TP + FN);
f1TestBlend        = 2 * (precisionTestBlend .* recallTestBlend) ./ (precisionTestBlend + recallTestBlend);
testPrecisionBlend = mean(precisionTestBlend)*100;
testRecallBlend    = mean(recallTestBlend)*100;
testF1Blend        = mean(f1TestBlend)*100;

names = {'SVM'; 'KNN'; 'NeuralNet'; 'Blended'};
trainA = [trainAccSVM;     trainAccKNN;     trainAccNN;     NaN];
valA   = [valAccSVM;       valAccKNN;       valAccNN;       NaN];
testA  = [testAccSVM;      testAccKNN;      testAccNN;      testAccBlend];

trainPrec = [trainPrecisionSVM; trainPrecisionKNN; trainPrecisionNN; NaN];
valPrec   = [valPrecisionSVM;   valPrecisionKNN;   valPrecisionNN;   NaN];
testPrec  = [testPrecisionSVM;  testPrecisionKNN;  testPrecisionNN;  testPrecisionBlend];

trainRec  = [trainRecallSVM;  trainRecallKNN;  trainRecallNN;  NaN];
valRec    = [valRecallSVM;    valRecallKNN;    valRecallNN;    NaN];
testRec   = [testRecallSVM;   testRecallKNN;   testRecallNN;   testRecallBlend];

trainF1s  = [trainF1SVM;      trainF1KNN;      trainF1NN;      NaN];
valF1s    = [valF1SVM;        valF1KNN;        valF1NN;        NaN];
testF1s   = [testF1SVM;       testF1KNN;       testF1NN;       testF1Blend];

disp('----- Summary of Classifier Accuracies and Metrics -----');
T = table(names, ...
    trainA, valA, testA, ...
    trainPrec, valPrec, testPrec, ...
    trainRec, valRec, testRec, ...
    trainF1s, valF1s, testF1s, ...
    'VariableNames', { ...
    'Classifier', ...
    'TrainAcc','ValAcc','TestAcc', ...
    'TrainPrecision','ValPrecision','TestPrecision', ...
    'TrainRecall','ValRecall','TestRecall', ...
    'TrainF1','ValF1','TestF1'});
disp(T);

for i = 1:5
    fprintf('Example %d - True: %s, SVM: %s, KNN: %s, NN: %s, Blend: %s\n',...
        i, char(labelsTest(i)), char(predTestSVM(i)), ...
        char(predTestKNN(i)), char(predNNTe(i)), char(predBlendTe(i)));
end

figure;
confusionchart(labelsTest, predBlendTe);
title('Confusion Matrix Blended Model');

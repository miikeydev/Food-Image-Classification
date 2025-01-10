% Part iii: Transfer Learning with AlexNet

if ~exist('alexnet', 'file')
    error('AlexNet is not installed.');
end

net = alexnet;
inputSize = net.Layers(1).InputSize(1:2);

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
    for c = 1:numel(classes)
        class = classes(c);
        classIdx = find(imds.Labels == class);
        if strcmp(subset, 'training')
            sel = classIdx(randperm(length(classIdx), ...
                min(numTrain, length(classIdx))));
            trainImages = [trainImages; imds.Files(sel)];
            trainLabels = [trainLabels; repmat(class, length(sel), 1)];
        elseif strcmp(subset, 'validation')
            sel = classIdx(randperm(length(classIdx), ...
                min(numVal, length(classIdx))));
            valImages = [valImages; imds.Files(sel)];
            valLabels = [valLabels; repmat(class, length(sel), 1)];
        else
            sel = classIdx(randperm(length(classIdx), ...
                min(numTest, length(classIdx))));
            testImages = [testImages; imds.Files(sel)];
            testLabels = [testLabels; repmat(class, length(sel), 1)];
        end
    end
end

imdsTrain = imageDatastore(trainImages, 'Labels', trainLabels);
imdsVal   = imageDatastore(valImages,   'Labels', valLabels);
imdsTest  = imageDatastore(testImages,  'Labels', testLabels);

augTrain = augmentedImageDatastore(inputSize, imdsTrain);
augVal   = augmentedImageDatastore(inputSize, imdsVal);
augTest  = augmentedImageDatastore(inputSize, imdsTest);

numClasses = numel(categories(imdsTrain.Labels));
layers = net.Layers;
layers(end-2) = fullyConnectedLayer(numClasses, 'Name', 'fc_custom');
layers(end) = classificationLayer('Name', 'output');

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

netTransfer = trainNetwork(augTrain, layers, options);

predTrain = classify(netTransfer, augTrain);
trainAcc = mean(predTrain == imdsTrain.Labels)*100;

predVal = classify(netTransfer, augVal);
valAcc = mean(predVal == imdsVal.Labels)*100;

predTest = classify(netTransfer, augTest);
testAcc = mean(predTest == imdsTest.Labels)*100;

% Confusion matrices
[confMatTr, ~] = confusionmat(imdsTrain.Labels, predTrain);
[confMatVal, ~] = confusionmat(imdsVal.Labels, predVal);
[confMatTest, ~] = confusionmat(imdsTest.Labels, predTest);

% Compute Precision / Recall / F1 for Train
TP = diag(confMatTr);
FP = sum(confMatTr,1)' - TP;
FN = sum(confMatTr,2) - TP;
precisionTr = TP ./ (TP + FP);
recallTr    = TP ./ (TP + FN);
f1Tr        = 2 * (precisionTr .* recallTr) ./ (precisionTr + recallTr);
trainPrecision = mean(precisionTr)*100;
trainRecall    = mean(recallTr)*100;
trainF1        = mean(f1Tr)*100;

% Validation
TP = diag(confMatVal);
FP = sum(confMatVal,1)' - TP;
FN = sum(confMatVal,2) - TP;
precisionVal = TP ./ (TP + FP);
recallVal    = TP ./ (TP + FN);
f1Val        = 2 * (precisionVal .* recallVal) ./ (precisionVal + recallVal);
valPrecision = mean(precisionVal)*100;
valRecall    = mean(recallVal)*100;
valF1        = mean(f1Val)*100;

% Test
TP = diag(confMatTest);
FP = sum(confMatTest,1)' - TP;
FN = sum(confMatTest,2) - TP;
precisionTest = TP ./ (TP + FP);
recallTest    = TP ./ (TP + FN);
f1Test        = 2 * (precisionTest .* recallTest) ./ (precisionTest + recallTest);
testPrecision = mean(precisionTest)*100;
testRecall    = mean(recallTest)*100;
testF1        = mean(f1Test)*100;

% Create summary table
names         = {'Transfer_AlexNet'};
trainAccs     = trainAcc;
valAccs       = valAcc;
testAccs      = testAcc;
trainPrec     = trainPrecision;
valPrec       = valPrecision;
testPrec      = testPrecision;
trainRec      = trainRecall;
valRec        = valRecall;
testRec       = testRecall;
trainF1s      = trainF1;
valF1s        = valF1;
testF1s       = testF1;

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

fprintf('Accuracy - Train: %.2f%%, Val: %.2f%%, Test: %.2f%%\n',...
    trainAcc, valAcc, testAcc);

figure;
confusionchart(imdsTest.Labels, predTest);
title('Confusion Matrix - Test Set');

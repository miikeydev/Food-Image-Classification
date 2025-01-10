% Part iv.2: Transfer Learning with GoogLeNet

% Check if GoogLeNet is installed
if ~exist('googlenet', 'file')
    error('GoogLeNet is not installed.');
end

% Load pretrained GoogLeNet
net = googlenet;
inputSize = net.Layers(1).InputSize(1:2);
disp('GoogLeNet loaded successfully.');

% Define dataset parameters
datasetPath = 'Food-11';
subsets = {'training', 'validation', 'evaluation'};
numTrain = 280; 
numVal = 80;    
numTest = 80;   
imageFormats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'};
trainImages = [];
trainLabels = [];
valImages = [];
valLabels = [];
testImages = [];
testLabels = [];

% Prepare image data
for i = 1:length(subsets)
    subset = subsets{i};
    subsetPath = fullfile(datasetPath, subset);
    imds = imageDatastore(subsetPath, 'IncludeSubfolders', true,...
        'FileExtensions', imageFormats, 'LabelSource','foldernames');
    classes = unique(imds.Labels);
    for c = 1:length(classes)
        class = classes(c);
        classIdx = find(imds.Labels == class);
        if strcmp(subset, 'training') && length(classIdx) < numTrain
            warning('Not enough training images for class %s. Requested: %d, Found: %d',...
                char(class), numTrain, length(classIdx));
        end
        if strcmp(subset, 'validation') && length(classIdx) < numVal
            warning('Not enough validation images for class %s. Requested: %d, Found: %d',...
                char(class), numVal, length(classIdx));
        end
        if strcmp(subset, 'evaluation') && length(classIdx) < numTest
            warning('Not enough test images for class %s. Requested: %d, Found: %d',...
                char(class), numTest, length(classIdx));
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

% Create image datastores
imdsTrain = imageDatastore(trainImages, 'Labels', trainLabels,...
    'ReadFcn', @(filename) imread(filename));
imdsVal = imageDatastore(valImages, 'Labels', valLabels,...
    'ReadFcn', @(filename) imread(filename));
imdsTest = imageDatastore(testImages, 'Labels', testLabels,...
    'ReadFcn', @(filename) imread(filename));

% Augment images
augTrain = augmentedImageDatastore(inputSize, imdsTrain);
augVal   = augmentedImageDatastore(inputSize, imdsVal);
augTest  = augmentedImageDatastore(inputSize, imdsTest);

% Modify the layer graph to replace only the main classifier
lgraph = layerGraph(net);
numClasses = numel(categories(imdsTrain.Labels));

% Replace main classifier layers
newFCLayer = fullyConnectedLayer(numClasses, 'Name', 'fc_custom');
newClassLayer = classificationLayer('Name', 'output');
lgraph = replaceLayer(lgraph, 'loss3-classifier', newFCLayer);
lgraph = replaceLayer(lgraph, 'output', newClassLayer);

% Define training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
netTransfer = trainNetwork(augTrain, lgraph, options);

% Classify images
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

% Compute Precision / Recall / F1 for Validation
TP = diag(confMatVal);
FP = sum(confMatVal,1)' - TP;
FN = sum(confMatVal,2) - TP;
precisionVal = TP ./ (TP + FP);
recallVal    = TP ./ (TP + FN);
f1Val        = 2 * (precisionVal .* recallVal) ./ (precisionVal + recallVal);
valPrecision = mean(precisionVal)*100;
valRecall    = mean(recallVal)*100;
valF1        = mean(f1Val)*100;

% Compute Precision / Recall / F1 for Test
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
names         = {'Transfer_GoogLeNet'};
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


% Display confusion matrix for Test set
figure;
confusionchart(imdsTest.Labels, predTest);
title('Confusion Matrix - Test Set');

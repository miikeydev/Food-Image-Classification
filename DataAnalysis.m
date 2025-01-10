% Part i: Data Exploration on the Food-11 Dataset

% Clear workspace and command window for a fresh start
clear;
clc;

% Define paths and subsets
datasetPath = 'Food-11'; 
subsets = {'training', 'validation', 'evaluation'};
imageFormats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'};

% Initialize variables for metrics
classNames = [];
counts = struct();
brightnessTotal = [];
contrastTotal = [];
sharpnessTotal = [];
counts_total = [];
redDominance = [];
greenDominance = [];
blueDominance = [];
imageSizes = [];

% Initialize image counts for each subset
for i = 1:length(subsets)
    subset = subsets{i};
    counts.(subset) = [];
end

% Count images per class and set up metrics
for i = 1:length(subsets)
    subset = subsets{i};
    subsetPath = fullfile(datasetPath, subset);
    imds = imageDatastore(subsetPath, 'IncludeSubfolders', true, ...
        'FileExtensions', imageFormats, 'LabelSource', 'foldernames');
    tbl = countEachLabel(imds);
    counts.(subset) = tbl;
    
    if isempty(classNames)
        classNames = string(tbl.Label);
        numClasses = length(classNames);
        brightnessTotal = zeros(numClasses,1);
        contrastTotal = zeros(numClasses,1);
        sharpnessTotal = zeros(numClasses,1);
        counts_total = zeros(numClasses,1);
        redDominance = zeros(numClasses,1);
        greenDominance = zeros(numClasses,1);
        blueDominance = zeros(numClasses,1);
    end
end

% Display one sample image per class with enhanced figure size and layout
figure('Name', 'Sample Images per Class', 'NumberTitle', 'off', 'Position', [100, 100, 1600, 1200]);

numSamplesPerClass = 1; % Display one image per class
numRows = ceil(numClasses / 5); % Define number of rows based on classes and desired columns
numCols = 5; % Define number of columns (adjust as needed)

% Create a tiled layout for better control over subplot arrangement
tiledlayout(numRows, numCols, 'TileSpacing', 'compact', 'Padding', 'compact');

for c = 1:length(classNames)
    className = classNames(c);
    classPath = fullfile(datasetPath, 'training', char(className));
    classImds = imageDatastore(classPath, 'FileExtensions', imageFormats, ...
        'LabelSource', 'foldernames');
    
    for n = 1:numSamplesPerClass
        if hasdata(classImds)
            imgSample = read(classImds);
            nexttile
            imshow(imgSample);
            title(sprintf('%s', className), 'Interpreter', 'none', 'FontSize', 12);
        else
            nexttile
            text(0.5, 0.5, 'No Image', 'HorizontalAlignment', 'center', 'FontSize', 12);
            title(sprintf('%s', className), 'Interpreter', 'none', 'FontSize', 12);
        end
    end
end

% Add a super title to the entire figure for context
sgtitle('Sample Images from Food-11 Dataset (1 per Class)', 'FontSize', 20);

% Process each subset to compute image quality metrics
for i = 1:length(subsets)
    subset = subsets{i};
    subsetPath = fullfile(datasetPath, subset);
    imds = imageDatastore(subsetPath, 'IncludeSubfolders', true, ...
        'FileExtensions', imageFormats, 'LabelSource','foldernames');
    
    fprintf('Processing subset: %s\n', subset);
    while hasdata(imds)
        [img, info] = read(imds);
        label = string(info.Label);
        classIdx = find(strcmp(classNames, label));
        if isempty(classIdx)
            fprintf('Warning: Class not found for label: %s\n', label);
            continue;
        end
        grayImg = rgb2gray(img);
        brightness = mean(grayImg(:));
        contrast = std(double(grayImg(:)));
        laplaceFilter = [0 1 0; 1 -4 1; 0 1 0];
        laplacian = imfilter(double(grayImg), laplaceFilter, 'replicate');
        sharpness = var(laplacian(:));
        brightnessTotal(classIdx) = brightnessTotal(classIdx) + brightness;
        contrastTotal(classIdx) = contrastTotal(classIdx) + contrast;
        sharpnessTotal(classIdx) = sharpnessTotal(classIdx) + sharpness;
        counts_total(classIdx) = counts_total(classIdx) + 1;
        
        % Calculate color dominance
        R = double(img(:,:,1));
        G = double(img(:,:,2));
        B = double(img(:,:,3));
        redPixels = (R > G) & (R > B);
        greenPixels = (G > R) & (G > B);
        bluePixels = (B > R) & (B > G);
        redCount = sum(redPixels(:));
        greenCount = sum(greenPixels(:));
        blueCount = sum(bluePixels(:));
        redDominance(classIdx) = redDominance(classIdx) + redCount;
        greenDominance(classIdx) = greenDominance(classIdx) + greenCount;
        blueDominance(classIdx) = blueDominance(classIdx) + blueCount;
        
        % Record image size
        [height, width, ~] = size(img);
        imageSizes = [imageSizes; height, width];
    end
    fprintf('Finished processing subset: %s\n\n', subset);
end

% Calculate average metrics per class
brightnessMean = brightnessTotal ./ counts_total;
contrastMean = contrastTotal ./ counts_total;
sharpnessMean = sharpnessTotal ./ counts_total;

% Determine dominant color per class
totalDominance = redDominance + greenDominance + blueDominance;
dominantPercentages = zeros(length(classNames),1);
dominantColors = strings(length(classNames),1);
for c = 1:length(classNames)
    if totalDominance(c) == 0
        dominantPercentages(c) = 0;
        dominantColors(c) = 'None';
    else
        [maxVal, maxIdx] = max([redDominance(c), greenDominance(c), blueDominance(c)]);
        dominantPercentages(c) = (maxVal / totalDominance(c)) * 100;
        switch maxIdx
            case 1
                dominantColors(c) = 'Red';
            case 2
                dominantColors(c) = 'Green';
            case 3
                dominantColors(c) = 'Blue';
        end
    end
end

% Analyze image sizes
totalPixels = imageSizes(:,1) .* imageSizes(:,2);
[minTotalPixels, minIdx] = min(totalPixels);
minImageSize = imageSizes(minIdx,:);
[maxTotalPixels, maxIdx] = max(totalPixels);
maxImageSize = imageSizes(maxIdx,:);
combinedSizes = strcat(string(imageSizes(:,1)), 'x', string(imageSizes(:,2)));
combinedSizesCat = categorical(combinedSizes);
mostCommonSize = char(mode(combinedSizesCat));

% Display dominance counts per class
fprintf('\n----- Dominance Counts per Class -----\n');
for c = 1:length(classNames)
    cl = char(classNames(c));
    fprintf('Class: %s, R: %d, G: %d, B: %d, Total: %d\n', ...
        cl, redDominance(c), greenDominance(c), blueDominance(c), totalDominance(c));
end

% Display image size statistics
fprintf('\n----- Image Size Statistics -----\n');
fprintf('Min Size: %dx%d pixels\n', minImageSize(1), minImageSize(2));
fprintf('Max Size: %dx%d pixels\n', maxImageSize(1), maxImageSize(2));
fprintf('Most Common Size: %s\n', mostCommonSize);

% Plot Class Distribution Across Subsets
figure('Name', 'Class Distribution Across Subsets', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 800]);
hold on;
bar(counts.training.Count, 'FaceColor', 'b', 'BarWidth', 0.5);
bar(counts.validation.Count, 'FaceColor', 'r', 'BarWidth', 0.5, 'FaceAlpha', 0.5);
bar(counts.evaluation.Count, 'FaceColor', 'g', 'BarWidth', 0.5, 'FaceAlpha', 0.5);
hold off;
legend(subsets, 'Location', 'northeast');
set(gca, 'XTickLabel', classNames, 'XTick', 1:length(classNames));
xlabel('Food Classes', 'FontSize', 14);
ylabel('Number of Images', 'FontSize', 14);
title('Class Distribution Across Subsets', 'FontSize', 16);
xtickangle(45);
grid on;

% Plot Average Brightness per Class
figure('Name', 'Average Brightness per Class', 'NumberTitle', 'off', 'Position', [150, 150, 1200, 800]);
bar(brightnessMean, 'FaceColor', [0.2 0.2 0.5]);
set(gca, 'XTickLabel', classNames, 'XTick', 1:length(classNames));
xlabel('Food Classes', 'FontSize', 14);
ylabel('Average Brightness', 'FontSize', 14);
title('Average Brightness per Class', 'FontSize', 16);
xtickangle(45);
grid on;

% Plot Average Contrast per Class
figure('Name', 'Average Contrast per Class', 'NumberTitle', 'off', 'Position', [200, 200, 1200, 800]);
bar(contrastMean, 'FaceColor', [0.5 0.2 0.2]);
set(gca, 'XTickLabel', classNames, 'XTick', 1:length(classNames));
xlabel('Food Classes', 'FontSize', 14);
ylabel('Average Contrast', 'FontSize', 14);
title('Average Contrast per Class', 'FontSize', 16);
xtickangle(45);
grid on;

% Plot Average Sharpness per Class
figure('Name', 'Average Sharpness per Class', 'NumberTitle', 'off', 'Position', [250, 250, 1200, 800]);
bar(sharpnessMean, 'FaceColor', [0.2 0.5 0.2]);
set(gca, 'XTickLabel', classNames, 'XTick', 1:length(classNames));
xlabel('Food Classes', 'FontSize', 14);
ylabel('Average Sharpness', 'FontSize', 14);
title('Average Sharpness per Class', 'FontSize', 16);
xtickangle(45);
grid on;

% Plot Dominant Color Percentage per Class
figure('Name', 'Dominant Color per Class', 'NumberTitle', 'off', 'Position', [300, 300, 1200, 800]);
hold on;
for c = 1:length(classNames)
    switch dominantColors(c)
        case 'Red'
            barColor = [1 0 0];
        case 'Green'
            barColor = [0 1 0];
        case 'Blue'
            barColor = [0 0 1];
        otherwise
            barColor = [0.5 0.5 0.5];
    end
    bar(c, dominantPercentages(c), 'FaceColor', barColor, 'EdgeColor', 'k');
end
hold off;
set(gca, 'XTickLabel', classNames, 'XTick', 1:length(classNames));
xlabel('Food Classes', 'FontSize', 14);
ylabel('Dominant Color Percentage (%)', 'FontSize', 14);
title('Dominant Color per Class', 'FontSize', 16);
xtickangle(45);
grid on;

% Display Image Quality Metrics per Class
fprintf('\n----- Image Quality Metrics per Class -----\n');
fprintf('%-20s %-15s %-15s %-15s\n', 'Class', 'Brightness', 'Contrast', 'Sharpness');
for c = 1:length(classNames)
    fprintf('%-20s %-15.2f %-15.2f %-15.2f\n', ...
        char(classNames(c)), brightnessMean(c), contrastMean(c), sharpnessMean(c));
end

% Display Class Distribution Across Subsets
fprintf('\n----- Class Distribution Across Subsets -----\n');
fprintf('%-20s %-15s %-15s %-15s\n', 'Class', 'Training', 'Validation', 'Evaluation');
for c = 1:length(classNames)
    cl = char(classNames(c));
    trn = counts.training.Count(c);
    val = counts.validation.Count(c);
    evals = counts.evaluation.Count(c);
    fprintf('%-20s %-15d %-15d %-15d\n', cl, trn, val, evals);
end

% Display Dominant Colors per Class
fprintf('\n----- Dominant Colors per Class -----\n');
fprintf('%-20s %-10s %-20s\n', 'Class', 'Color', 'Percentage (%)');
for c = 1:length(classNames)
    fprintf('%-20s %-10s %-20.2f\n', ...
        char(classNames(c)), dominantColors(c), dominantPercentages(c));
end

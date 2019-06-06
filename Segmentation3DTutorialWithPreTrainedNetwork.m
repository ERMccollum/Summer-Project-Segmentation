%DOWNLOAD PRETRAINED NETWORK AND SAMPLE TEST SET
trained3DUnet_url = 'https://www.mathworks.com/supportfiles/vision/data/brainTumor3DUNet.mat';
sampleData_url = 'https://www.mathworks.com/supportfiles/vision/data/sampleBraTSTestSet.tar.gz';

imageDir = fullfile(tempdir,'BraTS');
if ~exist(imageDir,'dir')
    mkdir(imageDir);
end
downloadTrained3DUnetSampleData(trained3DUnet_url,sampleData_url,imageDir);

%TRAIN THE NETWORK
    %if using pretrained network, keep doTraining = false
doTraining = false;
if doTraining
    modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');
    [net,info] = trainNetwork(dsTrain,lgraph,options);
    save(['trained3DUNet-' modelDateTime '-Epoch-' num2str(maxEpochs) '.mat'],'net');
else
    load(fullfile(imageDir,'trained3DUNet','brainTumor3DUNet.mat'));
end

%PERFORMING SEGMENTATION OF DATA
    %if usefulltest set is false, code only processes 5 test volume
    %if true, uses all 55 test images
useFullTestSet = false;
if useFullTestSet
    volLocTest = fullfile(preprocessDataLoc,'imagesTest');
    lblLocTest = fullfile(preprocessDataLoc,'labelsTest');
else
    volLocTest = fullfile(imageDir,'sampleBraTSTestSet','imagesTest');
    lblLocTest = fullfile(imageDir,'sampleBraTSTestSet','labelsTest');
    classNames = ["background","tumor"];
    pixelLabelID = [0 1];
end

    %Crop the central portion of the images and labels to size 128-by-128-by-128 voxels by using the helper function centerCropMatReader.
    %This function is attached to the example as a supporting file. 
    %The voldsTest variable stores the ground truth test images. 
    %The pxdsTest variable stores the ground truth labels.
windowSize = [128 128 128];
volReader = @(x) centerCropMatReader(x,windowSize);
labelReader = @(x) centerCropMatReader(x,windowSize);
voldsTest = imageDatastore(volLocTest, ...
    'FileExtensions','.mat','ReadFcn',volReader);
pxdsTest = pixelLabelDatastore(lblLocTest,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',labelReader);


%For each test image, add the ground truth image volumes and labels to cell arrays. 
%Use the trained network with the semanticseg function to predict the labels for each test volume.

%After performing the segmentation, postprocess the predicted labels by labeling nonbrain voxels as 1, corresponding to the background. 
%Use the test images to determine which voxels do not belong to the brain. 
%You can also clean up the predicted labels by removing islands and filling holes using the medfilt3 function.
%medfilt3 does not support categorical data, so cast the pixel label IDs to uint8 before the calculation. 
%Then, cast the filtered labels back to the categorical data type, specifying the original pixel label IDs and class names.
% 
% id=1;
% while hasdata(voldsTest)
%     disp(['Processing test volume ' num2str(id)])
%     
%     groundTruthLabels{id} = read(pxdsTest);
%     
%     vol{id} = read(voldsTest);
%     tempSeg = semanticseg(vol{id},net);
% 
%     % Get the non-brain region mask from the test image.
%     volMask = vol{id}(:,:,:,1)==0;
%     % Set the non-brain region of the predicted label as background.
%     tempSeg(volMask) = classNames(1);
%     % Perform median filtering on the predicted label.
%     tempSeg = medfilt3(uint8(tempSeg)-1);
%     % Cast the filtered label to categorial.
%     tempSeg = categorical(tempSeg,pixelLabelID,classNames);
%     predictedLabels{id} = tempSeg;
%     id=id+1;
% end

 predictedLabels



%COMPARING GROUND TRUTH AGAINST NETWORK PREDICTION
    %select one of the test images to evaluate the accuracy segmentation
    %extract 4d data and store this 3d volume in vol3d variable
volId = 2;
vol3d = vol{volId}(:,:,:,1);


%displays in montage ground truth and predicted labels along depth
%direction
zID = size(vol3d,3)/2;
zSliceGT = labeloverlay(vol3d(:,:,zID),groundTruthLabels{volId}(:,:,zID));
zSlicePred = labeloverlay(vol3d(:,:,zID),predictedLabels{volId}(:,:,zID));

figure(1)
title('Labeled Ground Truth (Left) vs. Network Prediction (Right)')
montage({zSliceGT;zSlicePred},'Size',[1 2],'BorderSize',5)



%display ground truth labeled volume using labelvol function

%third line makes background transparent

%to make some brain voxels transparent, specify volume threshold as a
%number in the range of [0,1]

%All normalized volume intensities below this threshold value are fully 
%transparent. 


%This example sets the volume threshold as less than 1 so that some brain 
%pixels remain visible, to give context to the spatial location of the 
%tumor inside the brain.
figure(2)
h1 = labelvolshow(groundTruthLabels{volId},vol3d);
h1.LabelVisibility(1) = 0;
h1.VolumeThreshold = 0.68;

%for same volume, displays predicted labels
figure(3)
h2 = labelvolshow(predictedLabels{volId},vol3d);
h2.LabelVisibility(1) = 0;
h2.VolumeThreshold = 0.68;


%QUANTIFYING SEGMENTATION ACCURACY
   
    %measure segmentation accuracy using dice function
    %computes Dice coefficient
    %code below calculates averages dice score of background across set of test volumes
diceResult = zeros(length(voldsTest.Files),2);

for j = 1:length(vol)
    diceResult(j,:) = dice(groundTruthLabels{j},predictedLabels{j});
end

%code below calculates average dice score across the set of 5 test volumes
meanDiceBackground = mean(diceResult(:,1));
disp(['Average Dice score of background across ',num2str(j), ...
    ' test volumes = ',num2str(meanDiceBackground)])
%code below calculates average dice score of tumor across 5 test volumes
meanDiceTumor = mean(diceResult(:,2));
disp(['Average Dice score of tumor across ',num2str(j), ...
    ' test volumes = ',num2str(meanDiceTumor)])

%boxplot function allows for visualization of statistics about dice scores
%across test volumes
%to create a boxplot set createBoxplot to true
createBoxplot = true;
if createBoxplot
    figure (4)
    boxplot(diceResult)
    title('Test Set Dice Accuracy')
    xticklabels(classNames)
    ylabel('Dice Coefficient')
end


imageDir = fullfile('/', 'rsrch1', 'ip', 'ip-comp_rsch_lab', 'mccollum');
% if ~exist(imageDir, 'dir')
%     mkdir(imageDir, 'dir')
% end

sourceDataLoc = [imageDir filesep 'Task01_BrainTumour'];
preprocessDataLoc = fullfile('/', 'rsrch1', 'ip', 'ip-comp_rsch_lab', 'mccollum','preprocessedDataset');
preprocessBraTSdataset(preprocessDataLoc,sourceDataLoc);


volReader = @(x) matRead(x);
volLoc = fullfile(preprocessDataLoc,'imagesTr');
volds = imageDatastore(volLoc, ...
    'FileExtensions','.mat','ReadFcn',volReader);

labelReader = @(x) matRead(x);
lblLoc = fullfile(preprocessDataLoc,'labelsTr');
classNames = ["background","tumor"];
pixelLabelID = [0 1];
pxds = pixelLabelDatastore(lblLoc,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',labelReader);

volume = preview(volds);
label = preview(pxds);
figure(1)
h = labelvolshow(label,volume(:,:,:,1));
h.LabelVisibility(1) = 0;

patchSize = [64 64 64];
patchPerImage = 16;
miniBatchSize = 8;
patchds = randomPatchExtractionDatastore(volds,pxds,patchSize, ...
    'PatchesPerImage',patchPerImage);
patchds.MiniBatchSize = miniBatchSize;

dsTrain = transform(patchds,@augment3dPatch);

volLocVal = fullfile(preprocessDataLoc,'imagesVal');
voldsVal = imageDatastore(volLocVal, ...
    'FileExtensions','.mat','ReadFcn',volReader);

lblLocVal = fullfile(preprocessDataLoc,'labelsVal');
pxdsVal = pixelLabelDatastore(lblLocVal,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',labelReader);

dsVal = randomPatchExtractionDatastore(voldsVal,pxdsVal,patchSize, ...
    'PatchesPerImage',patchPerImage);
dsVal.MiniBatchSize = miniBatchSize;
 
inputSize = [64 64 64 4];
inputLayer = image3dInputLayer(inputSize,'Normalization','none','Name','input');

numFiltersEncoder = [
    32 64; 
    64 128; 
    128 256];

layers = [inputLayer];
for module = 1:3
    modtag = num2str(module);
    encoderModule = [
        convolution3dLayer(3,numFiltersEncoder(module,1), ...
            'Padding','same','WeightsInitializer','narrow-normal', ...
            'Name',['en',modtag,'_conv1']);
        batchNormalizationLayer('Name',['en',modtag,'_bn']);
        reluLayer('Name',['en',modtag,'_relu1']);
        convolution3dLayer(3,numFiltersEncoder(module,2), ...
            'Padding','same','WeightsInitializer','narrow-normal', ...
            'Name',['en',modtag,'_conv2']);
        reluLayer('Name',['en',modtag,'_relu2']);
        maxPooling3dLayer(2,'Stride',2,'Padding','same', ...
            'Name',['en',modtag,'_maxpool']);
    ];
    
    layers = [layers; encoderModule];
end

numFiltersDecoder = [
    256 512; 
    256 256; 
    128 128; 
    64 64];

decoderModule4 = [
    convolution3dLayer(3,numFiltersDecoder(1,1), ...
        'Padding','same','WeightsInitializer','narrow-normal', ...
        'Name','de4_conv1');
    reluLayer('Name','de4_relu1');
    convolution3dLayer(3,numFiltersDecoder(1,2), ...
        'Padding','same','WeightsInitializer','narrow-normal', ...
        'Name','de4_conv2');
    reluLayer('Name','de4_relu2');
    transposedConv3dLayer(2,numFiltersDecoder(1,2),'Stride',2, ...
        'Name','de4_transconv');
];

decoderModule3 = [
    convolution3dLayer(3,numFiltersDecoder(2,1), ...
        'Padding','same','WeightsInitializer','narrow-normal', ....
        'Name','de3_conv1');       
    reluLayer('Name','de3_relu1');
    convolution3dLayer(3,numFiltersDecoder(2,2), ...
        'Padding','same','WeightsInitializer','narrow-normal', ...
        'Name','de3_conv2'); 
    reluLayer('Name','de3_relu2');
    transposedConv3dLayer(2,numFiltersDecoder(2,2),'Stride',2, ...
        'Name','de3_transconv');
];

decoderModule2 = [
    convolution3dLayer(3,numFiltersDecoder(3,1), ...
        'Padding','same','WeightsInitializer','narrow-normal', ...
        'Name','de2_conv1');
    reluLayer('Name','de2_relu1');
    convolution3dLayer(3,numFiltersDecoder(3,2), ...
        'Padding','same','WeightsInitializer','narrow-normal', ...
        'Name','de2_conv2');
    reluLayer('Name','de2_relu2');
    transposedConv3dLayer(2,numFiltersDecoder(3,2),'Stride',2, ...
        'Name','de2_transconv');
];


numLabels = 2;
decoderModuleFinal = [
    convolution3dLayer(3,numFiltersDecoder(4,1), ...
        'Padding','same','WeightsInitializer','narrow-normal', ...
        'Name','de1_conv1');
    reluLayer('Name','de1_relu1');
    convolution3dLayer(3,numFiltersDecoder(4,2), ...
        'Padding','same','WeightsInitializer','narrow-normal', ...
        'Name','de1_conv2');
    reluLayer('Name','de1_relu2');
    convolution3dLayer(1,numLabels,'Name','convLast');
    softmaxLayer('Name','softmax');
    dicePixelClassification3dLayer('output');
];

layers = [layers; decoderModule4];
lgraph = layerGraph(layers);
lgraph = addLayers(lgraph,decoderModule3);
lgraph = addLayers(lgraph,decoderModule2);
lgraph = addLayers(lgraph,decoderModuleFinal);

concat1 = concatenationLayer(4,2,'Name','concat1');
lgraph = addLayers(lgraph,concat1);
lgraph = connectLayers(lgraph,'en1_relu2','concat1/in1');
lgraph = connectLayers(lgraph,'de2_transconv','concat1/in2');
lgraph = connectLayers(lgraph,'concat1/out','de1_conv1');

concat2 = concatenationLayer(4,2,'Name','concat2');
lgraph = addLayers(lgraph,concat2);
lgraph = connectLayers(lgraph,'en2_relu2','concat2/in1');
lgraph = connectLayers(lgraph,'de3_transconv','concat2/in2');
lgraph = connectLayers(lgraph,'concat2/out','de2_conv1');

concat3 = concatenationLayer(4,2,'Name','concat3');
lgraph = addLayers(lgraph,concat3);
lgraph = connectLayers(lgraph,'en3_relu2','concat3/in1');
lgraph = connectLayers(lgraph,'de4_transconv','concat3/in2');
lgraph = connectLayers(lgraph,'concat3/out','de3_conv1');

lgraph = createUnet3d(inputSize);


analyzeNetwork(lgraph)



options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'InitialLearnRate',5e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',5, ...
    'LearnRateDropFactor',0.95, ...
    'ValidationData',dsVal, ...
    'ValidationFrequency',400, ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'MiniBatchSize',miniBatchSize);

% 
% %DOWNLOAD PRETRAINED NETWORK AND SAMPLE TEST SET
% % trained3DUnet_url = 'https://www.mathworks.com/supportfiles/vision/data/brainTumor3DUNet.mat';
% % sampleData_url = 'https://www.mathworks.com/supportfiles/vision/data/sampleBraTSTestSet.tar.gz';
% % 
% % imageDir = fullfile(tempdir,'BraTS');
% % if ~exist(imageDir,'dir')
% %     mkdir(imageDir);
% % end
% % downloadTrained3DUnetSampleData(trained3DUnet_url,sampleData_url,imageDir);

% % TRAIN THE NETWORK
% % if using pretrained network, keep doTraining = false
% doTraining = true;
% if doTraining
%     modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');
%     [net,info] = trainNetwork(dsTrain,lgraph,options);
%     save(['trained3DUNet-' modelDateTime '-Epoch-' num2str(maxEpochs) '.mat'],'net');
% else
%     load(fullfile(imageDir,'trained3DUNet','brainTumor3DUNet.mat'));
% end
% 
% %PERFORMING SEGMENTATION OF DATA
%     %if usefulltest set is false, code only processes 5 test volume
%     %if true, uses all 55 test images
% useFullTestSet = false;
% if useFullTestSet
%     volLocTest = fullfile(preprocessDataLoc,'imagesTest');
%     lblLocTest = fullfile(preprocessDataLoc,'labelsTest');
% else
%     volLocTest = fullfile(imageDir,'sampleBraTSTestSet','imagesTest');
%     lblLocTest = fullfile(imageDir,'sampleBraTSTestSet','labelsTest');
%     classNames = ["background","tumor"];
%     pixelLabelID = [0 1];
% end
% 
%     %Crop the central portion of the images and labels to size 128-by-128-by-128 voxels by using the helper function centerCropMatReader.
%     %This function is attached to the example as a supporting file. 
%     %The voldsTest variable stores the ground truth test images. 
%     %The pxdsTest variable stores the ground truth labels.
% windowSize = [128 128 128];
% volReader = @(x) centerCropMatReader(x,windowSize);
% labelReader = @(x) centerCropMatReader(x,windowSize);
% voldsTest = imageDatastore(volLocTest, ...
%     'FileExtensions','.mat','ReadFcn',volReader);
% pxdsTest = pixelLabelDatastore(lblLocTest,classNames,pixelLabelID, ...
%     'FileExtensions','.mat','ReadFcn',labelReader);
% 
% 
% %For each test image, add the ground truth image volumes and labels to cell arrays. 
% %Use the trained network with the semanticseg function to predict the labels for each test volume.
% 
% %After performing the segmentation, postprocess the predicted labels by labeling nonbrain voxels as 1, corresponding to the background. 
% %Use the test images to determine which voxels do not belong to the brain. 
% %You can also clean up the predicted labels by removing islands and filling holes using the medfilt3 function.
% %medfilt3 does not support categorical data, so cast the pixel label IDs to uint8 before the calculation. 
% %Then, cast the filtered labels back to the categorical data type, specifying the original pixel label IDs and class names.
% % % 
% % if exist( 'predictedLabels') ==1  
%     
% %file exists in workspace 
% 
% else
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
% 
% end
% 
% % 
% %  predictedLabels
% 
% 
% 
% %COMPARING GROUND TRUTH AGAINST NETWORK PREDICTION
%     %select one of the test images to evaluate the accuracy segmentation
%     %extract 4d data and store this 3d volume in vol3d variable
% volId = 2;
% vol3d = vol{volId}(:,:,:,1);
% 
% 
% %displays in montage ground truth and predicted labels along depth
% %direction
% zID = size(vol3d,3)/2;
% zSliceGT = labeloverlay(vol3d(:,:,zID),groundTruthLabels{volId}(:,:,zID));
% zSlicePred = labeloverlay(vol3d(:,:,zID),predictedLabels{volId}(:,:,zID));
% 
% figure(1)
% title('Labeled Ground Truth (Left) vs. Network Prediction (Right)')
% montage({zSliceGT;zSlicePred},'Size',[1 2],'BorderSize',5)
% 
% 
% 
% %display ground truth labeled volume using labelvol function
% 
% %third line makes background transparent
% 
% %to make some brain voxels transparent, specify volume threshold as a
% %number in the range of [0,1]
% 
% %All normalized volume intensities below this threshold value are fully 
% %transparent. 
% 
% 
% %This example sets the volume threshold as less than 1 so that some brain 
% %pixels remain visible, to give context to the spatial location of the 
% %tumor inside the brain.
% figure(2)
% h1 = labelvolshow(groundTruthLabels{volId},vol3d);
% h1.LabelVisibility(1) = 0;
% h1.VolumeThreshold = 0.68;
% 
% %for same volume, displays predicted labels
% figure(3)
% h2 = labelvolshow(predictedLabels{volId},vol3d);
% h2.LabelVisibility(1) = 0;
% h2.VolumeThreshold = 0.68;
% 
% 
% %QUANTIFYING SEGMENTATION ACCURACY
%    
%     %measure segmentation accuracy using dice function
%     %computes Dice coefficient
%     %code below calculates averages dice score of background across set of test volumes
% diceResult = zeros(length(voldsTest.Files),2);
% 
% for j = 1:length(vol)
%     diceResult(j,:) = dice(groundTruthLabels{j},predictedLabels{j});
% end
% 
% %code below calculates average dice score across the set of 5 test volumes
% meanDiceBackground = mean(diceResult(:,1));
% disp(['Average Dice score of background across ',num2str(j), ...
%     ' test volumes = ',num2str(meanDiceBackground)])
% %code below calculates average dice score of tumor across 5 test volumes
% meanDiceTumor = mean(diceResult(:,2));
% disp(['Average Dice score of tumor across ',num2str(j), ...
%     ' test volumes = ',num2str(meanDiceTumor)])
% 
% %boxplot function allows for visualization of statistics about dice scores
% %across test volumes
% %to create a boxplot set createBoxplot to true
% createBoxplot = true;
% if createBoxplot
%     figure (4)
%     boxplot(diceResult)
%     title('Test Set Dice Accuracy')
%     xticklabels(classNames)
%     ylabel('Dice Coefficient')
% end
% 
% 
% 
% 

%Download nifti file,convert into mat file for use in segmentation
% load_nii('input.nii')
% convertedmatlabfile = readnifti('inputnifti.nii'


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
if exist( 'predictedLabels') ==1  
    
%file exists in workspace 

else
    
id=1;
while hasdata(voldsTest)
    disp(['Processing test volume ' num2str(id)])
    
    groundTruthLabels{id} = read(pxdsTest);
    
    vol{id} = read(voldsTest);
    tempSeg = semanticseg(vol{id},net);

    % Get the non-brain region mask from the test image.
    volMask = vol{id}(:,:,:,1)==0;
    % Set the non-brain region of the predicted label as background.
    tempSeg(volMask) = classNames(1);
    % Perform median filtering on the predicted label.
    tempSeg = medfilt3(uint8(tempSeg)-1);
    % Cast the filtered label to categorial.
    tempSeg = categorical(tempSeg,pixelLabelID,classNames);
    predictedLabels{id} = tempSeg;
    id=id+1;
end

end

% 
%  predictedLabels



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

layername  = 'input'     
originalFeatures = activations(net,vol{volId},layername );
norm(originalFeatures(:)-vol{volId}(:))
orignii = make_nii(originalFeatures,[],[],[],'original');
save_nii(orignii,'original.nii' ) ;

% features after first convolution layer 1
layername = 'conv_Module1_Level1';
convFeatures = activations(net,vol{volId},layername );
convnii = make_nii(convFeatures ,[],[],[],'convolution');
save_nii(convnii,'convolution.nii' ) ;


% features after first batch normalization layer 2
layername = 'BN_Module1_Level1'
bnFeatures = activations(net,vol{volId},layername );
bn1nii = make_nii(bnFeatures, [],[],[], 'batchnorm');
save_nii(bn1nii, 'batchnorm1.nii')
expand_nii_scan('batchnorm1.nii')

%features after relu module 1 level 1 layer 3
layername = 'relu_Module1_Level1'
relulevel1Features = activations(net,vol{volId},layername );
relulev1nii = make_nii(relulevel1Features, [],[],[], 'relu');
save_nii(relulev1nii, 'relu_level1.nii')
expand_nii_scan('relu_level1.nii')

% features after level 2 convolution in module 1 layer 4
layername = 'conv_Module1_Level2'  
convlevel2Features = activations(net,vol{volId},layername );
conv2nii = make_nii(convlevel2Features, [],[],[], 'convolution');
save_nii(conv2nii, 'convolution2.nii')
expand_nii_scan('convolution2.nii')

% relu features layer 5 
layername = 'relu_Module1_Level2'  
relulevel2Features = activations(net,vol{volId},layername );
relulev2nii = make_nii(relulevel2Features, [],[],[], 'relu');
save_nii(relulev2nii, 'relu_level2.nii')
expand_nii_scan('relu_level2.nii')

% max pool layer 6
layername = 'maxpool_Module1'      
maxpoolFeatures = activations(net,vol{volId},layername );
maxpoolmod1 = make_nii(maxpoolFeatures, [],[],[], 'maxpool');
save_nii(maxpoolmod1, 'maxpoolmod1.nii')
expand_nii_scan('maxpoolmod1.nii')

% features after level 1 convolution in module 2 layer 7
layername = 'conv_Module2_Level1'  
convmod2levFeatures = activations(net,vol{volId},layername );
conv3nii = make_nii(convmod2levFeatures, [],[],[], 'convolution');
save_nii(conv3nii, 'convolution3.nii')
expand_nii_scan('convolution3.nii')

% features after first batch normalization layer 8
layername = 'BN_Module2_Level1'
bn2Features = activations(net,vol{volId},layername );
bn2nii = make_nii(bn2Features, [],[],[], 'batchnorm');
save_nii(bn2nii, 'batchnorm2.nii')
expand_nii_scan('batchnorm2.nii')

% relu features layer 9 
layername = 'relu_Module2_Level1'  
relulevel2_1Features = activations(net,vol{volId},layername );
relulev2_1nii = make_nii(relulevel2_1Features, [],[],[], 'relu');
save_nii(relulev2_1nii, 'relu_mod2lev1.nii')
expand_nii_scan('relu_mod2lev1.nii')

% features after level 1 convolution in module 2 layer 10
layername = 'conv_Module2_Level2'  
convmod2level2Features = activations(net,vol{volId},layername );
conv4nii = make_nii(convmod2level2Features, [],[],[], 'convolution');
save_nii(conv4nii, 'convolution4.nii')
expand_nii_scan('convolution4.nii')

%features at layer 11
layername = 'relu_Module2_Level2'  
relulevel2_2Features = activations(net,vol{volId},layername );
relulev2_2nii = make_nii(relulevel2_2Features, [],[],[], 'relu');
save_nii(relulev2_2nii, 'relu_mod2lev2.nii')
expand_nii_scan('relu_mod2lev2.nii')

% max pool layer 12
layername = 'maxpool_Module2'      
maxpool2Features = activations(net,vol{volId},layername );
maxpoolmod2 = make_nii(maxpool2Features, [],[],[], 'maxpool');
save_nii(maxpoolmod2, 'maxpoolmod2.nii')
expand_nii_scan('maxpoolmod2.nii')

% features after level 1 convolution in module 3 layer 13
layername = 'conv_Module3_Level1'  
convmod3level1Features = activations(net,vol{volId},layername );
conv5nii = make_nii(convmod3level1Features, [],[],[], 'convolution');
save_nii(conv5nii, 'convolution5.nii')
expand_nii_scan('convolution5.nii')

%features at layer 15
layername = 'relu_Module3_Level1'  
relulevel3_1Features = activations(net,vol{volId},layername );
relulev3_1nii = make_nii(relulevel3_1Features, [],[],[], 'relu');
save_nii(relulev3_1nii, 'relu_mod3lev1.nii')
expand_nii_scan('relu_mod3lev1.nii')

% features after level 1 convolution in module 3 layer 16
layername = 'conv_Module3_Level2'  
convmod3level2Features = activations(net,vol{volId},layername );
conv6nii = make_nii(convmod3level2Features, [],[],[], 'convolution');
save_nii(conv6nii, 'convolution6.nii')
expand_nii_scan('convolution6.nii')


%features at layer 17
layername = 'relu_Module3_Level2'  
relulevel3_2Features = activations(net,vol{volId},layername );
relulev3_2nii = make_nii(relulevel3_2Features, [],[],[], 'relu');
save_nii(relulev3_2nii, 'relu_mod3lev2.nii')
expand_nii_scan('relu_mod3lev2.nii')

% max pool layer 18
layername = 'maxpool_Module3'      
maxpool3Features = activations(net,vol{volId},layername );
maxpoolmod3 = make_nii(maxpool3Features, [],[],[], 'maxpool');
save_nii(maxpoolmod3, 'maxpoolmod3.nii')
expand_nii_scan('maxpoolmod3.nii')

%layer 19
layername = 'input'
inputlayerfeat = activations(net, vol{volId},layername);
inputlayer3d = make_nii(inputlayerfeat, [],[],[], 'input');
save_nii(inputlayer3d, 'inputlayer3d.nii')
%need expansion?

% features after level 1 convolution in module 4 level 1 layer 20
layername = 'conv_Module4_Level1'  
convmod4level1Features = activations(net,vol{volId},layername );
conv7nii = make_nii(convmod4level1Features, [],[],[], 'convolution');
save_nii(conv7nii, 'convolution7.nii')
expand_nii_scan('convolution7.nii')


%features at layer 21
layername = 'relu_Module4_Level1'  
relulevel4_1Features = activations(net,vol{volId},layername );
relulev4_1nii = make_nii(relulevel4_1Features, [],[],[], 'relu');
save_nii(relulev4_1nii, 'relu_mod4lev1.nii')
expand_nii_scan('relu_mod4lev1.nii')

% features after level 1 convolution in module 4 level 2 layer 22
layername = 'conv_Module4_Level2'  
convmod4level2Features = activations(net,vol{volId},layername );
conv8nii = make_nii(convmod4level2Features, [],[],[], 'convolution');
save_nii(conv8nii, 'convolution8.nii')
expand_nii_scan('convolution8.nii')

%features at layer 23
layername = 'relu_Module4_Level2'  
relulevel4_2Features = activations(net,vol{volId},layername );
relulev4_2nii = make_nii(relulevel4_2Features, [],[],[], 'relu');
save_nii(relulev4_2nii, 'relu_mod4lev2.nii')
expand_nii_scan('relu_mod4lev2.nii')

%layer 24
layername = 'transConv_Module4'
transFeatures = activations(net, vol{volId}, layername);
trans1nii = make_nii(transFeatures, [],[],[], 'transConv');
save_nii(trans1nii, 'trans1.nii')
%expand?

%features after level 1 in module 4 convolution layer 25
layername = 'conv_Module5_Level1'  
convlevel1mod5Features = activations(net,vol{volId},layername );
conv9nii = make_nii(convlevel1mod5Features, [],[],[], 'convolution');
save_nii(conv9nii, 'convolution9.nii')
expand_nii_scan('convolution9.nii')

%features at layer 26
layername = 'relu_Module5_Level1'  
relulevel5_1Features = activations(net,vol{volId},layername );
relulev5_1nii = make_nii(relulevel5_1Features, [],[],[], 'relu');
save_nii(relulev5_1nii, 'relu_mod5lev1.nii')
expand_nii_scan('relu_mod5lev1.nii')

%features after level 1 in module 4 convolution layer 27
layername = 'conv_Module5_Level2'  
convlevel2mod5Features = activations(net,vol{volId},layername );
conv10nii = make_nii(convlevel2mod5Features, [],[],[], 'convolution');
save_nii(conv10nii, 'convolution10.nii')
expand_nii_scan('convolution10.nii')

%features at layer 28
layername = 'relu_Module5_Level2'  
relulevel5_2Features = activations(net,vol{volId},layername );
relulev5_2nii = make_nii(relulevel5_2Features, [],[],[], 'relu');
save_nii(relulev5_2nii, 'relu_mod5lev2.nii')
expand_nii_scan('relu_mod5lev2.nii')

%layer 29
layername = 'transConv_Module5'
trans1Features = activations(net, vol{volId}, layername);
trans2nii = make_nii(trans1Features, [],[],[], 'transConv');
save_nii(trans2nii, 'trans2.nii')
%expand?


%features after level 1 in module 4 convolution layer 30
layername = 'conv_Module6_Level1'  
convlevel1mod6Features = activations(net,vol{volId},layername );
conv11nii = make_nii(convlevel1mod6Features, [],[],[], 'convolution');
save_nii(conv11nii, 'convolution11.nii')
expand_nii_scan('convolution11.nii')

%features at layer 31
layername = 'relu_Module6_Level1'  
relulevel6_1Features = activations(net,vol{volId},layername );
relulev6_1nii = make_nii(relulevel6_1Features, [],[],[], 'relu');
save_nii(relulev6_1nii, 'relu_mod6lev1.nii')
expand_nii_scan('relu_mod6lev1.nii')

%features after level 1 in module 4 convolution layer 32
layername = 'conv_Module6_Level2'  
convlevel2mod6Features = activations(net,vol{volId},layername );
conv12nii = make_nii(convlevel2mod6Features, [],[],[], 'convolution');
save_nii(conv12nii, 'convolution12.nii')
expand_nii_scan('convolution12.nii')

%features at layer 33
layername = 'relu_Module6_Level2'  
relulevel6_2Features = activations(net,vol{volId},layername );
relulev6_2nii = make_nii(relulevel6_2Features, [],[],[], 'relu');
save_nii(relulev6_2nii, 'relu_mod6lev2.nii')
expand_nii_scan('relu_mod6lev2.nii')


%layer 34
layername = 'transConv_Module6'
trans2Features = activations(net, vol{volId}, layername);
trans3nii = make_nii(trans2Features, [],[],[], 'transConv');
save_nii(trans3nii, 'trans3.nii')
%expand?

%features after level 1 in module 4 convolution layer 35
layername = 'conv_Module7_Level1'  
convlevel1mod7Features = activations(net,vol{volId},layername );
conv13nii = make_nii(convlevel1mod7Features, [],[],[], 'convolution');
save_nii(conv13nii, 'convolution13.nii')
expand_nii_scan('convolution13.nii')

%features at layer 36
layername = 'relu_Module7_Level1'  
relulevel7_1Features = activations(net,vol{volId},layername );
relulev7_1nii = make_nii(relulevel7_1Features, [],[],[], 'relu');
save_nii(relulev7_1nii, 'relu_mod7lev1.nii')
expand_nii_scan('relu_mod7lev1.nii')


%features after level 1 in module 4 convolution layer 37
layername = 'conv_Module7_Level2'  
convlevel2mod7Features = activations(net,vol{volId},layername );
conv14nii = make_nii(convlevel2mod7Features, [],[],[], 'convolution');
save_nii(conv14nii, 'convolution14.nii')
expand_nii_scan('convolution14.nii')



%features at layer 38
layername = 'relu_Module7_Level1'  
relulevel7_1Features = activations(net,vol{volId},layername );
relulev7_1nii = make_nii(relulevel7_1Features, [],[],[], 'relu');
save_nii(relulev7_1nii, 'relu_mod7lev1.nii')
expand_nii_scan('relu_mod7lev1.nii')

% last convolution features
size(net.Layers(39).Weights)
% NOTE - no neighborhood covolution. output is a linear combination of the input channels
% ans = 1     1     1    64     2
layername  = 'ConvLast_Module7'
lastconvFeatures = activations(net,vol{volId},layername );
lastconvnii = make_nii(lastconvFeatures,[],[],[],'lastconv');
save_nii(lastconvnii,'lastconv.nii' ) ;
expand_nii_scan('lastconv.nii')

% last relu
layername  = 'relu_Module7_Level2'  
lastReluFeatures = activations(net,vol{volId},layername);
lastrelunii = make_nii(lastReluFeatures ,[],[],[],'lastrelu');
save_nii(lastrelunii,'lastrelu.nii' ) ;
expand_nii_scan('lastrelu.nii')

% softmax features
layername  = 'softmax'              
softmaxFeatures = activations(net,vol{volId},layername );
softnii = make_nii(softmaxFeatures,[],[],[],'softmax');
save_nii(softnii,'softmax.nii' ) ;
expand_nii_scan('softmax.nii')

%%end of movie code


% features after first batch normalization layer 14
layername = 'BN_Module3_Level1'
bn3Features = activations(net,vol{volId},layername );
bn3nii = make_nii(bn3Features, [],[],[], 'batchnorm');
save_nii(bn3nii, 'batchnorm3.nii')



% input/output to transpose convolution layer
layername  = 'relu_Module6_Level2'
inputFeatures = activations(net,vol{volId},layername );
size(inputFeatures )

layername  = 'transConv_Module6'     
outputFeatures = activations(net,vol{volId},layername );
size(outputFeatures )

% concatenate features
layername  = 'concat1'              
concatFeatures = activations(net,vol{volId},layername );



% output predictions
segnii = make_nii(uint8(predictedLabels{volId}),[],[],[],'segmentation');
save_nii(segnii,'output.nii' ) ;

% view output
% vglrun itksnap -g original.nii -s output.nii -o convolution.nii lastrelu.nii lastconv.nii softmax.nii 

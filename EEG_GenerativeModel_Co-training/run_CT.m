clearvars;
%% preprocessing: read data and normalize
load EEG.mat;
EEG_corr = corr(EEG(:,1:14));
[N,m] = size(EEG); m = m-1; % m = number of raw features
% randIdx = randperm(N);
% load randIdx from EEG.mat to ensure the same datasets are being used

%% split data into labeled (L), unlabeled (U) and test (T) sections
L = 300; % starting number of labeled samples
T = 1000; % number of testing samples
U = N-L-T; % number of unlabeled samples

dataL = [EEG(randIdx(1:L),1:m) EEG(randIdx(1:L),m+1)]; % L contains both X and Y
dataU = EEG(randIdx(L+1:L+U),1:m); % for U, we do not have label Y
dataT = [EEG(randIdx(L+U+1:L+U+T),1:m) EEG(randIdx(L+U+1:L+U+T),m+1)]; % T contains both X and Y

% h1F = [3 5 14]; % raw features that h1 can access
% h2F = [2 7 12]; % raw features that h2 can access
h1F = [1:2:14];
h2F = [2:2:14];

%% Co-training
numIter = 300;
% numSamp = 3000;
numPick = 10;
Ac1_prog = zeros(numIter,1); % progressive accuracies of h1
Ac2_prog = zeros(numIter,1); % progressive accuracies of h2
for t = 1:numIter
    numSamp = size(dataU,1);
    [Ac1_prog(t),p1,k1,theta1] = NB_Gamma(dataL,dataT,h1F); % train h1 using L
    idxSamp1 = randperm(size(dataU,1));
    randsample_h1 = dataU(idxSamp1(1:numSamp),:);% randomly take numSamp samples from U
    Y_pseudo_h1 = NB_Gamma_detection(randsample_h1,h1F,p1,k1,theta1); % compute the pseudo labels
    [~,sortIdx1] = sort(Y_pseudo_h1); % sort the label values to find most confident samples
    selectIdx1 = sortIdx1([1:numPick/2, numSamp-numPick/2+1:numSamp]); % selectIdx includes numPick most confident samples labeled by h1
    randsample_h1 = randsample_h1(selectIdx1,:);
    Y_pseudo_h1 = Y_pseudo_h1(selectIdx1); 
    Y_pseudo_h1 = (sign(Y_pseudo_h1)+1)/2; % -1 -> 0 and 1 -> 1
    idxSamp1(selectIdx1) = [];
    dataU = dataU(idxSamp1,:); % remove those selected samples from dataU
    
    numSamp = size(dataU,1);
    [Ac2_prog(t),p2,k2,theta2] = NB_Gamma(dataL,dataT,h2F); % train h2 using L
    idxSamp2 = randperm(size(dataU,1));
    randsample_h2 = dataU(idxSamp2(1:numSamp),:);% randomly take numSamp samples from U
    Y_pseudo_h2 = NB_Gamma_detection(randsample_h2,h2F,p2,k2,theta2); % compute the pseudo labels
    [~,sortIdx2] = sort(Y_pseudo_h2); % sort the label values to find most confident samples
    selectIdx2 = sortIdx2([1:numPick/2, numSamp-numPick/2+1:numSamp]); % selectIdx includes numPick most confident samples labeled by h2
    randsample_h2 = randsample_h2(selectIdx2,:);
    Y_pseudo_h2 = Y_pseudo_h2(selectIdx2); 
    Y_pseudo_h2 = (sign(Y_pseudo_h2)+1)/2; % -1 -> 0 and 1 -> 1
    idxSamp2(selectIdx2) = [];
    dataU = dataU(idxSamp2,:); % remove those selected samples from dataU
    
    dataL = [dataL; randsample_h1 Y_pseudo_h1; randsample_h2 Y_pseudo_h2];
end
fig = figure;
plot(1:numIter,Ac1_prog*100,'-*',1:numIter,Ac2_prog*100,'-^');
grid on; xlabel('Number of iterations'); ylabel('Accuracy (%)');
legend('NB Gamma - h1','NB Gamma - h2','Location','Southeast');
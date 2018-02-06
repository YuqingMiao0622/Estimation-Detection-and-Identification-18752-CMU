
clearvars;
%% read from data file
load('EEG.mat');
N = size(EEG,1); % N ~= 15k
m = size(EEG,2)-1; % m = 14
X = EEG(:,1:m); % features X
Y = EEG(:,m+1); % last column, which contains label Y
for j = 1:m
    X(:,j) = (X(:,j)-mean(X(:,j)))/std(X(:,j));
end
min_X = repmat(min(X),N,1); % = min of each column
X = X - min_X + 1e-6; % = X minus min value of each column
data = [X Y]; % in Data, X has been normalized and min_X is deducted

%% sample a random train/test set
minL = 10; % minimum number of training samples to start with
stepL = 10; % step size for L
maxL = 5000; % maximum number of training samples
numT = 1000; % number of testing samples

orgF = 1:m; % which feature to use

randIdx = randperm(N); % random index of N 
dataTest = data(randIdx(maxL+1:maxL+numT),:); % test set has numT samples

Gamma_Ac_prog = zeros((maxL-minL)/stepL+1,1); % progressive accuracy of NB_Gamma
Normal_Ac_prog = zeros((maxL-minL)/stepL+1,1); % progressive accuracy of NB_Normal
for L = minL:stepL:maxL
    dataTrain = data(randIdx(1:L),:); % at each value of L, make a new training set
    
    [Gamma_Ac,Gamma_p,Gamma_k,Gamma_theta] = NB_Gamma(dataTrain,dataTest,orgF); % train NB_Gamma and test it against the test set
    Gamma_Ac_prog((L-minL)/stepL+1) = Gamma_Ac; % report the accuracy
    
    [Normal_Ac,Normal_p,Normal_mu,Normal_sigma] = NB_Normal(dataTrain,dataTest); % train NB_Normal and test it against the test set
    Normal_Ac_prog((L-minL)/stepL+1) = Normal_Ac; % report the accuracy
end
% save('EEG.mat','randIdx','Gamma_*','Normal_*','-append');
fig = figure;
plot(minL:stepL:maxL,Gamma_Ac_prog*100,'-*',minL:stepL:maxL,Normal_Ac_prog*100,'-^'); grid on;
xlabel('Number of training samples'); ylabel('Accuracy (%)');
legend('NB Gamma','NB Normal','Location','Southeast');
% print(fig,'Pic/varyL_NBGamma_NBNormal.jpg','-djpeg','-r150');
% savefig(fig,'Fig/varyL_NBGamma_NBNormal.fig');
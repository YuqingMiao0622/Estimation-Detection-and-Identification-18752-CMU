%% preprocessing: read data
load EEG.mat;
[N,~] = size(EEG); % N = total number of samples ~ 15k
minL = 10; % minimum number of training samples to start with
stepL = 10; % step size for L
maxL = 2000; % maximum number of training samples
numT = 1000; % number of testing samples
% randIdx = randperm(N); % random index of N


%% test SVM, load randIdx from EEG.mat to ensure that the same datasets are used
dataTest = EEG(randIdx(maxL+1:maxL+numT),:); % test set has numT samples

SVM_Ac_prog = zeros((maxL-minL)/stepL+1,1); % progressive accuracy of SVM
for L = minL:stepL:maxL
    dataTrain = EEG(randIdx(1:L),:); % at each value of L,
    [SVM_Ac,SVM_W,SVM_C] = SVM(dataTrain,dataTest,1:14); % train SVM and test it against the test set
    SVM_Ac_prog((L-minL)/stepL+1) = SVM_Ac; % report the accuracy
end
save('EEG.mat','SVM_*','-append');
fig = figure;
plot(minL:stepL:maxL,Gamma_Ac_prog*100,'-*',...
    minL:stepL:maxL,Normal_Ac_prog*100,'-^',...
    minL:stepL:maxL,SVM_Ac_prog*100,'-o'); 
xlabel('Number of training samples'); ylabel('Accuracy (%)');
grid on; legend('NB Gamma','NB Normal','SVM');
print(fig,'Pic/varyL_NBGamma_NBNormal_SVM.jpg','-djpeg','-r150');
savefig(fig,'Fig/varyL_NBGamma_NBNormal_SVM.fig');

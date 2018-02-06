clearvars;
close all;

%% read raw data
EEG_raw = load('EEG.txt'); % read from text file into a matrix

%% removing outliers
EEG = EEG_raw; % create a copy
m = size(EEG_raw,2)-1; % number of features in X;
n = size(EEG,1); % number of raw samples
i = 2;
while i <= n
    X_prev = EEG(i-1,1:m);
    X_current = EEG(i,1:m);
    if any(X_current-X_prev > 1e3) % if extreme difference is spotted, remove this current sample
        EEG(i,:) = []; % remove X_current
        n = n-1; % decrease n by 1 because X_current has been removed
        i = i-1; % decrease i by 1 because the next sample to examine is now at the old i
    end
    i = i+1; % examine the next sample
end

%% plot histograms
for j = 1:m % for each feature in X
    for y = 0:1 % for each value of Y
        fig = figure; % create a new figure
        histogram(EEG(find(EEG(:,15)==y),j)); % plot the histogram of X(:,j)|Y=y
        title (sprintf('Histogram of raw feature X%d given Y=%d.jpg',j,y));
        xlabel('Feature value'); ylabel('Count'); grid on;
        print(fig,'-djpeg',sprintf('Pic/hist_X%d_Y%d.jpg',j,y),'-r150'); % print to .jpg
        saveas(fig,sprintf('Fig/hist_X%d_Y%d.fig',j,y)); % print to .fig
        close(fig); % close the figure
    end
end

%% normalization
% for j = 1:m
%     EEG(:,j) = (EEG(:,j)-mean(EEG(:,j)))/std(EEG(:,j));
% end
save EEG.mat EEG_raw EEG

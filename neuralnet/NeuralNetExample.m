% Example wrapper on ao_nn for train-and-test of the ff neural network
%
%
% AS2020

id   = id;   % vector describing group (0,1), size(N,1)
data = data; % matrix of size(N,M) containing M predictors for each data

numneuron = 4;  % number of neurons in hidden layer
numiter   = 18; % number of iterations for the optimisation

% shuffle the data and ids
I = shuffle(1:length(id));

data = data(I,:);
id   = id(I,:);

% separate train and test:
train_index = 1:20;
test_index  = 21:32;

% train the network (compute weights and activations)
[M] = ao_nn(id(train_index),data(train_index,:),numneuron,numiter);

% compare real with training prediction - predictive powers
group_prediction = imax(M.pred_raw')-1;
T = predictive([id(train_index) group_prediction])

% test the trained network on new data points
pred = M.fun(spm_unvec(M.weightvec,M.modelspace),data(test_index,:));

% see how it did on new data:
test_prediction = [id(test_index) pred]
test = predictive(test_prediction)


% to train the same network further:
%[M] = ao_nn(id(train_index),data(train_index,:),numneuron,numiter,M.weightvec);

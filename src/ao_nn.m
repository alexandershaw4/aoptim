function [M] = ao_nn(y,x,nh,niter,nc)
% Super simple FF NN for classification, optimised by minimising free energy
%
% [X,F,CP,f] = ao_nn(y,x,nh,niter)
%
% y  = group identifier (0,1), size(n,1)
% x  = predictor variances, size(n,m)
% nh = num neurons in hidden layer
% niter = number of iterations for optimisation
%
% AS

rng default;

% sort class order, just for visualisation consistency
[~,ii] = sort(y,'ascend');
y = y(ii);
x = x(ii,:);

% reduce x using an svd pca (ish)
if nargin == 5
    [u,s,v] = spm_svd(x);
    x = x*v(:,1:nc);
end

x  = full(real(x));
[nob,np] = size(x); % num obs and vars


ny = length(unique(y));
yy = zeros(length(y),ny);
for i = 1:ny
    yy(find(y==i-1),i)=1;
end

% weights and activations
HL = zeros(nh,1);
W1 = ones(np,nh)/np*nh;
W2 = ones(nh,ny)/nh.^2;

% for testing purposes: force group difference in params:
%x(find(yy(:,2)),:)= x(find(yy(:,2)),:)+4;

% accessible functions to gen
m  = {W1 HL W2};
p  = real(spm_vec(m)) ;
c  = (~~p)/32;
g  = @(p) gen(p,m,x);

% optimisation settings
op = AO('options');
op.fun = g;
op.x0 = p(:);
op.V = c(:);
op.y = {yy};
op.maxit = niter;
op.criterion = -500;
op.step_method=1;
op.BTLineSearch=0;
op.hyperparams=1;
op.inner_loop = 8;
[X,F,CP,Pp] = AO(op);

% for testing the trained machine
f = @(m,x) x*m{1}*diag(1./(1 + exp(-m{2})))*m{3};

prediction = f(spm_unvec(X,m),x);


% outputs
M.weightvec  = X(:);
M.modelspace = m;
M.F          = F;
M.covariance = CP;
M.fun        = f;
M.prediction = round(prediction);
M.pred_raw   = prediction;


end

function pred = gen(p,m,x)

p = p(:);
m = spm_unvec(p,m);

W1 = m{1};
HL = m{2};
W2 = m{3};

pred = x*W1*diag(1./(1 + exp(-HL)))*W2;

end


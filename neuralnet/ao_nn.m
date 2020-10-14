function [M] = ao_nn(y,x,nh,niter,nc,W)
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

% reduce x using an svd pca (ish)
if nargin == 5 && ~isempty(nc)
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

% accessible parameters to f
m  = {W1 HL W2};

if nargin == 6 && ~isempty(W)
    m = spm_unvec(W,m);
    fprintf('Using users initial weights!\n');
end

p  = real(spm_vec(m)) ;
c  = (~~p)/32;
c  = [spm_vec(ones(size(m{1})))/32;
      spm_vec(ones(size(m{2})))/32;
      spm_vec(ones(size(m{3})))/32 ];
  
%g  = @(p) gen(p,m,x);

% the neural network machine - no loops! just a matrix operation
f    = @(m,x) imax( (x*m{1}*diag(1./(1 + exp(-m{2})))*m{3})' )-1;
f_nr = @(m,x)       (x*m{1}*diag(1./(1 + exp(-m{2})))*m{3});
g    = @(p) f_nr(spm_unvec(p,m),x);

% note I'm optimisming using f_nr - i.e. on a continuous, scalar
% prediction landscape rather than binary (f)

% free energy optimisation settings (required AO.m)
op = AO('options');
op.fun       = g;
op.x0        = p(:);
op.V         = c(:);
op.y         = {yy};
op.maxit     = niter;
op.criterion = -500;
op.step_method  = 1;
op.BTLineSearch = 0;
op.hyperparams  = 1;
op.inner_loop   = 8;

[X,F,CP,Pp]  = AO(op);
prediction   = f(spm_unvec(X,m),x);

% outputs
M.weightvec  = X(:);
M.modelspace = spm_unvec(X(:),m);
M.F          = F;
M.covariance = CP;
M.fun        = f;
M.fun_nr     = f_nr;
M.prediction = prediction;
M.pred_raw   = f_nr(spm_unvec(X,m),x);
M.truth      = yy;


end

% function pred = gen(p,m,x)
% 
% p = p(:);
% m = spm_unvec(p,m);
% 
% W1 = m{1};
% HL = m{2};
% W2 = m{3};
% 
% pred = x*W1*diag(1./(1 + exp(-HL)))*W2;
% 
% end


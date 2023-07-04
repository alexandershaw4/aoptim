
% make a series of sines
dt  = 1/600;
t   = 0:dt:(2-dt);
fun = @(f,a,th) a(:).*sin(2*pi*f(:)*t(:)' - th(:));

S = {[4 13 50 60] [5 10 40 40] [1 1 1 1]};
Y = fun(S{1},S{2},S{3});

% generate guess
x0 = randi(100,12,1);
V  = ones(12,1)/8;

g  = @(X) fun(X{1},X{2},X{3});
gd = @(X) g(spm_unvec(X,S)); 

% add fft on top
w   = 1:100;
PY  = atcm.fun.tfdecomp(sum(Y,1),dt,1:100,2);
fgd = @(X) atcm.fun.tfdecomp(sum(gd(X),1),dt,1:100,2);

% optimise
op = AO('options');

op.fun = fgd;          % function/model f(x0)
op.x0  = x0(:);      % start values: x0
op.y   = PY(:);       % data we're fitting (for computation of objective fun, e.g. e = Y - f(x)
op.V   = V(:);       % variance / step for each parameter, e.g. ones(length(x0),1)/8

op.hyperparams=0;
op.objective='gauss'; % select smooth Gaussian error function

% run it:
[X,F,CV,~,Hi] = AO(op); 


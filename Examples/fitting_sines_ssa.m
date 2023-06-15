
% make a series of sines
dt  = 1/600;
t   = 0:dt:(2-dt);
fun = @(f,a,th) a(:).*sin(2*pi*f(:)*t(:)' - th(:));

S = {[3 10] [5 20] [4 12]};
Y = fun(S{1},S{2},S{3});

% generate guess
x0 = randi(20,6,1);
V  = ones(6,1)/8;

g  = @(X) fun(X{1},X{2},X{3});
gd = @(X) g(spm_unvec(X,S)); 

% optimise
op = AO('options');

op.fun = gd;          % function/model f(x0)
op.x0  = x0(:);      % start values: x0
op.y   = Y(:);       % data we're fitting (for computation of objective fun, e.g. e = Y - f(x)
op.V   = V(:);       % variance / step for each parameter, e.g. ones(length(x0),1)/8

op.objective='gauss'; % select smooth Gaussian error function

% run it:
[X,F,CV,~,Hi] = AO(op); 


% change objective to 'gaussmap' for MAP estimation
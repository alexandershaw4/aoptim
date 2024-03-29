function [b,F,Cp,fit] = ao_glm(x,y)
% use AO curvature optimiser to fit a GLM...
%
% - fits the lm: X = c + Y(:,1)*�(1) + Y(:,n)*�(n)
% - performs similarly to glmfit
% - see 'example_aoglm.m' for usage and comparison
%
% AS

% orthogonalise predictors
%----------------------------------------------------
oy  = [y] * real(inv([y]' * [y])^(1/2));
id  = y'/oy';                       % i.e. oy*id ~ y

% model: x = c + y(1)*b(1) + y(2)*b(2) ...
%----------------------------------------------------
nb = size(y,2);
n  = size(y,1);
b  = ones(1,nb+1);
V  = [2 b(2:end)/64];

fun = @(b) (b(1)+b(2:end)*oy');

op = AO('options');
op.fun = fun;
op.x0  = b;
op.V   = V;
op.y   = x;
op.step_method = 9;
op.maxit = 1000;

%op.hypertune=1;
op.memory_optimise=1;
op.hyperparams=0;
op.rungekutta=8;
op.ismimo=1;

[b,F,Cp] = AO(op);

%[b,F,Cp] = AO(fun,b,V,x,inf);

% compute betas on the non-orthongal predictors
%----------------------------------------------------
m             = b(2:end)'*(y/id)';
b             = [b(1) m/y'];

% assess fit using adjusted r^2
%----------------------------------------------------
[fit.r,fit.p] = corr(x(:),m(:)); 
fit.r2        = (fit.r).^2;
fit.ar2       = 1 - (1 - fit.r2) * (n - 1)./(n - nb - 1);


% explicitly compute fitted variables to get correlations
%----------------------------------------------------
for i = 1:nb
    v0(:,i) = b(i+1)*y(:,i);
end

% fitted predictor correlations
%----------------------------------------------------
[fit.fitted_par_r,fit.fitted_par_p]=corr(x,v0);
fit.fitted_params = v0;

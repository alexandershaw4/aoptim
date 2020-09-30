function [b,F,Cp,fit] = ao_glm_no(x,y)
% use AO curvature optimiser to fit a GLM...
%
% - fits the lm: X = c + Y(:,1)*ß(1) + Y(:,n)*ß(n)
% - performs similarly to glmfit
% - see 'example_aoglm.m' for usage and comparison
% - no orthogonalisation version
%
% AS

% orthogonalise predictors
%----------------------------------------------------
%oy  = [y] * real(inv([y]' * [y])^(1/2));
%id  = y'/oy';                       % i.e. oy*id ~ y

% model: x = c + y(1)*b(1) + y(2)*b(2) ...
%----------------------------------------------------
nb = size(y,2);
n  = size(y,1);
b  = ones(1,nb+1);
V  = [2 b(2:end)/8];

fun = @(b) (b(1)+b(2:end)*y');
%[b,F,Cp] = AO(fun,b,V,x,inf);

opts     = AO('options');      
opts.fun = fun;
opts.x0  = b;
opts.V   = V;
opts.y   = x;
opts.maxit = inf;
opts.step_method = 3;
opts.hyperparams = 1;
[b,F,Cp] = AO(opts);


% compute betas on the non-orthongal predictors
%----------------------------------------------------
%m             = b(2:end)'*(y/id)';
%b             = [b(1) m/y'];
m = b(1) + b(2:end)'*(y);

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

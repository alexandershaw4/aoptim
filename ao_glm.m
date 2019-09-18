function [b,F,Cp,fit] = ao_glm(x,y)
% use AO curvature optimser to fit a GLM...
%
% fits: X = c + Y(:,1)*�(1) + Y(:,n)*�(n)
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
V  = [2 b(2:end)/8];

fun = @(b) (b(1)+b(2:end)*oy');
[b,F,Cp] = AO(fun,b,V,x,inf);

% compute betas on the non-orthongal predictors
%----------------------------------------------------
m             = b(2:end)'*(y/id)';
b             = [b(1) m/y'];

% assess fit using adjusted r^2
%----------------------------------------------------
[fit.r,fit.p] = corr(x(:),m(:)); 
fit.r2        = (fit.r).^2;
fit.ar2       = 1 - (1 - fit.r2) * (n - 1)./(n - nb - 1);

end

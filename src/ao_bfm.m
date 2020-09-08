function [b,F,Cp,fit] = ao_bfm(x,y)
% use AO curvature optimiser to fit a bilinear factor model
%
% essentially fits a 2-level lm to an eigendecomposition of the predictor
% matrix (y):
%
%   y    = u*s*v'
%   m{n} = u(n)*s(n,n)*v(n)'
%   
%   k * { c(1) + m(1){1}*b{1} + ... m(1){m}*b{m}
%       { ...
%       { c(n) + m(n){1}*b{1} + ... m(n){m}*b{m}
%   
%   betas = [k c b]
% AS

global gm

% % orthogonalise predictors
% %----------------------------------------------------
oy  = [y] * real(pinv([y]' * [y])^(1/2));
id  = y'/oy';                       % i.e. oy*id ~ y


% Use SPM to do the eigendecomposition [could use matlab svd]
%----------------------------------------------------
[u,s,v] = spm_svd(oy);

ne = length(s);                             % NUMBER EIG COMP
for i = 1:ne
    m{i} = full( u(:,i)*s(i,i)*v(:,i)' );
end
%                         oy == ( m{1} + m{2} ... + m{n} )


% construct extended model: x = c + y(1)*b(1) + y(2)*b(2) ...
%----------------------------------------------------
np = size(y,2);
nb = np*ne;
n  = size(y,1);
b  = ones(1,nb+ne+ne);
V  = [repmat(2,[1 ne]) b(ne+1:end)/8];

gm.ne = ne;
gm.np = np;
gm.y  = m;

% initial (non-iterative) fit
m0 = reshape(spm_vec(m),[n nb]);
m0 = [ones(n,1) m0];
bx = pinv(m0'*m0)*m0'*x;
b(end-nb+1:end)= bx(2:end);
b(ne+1:ne+ne)  = 1./bx(1)/3;
b(1:ne)        = 0.1;

op = AO('options');
op.fun = @fun;
op.x0 = b;
op.V = V;
op.y = x;
op.maxit=10;

[b,F,Cp] = AO(op);

%[b,F,Cp] = AO(@fun,b,V,x,inf,10);

% compute betas on the non-orthongal predictors
%----------------------------------------------------
% b = b';
% 
% k = b(1:gm.ne); b(1:gm.ne) = [];
% c = b(1:gm.ne); b(gm.ne+1:gm.ne+gm.ne) = [];
% w = reshape( b , [gm.ne, ( length(b)/ gm.ne )] );
% X = gm.y;
% 
% for i = 1:gm.ne
%     dy = c(i) + ( (w(i,:) * (X{i}/id)') );
%     nbeta(i,:) = dy/X{i}';
% end
% 
% %nbeta = sum(nbeta,1);
% for i = 1:gm.ne
%     p(i,:) = nbeta(i,:)*oy';
% end
% 
% YY = k*p;




% assess fit using adjusted r^2
%----------------------------------------------------
[fit.r,fit.p] = corr(x(:),m(:)); 
fit.r2        = (fit.r).^2;
fit.ar2       = 1 - (1 - fit.r2) * (n - 1)./(n - nb - 1);


% % explicitly compute fitted variables to get correlations
% %----------------------------------------------------
% for i = 1:nb
%     v0(:,i) = b(i+1)*y(:,i);
% end
% 
% % fitted predictor correlations
% %----------------------------------------------------
% [fit.fitted_par_r,fit.fitted_par_p]=corr(x,v0);
% fit.fitted_params = v0;

end

function y = fun(b)
global gm

% This function constructs the model and is called by the optimiser
%
%        x      k
%       / \     
%      x   x    c
%     / \ / \
%    x  x x  x  y
%  

k = b(1:gm.ne);
b(1:gm.ne) = [];

c = b(1:gm.ne);
b(gm.ne+1:gm.ne+gm.ne) = [];

w = reshape( b , [gm.ne, ( length(b)/ gm.ne )] );
X = gm.y;

for i = 1:gm.ne
    p(i,:) = c(i) + w(i,:)*X{i}' ;
end

y = k*p;

end



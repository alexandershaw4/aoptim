function [b,F,Cp,fit] = ao_glcnn(x,y)
% use AO curvature optimiser to invert a cnn...
%
% - fits the 'convolutional' model: 
%
%       X = ß{1} * ß{2} * ß{n} * y'
%
% where ß{1} = 1x2
%       ß{2} = 2x3
%       ß{n} = (n-1)xn
%
% AS

global aopt

% model: m{1}*m{2}...*m{n}
%----------------------------------------------------
nl = size(y,2); % num layers/tiers
n  = size(y,1); % num points

% Approx model topography:
% ---------------------------------
%
%          x      | m{1}
%         / \     |
%        x   x    | m{2}
%       / \ / \   |
%      x  x x  x  | m{n}
%      |  | |  |
% y = [0  1 2  3 ;
%      0  1 2  3 ;
%      0  1 2  3 ;
%      0  1 2  3 ];
%
% prediction y = m{1}*m{2}...*m{n}*y
%

% initial matrices
for i = 1:(nl-1)
    m{i} = ones(i,i+1) ;
end

nm = length(m);

aopt.m  = m;
aopt.yy = y;

b  = [1 *ones(length(m),1) ;    spm_vec(m)     ]; 
V  = [10*ones(length(m),1) ; (0*spm_vec(m))+1/8];

%[b,F,Cp] = AO(@fun,b(:),V,x,inf);

%obj = @(b) sum(spm_vec(x)-spm_vec(fun(b))).^2;


opts     = AO('options');      
opts.fun = @fun;
opts.x0  = b;
opts.V   = V;
opts.y   = x;
opts.maxit = inf;
opts.step_method = 1;
[b,F,Cp] = AO(opts);



%[b,F]=COA(obj,[b-1 b+1]',100000,10,10);
%Cp = [];

b = spm_unvec(b,m);

% assess fit using adjusted r^2
%----------------------------------------------------
m = fun(b);
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

end

function y = fun(b)
global aopt

m = aopt.m;
c = b(1:length(m));
b(1:length(m))=[];
b = spm_unvec(b,m);

for i = 1:(length(b)-1)
    
    dx = c(i) + ( b{i} * b{i+1} );
    b{i+1} = dx;
end

y = dx*aopt.yy';

end
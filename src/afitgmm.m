function [dy,pars] = afitgmm(y,start)
% fits a series of Gaussians to some data and returns the fit and its
% parameters. Uses a straightforward Gradient Descent on the SSE.
%
%
%
%

if nargin < 2 || isempty(start)
    x = {[10 30 50],[2 2 2],[4 4 4]};
else
    x = start;
end

n = length(y);
w = (1:n)';
f = @(a,f,wd) makef(w,a,f,wd);

fun  = @(dx) f(dx{:});
cfun = @(dx) fun(spm_unvec(dx,x));
g    = @(dx) spm_vec(sum( ((spm_vec(y)) - (spm_vec(cfun(dx))) ).^2 ));

lr = 1e-2; % learning rate
e  = g(x); % start position
n  = 0;

crit = 1e-3;

while e > crit
    
    n = n + 1;
        
    j = jaco(g,spm_vec(x),lr*~~real(spm_vec(x)),0,1);

    j(isinf(j))=0;
    j(isnan(j))=0;
    
    x  = spm_unvec( spm_vec(x) - (lr * spm_vec(j)), x);
    
    e = g(x);
    e
        
    if n == 4000
        break;
    end
end

dy   = f(x{:});
pars = x;
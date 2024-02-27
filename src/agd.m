function [X,F] = agd(f,x,n,a)
% super simple gradient descent of a multivariate function
% using Newton steps ('Newton's method')
%
% [X,F] = agd(fun,x0,Niterations,[initial_lr])
%
% AS

e = f(x);

if nargin < 4 || isempty(a);
    a = repmat(4,[length(x) 1]);
end
if length(a) == 1
    a = repmat(a,[length(x) 1]);
end

doplot = 1;
alle   = e;
aorig  = a;

fprintf('Initialise\n');
fprintf('It: %d | f = %d\n',0,e);

% iterate
for i = 1:n
    
    % gradients
    g = gradfind(f,x);
    g = g ./ norm(g);

    % check whether f'(x) == f(x)
    if norm(g) < 1e-6
        fprintf('finished\n');
        X = x; F = e;
        return;
    end

    % optimise learning rate (a) & Newtons method
    fa = @(a) f(x + (a).*-g(:));
    ga = gradfind(fa,a,2);
    ga = ga./norm(ga);
    a  = ((pinv(ga'*ga))*aorig);

    % step and evaluate
    x  = x + (a).*-g(:);
    e  = f(x);

    fprintf('It: %d | f = %d\n',i,e);
    alle = [alle e];

    % stop at maximum iterations
    if i == n
        X = x;
        F = e;
        fprintf('Did not finish\n');
    end

    if doplot
        plot(1:i+1,alle,'*',1:i+1,alle);drawnow;
    end

end


end

function g = gradfind(f,x,k)

g  = dfdx(f,x);
up = find(g==0);

if nargin < 3 || isempty(k)
    k = 1;
end

iterate = 1; n = 0;
while iterate
    n  = n + 1;
    gx = dfdx(f,x,abs(max(x))*n*k);
    ux = find(gx==0);

    if isempty(ux) || n == 16
        iterate = false;
    end
end

% reconstruct grad vector
g(up) = gx(up);


end

function g = dfdx(f,x,k)
% simple 1-step finite difference routine for compute partial gradients

e0 = f(x);

if nargin < 3 || isempty(k)
    k  = exp(-8);
end

for i  = 1:length(x)
    dx    = x;
    dx(i) = dx(i) + k;
    g(i)  = (f(dx) - e0) ./ k;
end

end
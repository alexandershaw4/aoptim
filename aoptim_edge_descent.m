function [X,F] = aoptim_edge_descent(fun,x0,V,y,maxit)
global aopt
% gradient descent based optimisation
%
% minimise a problem of the form:
%
% y = f(x)
% e = sum(Y0 - y).^2
%
% usage:
%   [X,F] = aoptim_edge(fun,x0,V,y,maxit,type)
%
% fun = functional handle / anonymous function
% x0  = starting points (vector input to fun)
% V   = variances for each element of x0
% y   = Y0, for computing the objective: e = sum(Y0 - y).^2
% maxit = number of iterations (def=128)
%
% 
% To fit problems of the form:
% 
% e = f(x)
%
% usage - set y=0:
%   [X,F] = aoptim_edge(fun,x0,V,0,maxit,type)
%
%
% AS2019

if nargin < 5 || isempty(maxit)
    maxit = 128;
end

% check functions
%--------------------------------------------------------------------------
aopt.fun  = fun;
aopt.y    = y(:);
x0        = full(x0(:));
V         = full(V(:));
[e0]      = obj(x0);

n         = 0;
iterate   = true;
criterion = 1e-2;
doplot    = 1;

V  = smooth(V);
Vb = V;

% initial point plot
if doplot
    makeplot(x0);
end

n_reject_consec = 0;
    
% initialise step size
red = 1;

% start loop
while iterate
    
    % counter
    n = n + 1;
   
    % construct an optimiser
    [de,dp,V] = pol_opt(x0,red);
    
    % ignore complex parameter values?
    dp = real(dp);
    
    % assess output
    if de < e0
        x0    = dp;
        e0    = de;
        red   = red / 2;
        
        pupdate(n,de,e0,'accept');
        if doplot; makeplot(x0); end
        n_reject_consec = 0;
    else
        pupdate(n,de,e0,'reject');
        
        % reset grid and variance
        red             = red * 2;
        n_reject_consec = n_reject_consec + 1;
            
    end
        
    % stop at max iterations
    if n == maxit
        X = x0;
        F = e0;
        return;
    end
    
    % check for convergence
    if e0 <= criterion
        fprintf('Convergence.\n');
        X = x0;
        F = e0;
        return;
    end
    
    % give up after 10 failed iterations
    if n_reject_consec == 10
        fprintf('Failed to converge... \nReturning best estimates.\n');
        X = x0;
        F = e0;
        return;
    end
    
end



end

function [e,x1,V] = pol_opt(x0,red)
global aopt

% points
x0 = full(x0(:));

% get gradient
[e0,J,er,Q] = obj(x0);

% compute search directions
df0 = J;
%red = 1;
s   = -df0; 
d0  = -s'*s;           % initial search direction (steepest) and slope
x3  = red/(1-d0);     % initial step is red/(|s|+1)

improve = true;
while improve
    % descend 
    dx       = (x0+x3.*s);
    [f3,df3] = obj(dx);
    
    if f3 < e0
        e0 = f3;
        x0 = dx;
        
        d3 = df3'*s; % new slope
        x3 = red/(1-d3); 
    else
        % return
        improve = false;
    end
end

e  = e0;
x1 = x0;

end

function pupdate(it,err,best,action)

fprintf('| Main It: %04i | Err: %04i | Best: %04i | %s |\n',it,err,best,action);

end

function makeplot(x)

% compute objective and get data and model preidction
[e,~,er,Q,Y,y] = obj(x);

if iscell(Y)
    plot(spm_cat(Y),':'); hold on;
    plot(spm_cat(y)    ); hold off;
    drawnow;
end


end

function [e,J,er,Q,Y,y] = obj(x0)
global aopt

IS = aopt.fun;
P  = x0(:);

y  = IS(P); 
Y  = aopt.y;
Q  = 1;

try;   e  = sum( Q*(spm_vec(Y) - spm_vec(y)).^2 );
catch; e  = sum(   (spm_vec(Y) - spm_vec(y)).^2 );
end

% error along output vector
er = spm_vec(y) - spm_vec(Y);

if nargout > 1
    % compute jacobi
    V = ones(size(x0));
    [J,ip] = jaco(@obj,x0,V);
end


end
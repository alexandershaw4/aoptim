function [X,F,Cp] = aoptim_edge_descent(fun,x0,V,y,maxit)
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
search          = 0;

% initialise step size
%red = V./max(V);
v     = V;
pC    = diag(V);
% variance (reduced space)
V     = spm_svd(pC);
pC    = V'*pC*V;
ipC   = inv(spm_cat(spm_diag({pC})));
red   = diag(pC);

% parameters (reduced space)
np    = size(V,2); 
p     = [V'*x0];
ip    = (1:np)';
Ep    = V*p(ip);

% now: x0 = V*p(ip)
%-------------------------------------------------------------
if obj( V*p(ip) ) ~= e0
    fprintf('Something went wrong during svd parameter reduction\n');
else
    % backup start points
    X0 = x0;
    % overwrite x0
    x0 = p;
end

% print start point
pupdate(n,0,e0,e0,'start:');

% start loop
while iterate
    
    % counter
    n = n + 1;
   
    % compute gradients & search directions
    %----------------------------------------------------------------------
    [e0,df0,er,Q] = obj( V*x0(ip) );
    
    s   = -df0';
    d0  = -s'*s;           % initial search direction (steepest) and slope
    x3  = V*red(ip)./(1-d0);     % initial step is red/(|s|+1)
    
    % make copies of error and param set
    x1  = x0;
    e1  = e0;

    % start counters
    improve = true;
    nfun    = 0;

    % iterative descent on this slope
    %----------------------------------------------------------------------
    while improve
        
        % descend while we can
        nfun = nfun + 1;
        %dx   = (V*x1(ip)+V*x3(ip).*s);
        dx = (V*x1(ip)+x3*s');
        %[de] = obj(dx);

        % assess each new parameter individually, then find the best mix
        for nip = 1:length(dx)
            XX       = V*x0;
            XX(nip)  = dx(nip);
            DFE(nip) = obj(XX);
        end
        
        % compute improver-parameters
        gp  = double(DFE < e0);
        gpi = find(gp);
        
        % only update select parameters
        ddx        = V*x0;
        ddx(gpi)   = dx(gpi);
        dx         = ddx;
        de         = obj(dx);
                
        if de < e1
            % update the error
            e1 = de;
            % update the (reduced) parameter set
            x1 = V'*dx;
        else
            % return
            improve = false;
        end
    end
    
    % ignore complex parameter values?
    x1 = real(x1);
    
    % evaluate - accept/reject - adjust variance
    %----------------------------------------------------------------------
    if e1 < e0
        x0 = x1;
        e0 = e1;
        pupdate(n,nfun,e1,e0,'accept');
        if doplot; makeplot(V*x0(ip)); end
        n_reject_consec = 0;
                
    else
        % if didn't improve: what to do?
        %----------------------------------------------------------
        pupdate(n,nfun,e1,e0,'reject');
        
        % update covariance
        %----------------------------------------------------------
        Pp    = real(df0*df0');
        Pp    = V'*Pp*V;            % compact covariance
        Cp    = spm_inv(Pp + ipC);
        ipC   = inv(Cp);
        
        % update 'variance' term from covariance
        %----------------------------------------------------------
        red = diag(Cp);
        
        % change step: distance between initial point and accepted updates
        % so far for each parameter
        %----------------------------------------------------------
        eu  = diag(cdist(X0,x0));
        red = (red./eu);
        
        % quick diagnostic on parameter contributions
        %----------------------------------------------------------
        pupdate(n,nfun,e1,e0,'study:');
        for nip = 1:length(dx)
            XX       = V*x0;
            XX(nip)  = dx(nip);
            dfe(nip) = obj(XX); % observe f(x,P[i])
            if dfe(nip) < e0
                px_check(nip) = 1;
            else
                px_check(nip) = 0;
            end
        end
        
        % check whether accepting all 'good' together is a good fit
        %----------------------------------------------------------
        XX              = V*x0;
        XX(px_check==1) = dx(px_check==1);
        DFE             = obj(XX);
        
        if DFE < e0
            % ok, good - accept
            %-----------------------------------
            e0 = DFE;    % new error
            x0 = V'*XX;  % compact parameter form
            
            % now adjust 'red' accounting for this new knowledge
            %-----------------------------------
            dred              = V*red;
            dred(px_check==0) = dred(px_check==0)*0.8;
            red               = V'*dred;
            
            pupdate(n,nfun,e0,e0,'accept');
            if doplot; makeplot(V*x0(ip)); end
        else
            % do this adjustment without updating model
            %-----------------------------------
            dred              = V*red;
            dred(px_check==0) = dred(px_check==0)*0.8;
            red               = V'*dred;
            
            pupdate(n,nfun,DFE,e0,'reject');
        end
        
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

function pupdate(it,nfun,err,best,action)

fprintf('| Main It: %04i | nf: %04i | Err: %04i | Best: %04i | %s |\n',it,nfun,err,best,action);

end

function makeplot(x)

% compute objective and get data and model preidction
%[e,~,er,Q,Y,y] = obj(x);
[Y,y] = GetStates(x);

if iscell(Y)
    plot(spm_cat(Y),':'); hold on;
    plot(spm_cat(y)    ); hold off;
    drawnow;
end


end

function [Y,y] = GetStates(x)
global aopt

IS = aopt.fun;
P  = x(:);

y  = IS(P); 
Y  = aopt.y;

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
    [J,ip] = jaco(@obj,x0,V,0);
end


end
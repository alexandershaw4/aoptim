function [X,F] = AO(fun,x0,V,y,maxit,inner_loop,Q)
% gradient descent based optimisation primarily for model fitting
%
% minimise a model fitting problem of the form:
%   y = f(x)
%   e = sum(Y0 - y).^2
%
% usage:
%   [X,F] = AO(fun,x0,V,y,maxit,type,Q)
%
% fun   = functional handle / anonymous function
% x0    = starting points (vector input to fun)
% V     = variances for each element of x0
% y     = Y0, for computing the objective: e = sum(Y0 - y).^2
% maxit = number of iterations (def=128) to restart descent
% inner_loop = num iters to continue on a specific descent
% Q     = optional precision matrix, e.g. e = sum( Q*(Y0-y) ).^2
%
%
% to minimise objective problems of the form:
% e = f(x)
%
% usage is set y=0:
%   [X,F] = AO(fun,x0,V,0,maxit,type)
%
%
% * note: - if failing to run properly, try transposing input vector x0
% *       - may need to remove the transpose at line 299: P = x0(:)';
%
% AS2019
% alexandershaw4@gmail.com
%
global aopt

if nargin < 7 || isempty(Q)
    Q = 1;
end
if nargin < 6 || isempty(inner_loop)
    inner_loop = 9999;
end
if nargin < 5 || isempty(maxit)
    maxit = 128;
end

% check functions
%--------------------------------------------------------------------------
aopt.fun  = fun;
aopt.y    = y(:);
aopt.Q    = Q;
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
%--------------------------------------------------------------------------
if doplot
    makeplot(x0);
end

% initialise counters
%--------------------------------------------------------------------------
n_reject_consec = 0;
search          = 0;

% initialise step size
%--------------------------------------------------------------------------
v     = V;
pC    = diag(V);

% variance (reduced space)
%--------------------------------------------------------------------------
V     = spm_svd(pC);
pC    = V'*pC*V;
ipC   = inv(spm_cat(spm_diag({pC})));
red   = diag(pC);

aopt.pC = V*red; % store for derivative function access

% parameters (reduced space)
%--------------------------------------------------------------------------
np    = size(V,2); 
p     = [V'*x0];
ip    = (1:np)';
Ep    = V*p(ip);

% print updates at n_print intervals along the inner loop
n_print    = 0;

% now: x0 = V*p(ip)
%--------------------------------------------------------------------------
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
%==========================================================================
while iterate
    
    % counter
    %----------------------------------------------------------------------
    n = n + 1;      
   
    % compute gradients & search directions
    %----------------------------------------------------------------------
    [e0,df0] = obj( V*x0(ip) );
    
    % initial search direction (steepest) and slope
    %----------------------------------------------------------------------
    s   = -df0';
    d0  = -s'*s;           
    x3  = V*red(ip)./(1-d0);                 % initial step is red/(|s|+1)
    
    % make copies of error and param set
    x1  = x0;
    e1  = e0;

    % start counters
    improve = true;
    nfun    = 0;

    % iterative descent on this slope
    %======================================================================
    while improve
        
        % descend while we can
        nfun = nfun + 1;
        
        % print updates and update plot intermittently
        if ismember(nfun,round(linspace( (inner_loop/n_print),inner_loop,n_print )) )
            pupdate(nfun,nfun,e1,e0,'contin');
            if doplot; makeplot(V*x1(ip)); end
        end
        
        %dx   = (V*x1(ip)+V*x3(ip).*s);
        %[de] = obj(dx);
        
        % continue the descent
        dx    = (V*x1(ip)+x3*s');
                
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
                
        if de  < e1
            
            % update the error & the (reduced) parameter set
            %--------------------------------------------------------------
            e1 = de;
            x1 = V'*dx;
        else
            
            % flag to stop this loop
            %--------------------------------------------------------------
            improve = false;
        end
        
        % upper limit on the length of this loop
        if nfun == inner_loop
            improve = false;
        end
        
    end  % end while improve...
      
    % ignore complex parameter values - for most functions, yes
    %----------------------------------------------------------------------
    x1 = real(x1);
    
    % evaluate - accept/reject - plot - adjust rate
    %======================================================================
    if e1 < e0
        
        % compute deltas & accept new parameters and error
        %------------------------------------------------------------------
        %e0 = e1;
        %x0 = x1;
        %eX =  e0;
        
        df =  e0 - e1;
        dp =  x0 - x1;
        x0 = -dp + x0;
        e0 =  e1;
        
        %m  = (eX-e0)./e0;
        
        %if the system is (locally?) linear, and we know what dp caused de
        %we can quickly exploit this to estimate the minimum on this descent
        %df./dp 
        %------------------------------------------------------------------
        exploit = true;
        nexpl   = 0;
        while exploit
            if obj(V*(x1-(dp./df))) < e1
               x1    = V*(x1-(dp./df));
                e1    = obj(x1);
                x1    = V'*x1;
                nexpl = nexpl + 1;
            else
                exploit = false;
                pupdate(n,nexpl,e1,e0,'extrap');
            end
        end
        e0 = e1;
        x0 = x1;
            
        
        % print & plots success
        %------------------------------------------------------------------
        pupdate(n,nfun,e1,e0,'accept');
        if doplot; makeplot(V*x0(ip)); end
        n_reject_consec = 0;
                
    else
        
        % if didn't improve: what to do?
        %------------------------------------------------------------------
        pupdate(n,nfun,e1,e0,'adjust');
                                
        % sample from improvers params in dx
        %------------------------------------------------------------------
        pupdate(n,nfun,e1,e0,'sample');
        thisgood = gp*0;
        if any(gp)
            
            % sort good params by improvement amount
            %--------------------------------------------------------------
            [~,PO] = sort(DFE(gpi),'ascend');
            dx0    = dx;
            
            % loop the good params in effect-size order
            % accept on the fly (additive effects)
            %--------------------------------------------------------------
            improve1 = 1;
            while improve1
                thisgood = gp*0;
                
                for i  = 1:length(gpi)
                    xnew             = real(V*x0);
                    xnew(gpi(PO(i))) = dx0(gpi(PO(i)));
                    xnew             = real(xnew);
                    enew             = obj(xnew);

                    if enew < e0
                        x0  = V'*xnew;
                        e0  = enew;
                        thisgood(gpi(PO(i))) = 1;
                    end
                end
                if any(thisgood)

                    % print & plot update
                    pupdate(n,nfun,e0,e0,'accept');
                    if doplot; makeplot(V*x0); end

                    % update step size for these params
                    red = red+V'*((V*red).*thisgood'*.2);
                else
                    % reduce step and go back to main loop
                    red = red*.8;
                    
                    % halt this while loop
                    improve1 = 0;
                    
                    % keep counting rejections
                    n_reject_consec = n_reject_consec + 1;
                end
                
                % update global store of V
                aopt.pC = V*red;
            end
        else
            
            pupdate(n,nfun,e0,e0,'nogood');
            
            % reduce step and go back to main loop
            red = red*.8;
            % update global store of V
            aopt.pC = V*red;
            % keep counting rejections
            n_reject_consec = n_reject_consec + 1;
        end
                            
        
    end
    
    % stopping criteria, rules etc.
    %======================================================================
    
    % if 3 fails, reset the reduction term (based on the specified variance)
    if n_reject_consec == 3
        pupdate(n,nfun,e1,e0,'resetV');
        %red = red ./ max(red(:));
        red = V*diag(pC);
        aopt.pC = red;
    end
    
    % stop at max iterations
    if n == maxit
        X = V*x0(ip);
        F = e0;
        return;
    end
    
    % check for convergence
    if e0 <= criterion
        fprintf('Convergence.\n');
        X = V*x0(ip);
        F = e0;
        return;
    end
    
    % give up after 10 failed iterations
    if n_reject_consec == 5
        fprintf('Failed to converge... \nReturning best estimates.\n');
        X = V*x0(ip);
        F = e0;
        return;
    end
    
end



end

function pupdate(it,nfun,err,best,action)

fprintf('| Main It: %04i | nf: %04i | Err: %04i | Best: %04i | %s |\n',it,nfun,err,best,action);

end

function makeplot(x)
% plot the function output (f(x)) on top of the thing we're ditting (Y)
%
%

[Y,y] = GetStates(x);

if iscell(Y)
    plot(spm_cat(Y),':'); hold on;
    plot(spm_cat(y)    ); hold off;
    drawnow;
end


end

function [Y,y] = GetStates(x)
% - evaluates the model and returns it along with the stored data Y
%

global aopt

IS = aopt.fun;
P  = x(:)';

y  = IS(P); 
Y  = aopt.y;

end

function [e,J,er,Q,Y,y] = obj(x0)
% - compute the objective function - i.e. the sqaured error to minimise
% - also returns the parameter Jacobian
%

global aopt

IS = aopt.fun;
P  = x0(:)';

y  = IS(P); 
Y  = aopt.y;
Q  = aopt.Q;

%e  = sum( (spm_vec(Y) - spm_vec(y)).^2 );

if all(size(Q)>1)
    Q = diag(Q);
    Q = (Q)./sum( ( Q(:) ));
    e = (spm_vec(Y) - spm_vec(y)).^2;
    e = e + (e.*Q);
    e = sum(e);
else
    e  = sum( (spm_vec(Y) - spm_vec(y)).^2 );
end

%try;   e  = sum( Q*(spm_vec(Y) - spm_vec(y)).^2 );
%catch; e  = sum(   (spm_vec(Y) - spm_vec(y)).^2 );
%end

% error along output vector
er = spm_vec(y) - spm_vec(Y);

if nargout > 1
    V  = aopt.pC;
    % compute jacobi
    %V = ones(size(x0));
    [J,ip] = jaco(@obj,x0,V,0);
end


end
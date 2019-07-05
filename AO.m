function [X,F] = AO(fun,x0,V,y,maxit,inner_loop,Q)
% gradient/curvature descent based optimisation, primarily for model fitting
% 
% Fit models of the form:
%   Y0 = f(x) + e
%
% Y0 = empirical data (to fit)
% f  = model (function)
% x  = model parameters/inputs to be optimised
%
%
% to minimise a model fitting problem of the form:
%   y = f(x)
%   e = sum(Y0 - y).^2
%
% the usage is:
%   [X,F] = AO(fun,x0,V,y,maxit,type,Q)
%
% fun        = function handle / anonymous function
% x0         = starting points (vector input to fun)
% V          = variances controlling each element of x0
% y          = Y0 / the data to fit, for computing the objective: e = sum(Y0 - y).^2
% maxit      = number of iterations (def=128) to restart descent
% inner_loop = num iters to continue on a specific descent (def=9999)
% Q          = optional precision matrix, e.g. e = sum( diag(Q).*(Y0-y) ).^2
%
%
% to minimise objective problems of the form:
%   e = f(x)
%
% the usage is: [note: set y=0 if f(x) returns the error/objective to be minimised]
%   [X,F] = AO(fun,x0,V,0,maxit,type) 
%
%
% * note: - if failing to run properly, try transposing input vector x0
% *       - may need to remove the transpose at line 406: P = x0(:)';
%
%
% Pseudo-code on how it works:
%
% - while improving:
%   * compute parameter gradients (1st order partial derivatives)
%   * derive descent path / initial step
%   - while improving on this path:
%     * descend a step
%     * evaluate effect of each parameters new value
%     * evaluate effect of accepting all params that decreased error
%     * if good, restart this loop (keep descending)
%     * if bad, stop this loop and:
%       * try exploiting best model achieved in prev loop using dp/df
%         (e.g. all good parameters taking the same steps on same paths again)
%       * if this works, keep doing it until it doesn't improve, then:
%         * of the good parameters, compute relative effects
%         * in effect-size order, individually test and accept each param
%         * when this stops improving, go back to the start, i.e. recompute
%         the gradients
%
% other notes:
% - an svd parameter reduction is applied
% - the step size is adjusted as the loops progress 
% - after 3 complete runs without any improvement, step size resets
% - after 5 complete runs without any improvement, gives up
% - if the variance in the improvement over a period is small, assumes
% stuck in a local minima and gives up
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

% check functions, inputs, options...
%--------------------------------------------------------------------------
aopt.order = 2;               % first or second order derivatives [use 2nd]
aopt.fun   = fun;
aopt.y     = y(:);
aopt.Q     = Q;
x0         = full(x0(:));
V          = full(V(:));
[e0]       = obj(x0);

n          = 0;
iterate    = true;
criterion  = 1e-2;
doplot     = 1;

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

dff          = [];
localminflag = 0;

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
    x3  = V*red(ip)./(1-d0);                 % initial step 
    
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
            DFE(nip) = obj(real(XX));
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
            df  = e1 - de;
            e1  = de;
            x1  = V'*dx;
            dff = [dff df];
        else
            
            % flag to stop this loop
            %--------------------------------------------------------------
            improve = false;            
        end
        
        % upper limit on the length of this loop
        if nfun >= inner_loop
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
        df =  e0 - e1;
        dp =  x0 - x1;
        x0 = -dp + x0;
        e0 =  e1;
                
        %if the system is (locally?) linear, and we know what dp caused de
        %we can quickly exploit this to estimate the minimum on this descent
        %df./dp 
        %------------------------------------------------------------------
        exploit = true;
        nexpl   = 0;
        while exploit
            if obj(V*(x1-(dp./df))) < e1
                x1    = V*(x1-(dp./df));
                e1    = obj(real(x1));
                x1    = V'*real(x1);
                nexpl = nexpl + 1;
            else
                exploit = false;
                pupdate(n,nexpl,e1,e0,'extrap');
            end
            
            % upper limit on the length of this loop: no don't do this
            if nexpl == (inner_loop*10)
                exploit = false;
            end
        end
        e0 = e1;
        x0 = x1;
            
        
        % print & plots success
        %------------------------------------------------------------------
        pupdate(n,nfun,e1,e0,'accept');
        if doplot; makeplot(V*x0(ip)); end
        n_reject_consec = 0;
        dff = [dff df];
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
            dx0    = real(dx);
            
            % loop the good params in effect-size order
            % accept on the fly (additive effects)
            %--------------------------------------------------------------
            improve1 = 1;
            while improve1
                thisgood = gp*0;
                % evaluate the 'good' parameters
                for i  = 1:length(gpi)
                    xnew             = real(V*x0);
                    xnew(gpi(PO(i))) = dx0(gpi(PO(i)));
                    xnew             = real(xnew);
                    enew             = obj(xnew);
                    % accept new error and parameters and continue
                    if enew < e0
                        dff = [dff (e0-enew)];
                        x0  = V'*real(xnew);
                        e0  = enew;
                        thisgood(gpi(PO(i))) = 1;
                    end
                end
                if any(thisgood)

                    % print & plot update
                    pupdate(n,nfun,e0,e0,'accept');
                    if doplot; makeplot(V*x0); end

                    % update step size for these params
                    red = red+V'*((V*red).*thisgood');
                    
                    % reset rejection counter
                    n_reject_consec = 0;
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
            
            pupdate(n,nfun,e0,e0,'reject');
            
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
    if aopt.order == 1; ldf = 30; dftol = 0.002;  end
    if aopt.order == 2; ldf = 50; dftol = 0.0001; end
    
    if length(dff) > ldf
        if var( dff(end-ldf:end) ) < dftol
            localminflag = 3;            
        end
    end
    if length(dff) > 31
        dff = dff(end-30:end);
    end
        
    if localminflag == 3
        fprintf('I think we''re stuck...stopping\n');
        X = V*real(x0(ip));
        F = e0;
        return;
    end
    
    % if 3 fails, reset the reduction term (based on the specified variance)
    if n_reject_consec == 3
        pupdate(n,nfun,e1,e0,'resetV');
        %red = red ./ max(red(:));
        red     = diag(pC);
        aopt.pC = V*red;
    end
    
    % stop at max iterations
    if n == maxit
        X = V*(x0(ip));
        F = e0;
        return;
    end
    
    % check for convergence
    if e0 <= criterion
        fprintf('Convergence.\n');
        X = V*real(x0(ip));
        F = e0;
        return;
    end
    
    % give up after 10 failed iterations
    if n_reject_consec == 5
        fprintf('Failed to converge... \nReturning best estimates.\n');
        X = V*real(x0(ip));
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
    V   = aopt.pC;
    Ord = aopt.order; 
    % compute jacobi
    %V = ones(size(x0));
    [J,ip] = jaco(@obj,x0,V,0,Ord);
end


end
function [X,F,Cp] = AOf(fun,x0,V,y,maxit,inner_loop,Q,criterion,min_df,mimo,order)
% Gradient/curvature descent based optimisation, primarily for model fitting
% [system identification & parameter estimation]
%
% The same AO.m, but this version minimises free energy, than than just error.
%
% Fit multivariate linear/nonlinear models of the forms:
%   Y0 = f(x) + e   ..or
%   e  = f(x)
%
% Y0 = empirical data (to fit)
% f  = model (function)
% x  = model parameters/inputs to be optimised
%
%
% Usage 1: to minimise a model fitting problem, using objective:
%--------------------------------------------------------------------------
%   y  = f(x)
%   ey = Y0 - y
%   ep = dx - x
%   qh = real(ey')*iS*real(ey) + imag(ey)'*iS*imag(ey);
%   F  = - ns*log(qh)/2 - ep'*ipC*ep/2;
%
% the usage is:
%   [X,F] = AO(fun,x0,V,y,maxit,inner_loop,Q,crit,min_df)
%
% minimum usage (using defaults):
%   [X,F] = AO(fun,x0,V,[y])
%
% fun        = function handle / anonymous function
% x0         = starting points (vector input to fun)
% V          = variances controlling each element of x0
% y          = Y0 / the data to fit, for computing the objective: e = sum(Y0 - y).^2
% maxit      = number of iterations (def=128) to restart descent
% inner_loop = num iters to continue on a specific descent (def=9999)
% Q          = optional precision matrix, e.g. e = sum( diag(Q).*(Y0-y) ).^2
% crit       = convergence value @(e=crit)
% min_df     = minimum change in function value (Error) to continue
%
% Usage 2: to minimise objective problems of the form:
%--------------------------------------------------------------------------
%   e = f(x)
%
% the usage is: [note: set y=0 if f(x) returns the error/objective to be minimised]
%   [X,F] = AO(fun,x0,V,0,maxit,inner_loop) 
%
% Example: Minimise Ackley function:
%--------------------------------------------------------------------------
%     arg min: e = 20*(1 - exp(-0.2*sqrt(0.5*(x1.^2 + x2.^2)))) ...
%                                - exp(0.5*(cos(2*pi*x1)...
%                                         + cos(2*pi*x2))) + exp(1);
%
%     [X,F] = AO(@ackley_fun,[3 .5],[1 1]/32,0,[],[],[],1e-6)
%
% Scroll to the bottom for a step-by-step description of how it works.
% See also ao_glm AO_dcm
%
% AS2019
% alexandershaw4@gmail.com
%
global aopt

if nargin < 11 || isempty(order);      order = 2;         end
if nargin < 10 || isempty(mimo);       mimo = 0;          end
if nargin < 9  || isempty(min_df);     min_df = 0;        end
if nargin < 8  || isempty(criterion);  criterion = 1e-2;  end
if nargin < 7  || isempty(Q);          Q = 1;             end
if nargin < 6  || isempty(inner_loop); inner_loop = 9999; end
if nargin < 5  || isempty(maxit);      maxit = 128;       end
if nargin < 4  || isempty(y);          y = 0;             end

% check functions, inputs, options...
%--------------------------------------------------------------------------
aopt.order   = order;    % first or second order derivatives [-1,0,1,2]
aopt.fun     = fun;      % (objective?) function handle
aopt.y       = y(:);     % truth / data to fit
aopt.Q       = Q;        % precision
aopt.history = [];       % error history when y=e & arg min y = f(x)
aopt.mimo    = mimo;     % flag compute derivs w.r.t multi-outputs

x0         = full(x0(:));
x1         = x0;
V          = full(V(:));
n          = 0;
iterate    = true;
doplot     = 1;

V  = smooth(V);
Vb = V;

% initial point plot
%--------------------------------------------------------------------------
if doplot
    makeplot(x0,x0);
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

[e0]       = obj(x0,x0,red,0);
e1         = e0;

aopt.pC  = V*red;      % store for derivative function access

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

% print start point
refdate();
pupdate(n,0,e0,e0,'start:');

% start loop
%==========================================================================
while iterate
    
    % counter
    %----------------------------------------------------------------------
    n = n + 1;    tic;
   
    % compute gradients & search directions
    %----------------------------------------------------------------------
    [e0,df0] = obj( V*x0(ip),x1,red,e1 );
    
    if mimo && ismatrix(Q)
        df0 = df0*HighResMeanFilt(full(Q),1,4) ;
    end
    
    % initial search direction (steepest) and slope
    %----------------------------------------------------------------------
    s   = -df0';
    d0  = -s'*s;                             % trace
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
                                
        % continue the ascent / descent
        if ~mimo; dx    = (V*x1(ip)+x3*s');        % MISO
        else;     dx    = (V*x1(ip)+x3*sum(s)');   % MIMO
        end
                
        % assess each new parameter individually, then find the best mix
        for nip = 1:length(dx)
            XX       = V*x0;
            XX(nip)  = dx(nip);
            DFE(nip) = obj(real(XX),x1,red,e1);
        end
        
        % compute improver-parameters
        gp  = double(DFE < e0); % e0
        gpi = find(gp);
        
        % only update select parameters
        ddx        = V*x0;
        ddx(gpi)   = dx(gpi);
        dx         = ddx;
        de         = obj(dx,x1,red,e1);
                
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
                
        % M-step (line search)
        %==================================================================
        
        %if the system is (locally?) linear, and we know what dp caused de
        %we can quickly exploit this to estimate the minimum on this descent
        %dp./de using an expansion
        %------------------------------------------------------------------
        exploit = true;
        nexpl   = 0;
        pupdate(n,nexpl,e1,e0,'descnd',toc);
        while exploit
            if obj(V*(x1+(dp./df)),x1,red,e1) < e1
                x1    = V*(x1+(dp./df));
                e1    = obj(real(x1),x1,red,e1);
                x1    = V'*real(x1);
                nexpl = nexpl + 1;
            else
                exploit = false;
                pupdate(n,nexpl,e1,e0,'finish',toc);
            end
            
            % upper limit on the length of this loop: no don't do this
            %if nexpl == (inner_loop*10)
            %    exploit = false;
            %end
        end
        e0 = e1;
        x0 = x1;
            
        
        % print & plots success
        %------------------------------------------------------------------
        pupdate(n,nfun,e1,e0,'accept',toc);
        if doplot; makeplot(V*x0(ip),x1); end
        n_reject_consec = 0;
        dff = [dff df];
    else
        
        % if didn't improve: what to do?
        %------------------------------------------------------------------
        pupdate(n,nfun,e1,e0,'adjust',toc);             
        
        % sample from improvers params in dx
        %------------------------------------------------------------------
        pupdate(n,nfun,e1,e0,'sample',toc);
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
                    enew             = obj(xnew,x1,red,e1);
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
                    pupdate(n,nfun,e0,e0,'accept',toc);
                    if doplot; makeplot(V*x0,x1); end

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
            
            pupdate(n,nfun,e0,e0,'reject',toc);
            
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
    if min_df ~= 0
        % user can define minimum DF
        ldf = 100; dftol = min_df;
    else
        % otherwise invoke some defaults
        if aopt.order == 1; ldf = 30; dftol = 0.002;  end
        if aopt.order == 2; ldf = 50; dftol = 0.0001; end
        if aopt.order == 0; ldf = 30; dftol = 0.002; end
        if aopt.order <  0; ldf = 30; dftol = 0.002; end
    end
    
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
        
        % return current best
        X = V*real(x0(ip));
        F = e0;
                
        % covariance estimation
        J       = df0;
        Pp      = real(J*1*J');
        Pp      = V'*Pp*V;
        Cp      = spm_inv(Pp + ipC);
        
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
        fprintf('Reached max iterations: stopping\n');
        
        % return current best
        X = V*real(x0(ip));
        F = e0;
                
        % covariance estimation
        J       = df0;
        Pp      = real(J*1*J');
        Pp      = V'*Pp*V;
        Cp      = spm_inv(Pp + ipC);
        
        return;
    end
    
    % check for convergence
    if e0 <= criterion
        fprintf('Convergence.\n');
        
        % return current best
        X = V*real(x0(ip));
        F = e0;
                
        % covariance estimation
        J       = df0;
        Pp      = real(J*1*J');
        Pp      = V'*Pp*V;
        Cp      = spm_inv(Pp + ipC);

        return;
    end
    
    % give up after 10 failed iterations
    if n_reject_consec == 5
        fprintf('Failed to converge... \nReturning best estimates.\n');
        
        % return current best
        X = V*real(x0(ip));
        F = e0;
                
        % covariance estimation
        J       = df0;
        Pp      = real(J*1*J');
        Pp      = V'*Pp*V;
        Cp      = spm_inv(Pp + ipC);

        return;
    end
    
end



end

function refdate()
fprintf('\n');

fprintf('| ITERATION     | FUN EVAL | CURRENT ERROR     | BEST ERROR SO FAR  | ACTION | TIME\n');
fprintf('|---------------|----------|-------------------|--------------------|--------|-------------\n');

end

function pupdate(it,nfun,err,best,action,varargin)

if nargin >= 6
    n = varargin{1};
    fprintf('| Main It: %04i | nf: %04i | Err: %04i | Best: %04i | %s | %d\n',it,nfun,err,best,action,n);
else
    fprintf('| Main It: %04i | nf: %04i | Err: %04i | Best: %04i | %s |\n',it,nfun,err,best,action);
end

end

function makeplot(x,ox)
% plot the function output (f(x)) on top of the thing we're ditting (Y)
%
%
global aopt

[Y,y] = GetStates(x);

if length(y)==1 && length(Y) == 1 && isnumeric(y)
    % memory based error trace when y==e
    aopt.history = [aopt.history y];
    plot(aopt.history,'ro');hold on;
    plot(aopt.history,'--b');hold off;drawnow;
    ylabel('Error^2');xlabel('Step'); title('Error plot');
else
%if iscell(Y)
    subplot(211)
    plot(spm_cat(Y),':','linewidth',2); hold on;
    plot(spm_cat(y)    ,'linewidth',2); hold off;
    grid on;grid minor;title('System Identification');
    subplot(212)
    bar([ x(:)-ox(:) ]);title('dParameter');drawnow;
%end
end
title('System Identification');

end

function [Y,y] = GetStates(x)
% - evaluates the model and returns it along with the stored data Y
%

global aopt

IS = aopt.fun;
P  = x(:)';

try    y  = IS(P); 
catch; y  = spm_vec(aopt.y)*0;
end
Y  = aopt.y;

end

function [e,J] = obj(x0,x1,Cp,e1)
% - compute the objective function - i.e. the sqaured error to minimise
% - also returns the parameter Jacobian
%

global aopt

IS = aopt.fun;
P  = x0(:)';

try    y  = IS(P); 
catch; y  = spm_vec(aopt.y)*0;
end
Y   = aopt.y;
Q   = aopt.Q;
ns  = size(y,1);
ep  = x1 - x0;
ipC = spm_inv(diag(Cp));
np  = length(x0);
iS  = 1;

if all(size(Q)>1)
    Q = diag(Q);
    %Q = 1 + (2 - 1) .* (Q - min(Q)) / ( max(Q) - min(Q) );
    %e = (spm_vec(Y) - spm_vec(y)).^2;
    %e = Q.*e;
    %e = sum(e.^2);
    
    ey    = spm_vec(Y) - spm_vec(y);
    qh    = real(ey')*iS*real(ey) + imag(ey)'*iS*imag(ey);
    F     = - ns*log(qh)/2 - ep'*ipC*ep/2;
        
else
    
    ey    = spm_vec(Y) - spm_vec(y);
    qh    = real(ey')*iS*real(ey) + imag(ey)'*iS*imag(ey);
    F     = - ns*log(qh)/2 - ep'*ipC*ep/2;

end

e = -F;

% error along output vector
er = spm_vec(y) - spm_vec(Y);

% this wraps obj to return only the third output for MIMOs
% when we want the derivatives w.r.t. each output 
function er  = inter(x0)
    [~,~,er] = obj(x0);
end

J = [];

% this hands off to jaco, which computes the derivatives
% - note this can be used differently for multi-output systems
if nargout == 2
    V    = aopt.pC;
    Ord  = aopt.order; 
    mimo = aopt.mimo;
    
    if ~mimo; [J,ip] = jaco(@obj,x0,V,0,Ord,{x1,Cp,e1} );    ... df[e]   /dx [MISO]
    else;     [J,ip] = jaco(@inter,x0,V,0,Ord,{x1,Cp,e1} );  ... df[e(k)]/dx [MIMO]
              %J = repmat(V,[1 size(J,2)])./J;
    end

    %[J,ip] = jaco(IS,P',V,0,Ord);   ... df/dx
    %J = repmat(V,[1 size(J,2)])./J;
end

end

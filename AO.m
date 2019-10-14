function [X,F,Cp] = AO(fun,x0,V,y,maxit,inner_loop,Q,criterion,min_df,mimo,order)
% Gradient/curvature descent based optimisation, primarily for model fitting
% [system identification & parameter estimation]
%
% The same AOf.m, but this version minimises sq-error, not free energy.
%
% Fit multivariate linear/nonlinear models of the forms:
%   Y0 = f(x) + e   ..or
%   e  = f(x)
%
% Y0 = empirical data (to fit)
% f  = model (function)
% x  = model parameters/inputs to be optimised
%
% The output of fun(x) can be either a single value or a vector. The
% derivatives can be returned w.r.t the objective (SSE, e.g. a MISO), or 
% w.r.t the error at each point along the output (e.g. when model 
% fitting, a MIMO). To invoke the latter, use the 10th input [0/1] to flag.
% When using this option, flag order == -1, which will use the complex
% conjugate method for the derivatives, otherwise probably use 2.
%
% Usage 1: to minimise a model fitting problem of the form:
%--------------------------------------------------------------------------
%   y = f(x)
%   e = sum(Y0 - y).^2
%
% the usage is:
%   [X,F] = AO(fun,x0,V,y,maxit,inner_loop,Q,crit,min_df,mimo,ordr)
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
% Q          = optional precision matrix of size == (fun(x),fun(x))
% crit       = convergence value @(e=crit)
% min_df     = minimum change in function value (Error) to continue
%              (set to -1 to switch off)
% mimo       = flag for a MIMO system: i.e. Y-fun(x) returns an error VECTOR
% order      = [-1, 0, 1, or 2] - see jaco.m
%
% Usage 2: to minimise objective problems of the form:
%--------------------------------------------------------------------------
%   e = f(x)
%
% the usage is: [note: set y=0 if f(x) returns the error/objective to be minimised]
%   [X,F] = AO(fun,x0,V,0,maxit,inner_loop) 
%
%
% Example: Minimise Ackley function:
%--------------------------------------------------------------------------
%     arg min: e = 20*(1 - exp(-0.2*sqrt(0.5*(x1.^2 + x2.^2)))) ...
%                                - exp(0.5*(cos(2*pi*x1)...
%                                         + cos(2*pi*x2))) + exp(1);
%
%     [X,F] = AO(@ackley_fun,[3 .5],[1 1]/32,0,[],[],[],1e-13,-1)
%
% Scroll to the bottom for a step-by-step description of how it works.
% See also ao_glm AO_dcm AOf jaco AOls
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
aopt.svd     = 0;        % use PCA on the gradient matrix

x0         = full(x0(:));
V          = full(V(:));
[e0]       = obj(x0);

n          = 0;
iterate    = true;
doplot     = 1;
Vb         = V;

% initial point plot
%--------------------------------------------------------------------------
if doplot
    setfig();
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
%V     = spm_svd(pC);
V     = eye(length(x0));    %turn off svd 
pC    = V'*pC*V;
ipC   = inv(spm_cat(spm_diag({pC})));
red   = diag(pC);

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

% now: x0 = V*p(ip) 
% --> this check was only valid for purely deterministic systems
%--------------------------------------------------------------------------
% if obj( V*p(ip) ) ~= e0
%     fprintf('Something went wrong during svd parameter reduction\n');
% else
%     % backup start points
%     X0 = x0;
%     % overwrite x0
%     x0 = p;
% end

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
    [e0,df0] = obj( V*x0(ip) );
    
    if mimo && ismatrix(Q)
        df0 = df0*HighResMeanFilt(full(Q),1,4) ;
    end

    % initial search direction (steepest) and slope
    %----------------------------------------------------------------------
    s   = -df0';    
    d0  = -s'*s;                             % trace
        
    if aopt.svd
        [uu,ss,vv] = spm_svd(d0);
        d0 = uu(:,1)*ss(1,1)*vv(:,1)';
    end
    
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
                                
        % continue the descent
        if ~mimo; dx    = (V*x1(ip)+x3*s');        % MISO
        else;     dx    = (V*x1(ip)+x3*sum(s)');   % MIMO
        end
        
        % assess each new parameter individually, then find the best mix
        for nip = 1:length(dx)
            XX       = V*x0;
            XX(nip)  = dx(nip);
            DFE(nip) = obj(real(XX));
        end
        
        % compute improver-parameters
        gp  = double(DFE < e0); % e0
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
            if obj(V*(x1+(dp./df))) < e1
                x1    = V*(x1+(dp./df));
                e1    = obj(real(x1));
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
        pupdate(n,nfun,e1,e0,'select',toc);             
        
        % sample from improvers params in dx
        %------------------------------------------------------------------
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
                    pupdate(n,nfun,e0,e0,'accept',toc);
                    if doplot; makeplot(V*x0,x1); end

                    % update step size for these params
                    red = red+V'*((V*red).*thisgood');       % CHANGE
                    
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
        pupdate(n,nfun,e1,e0,'resetv');
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

function setfig()

figure('Name','AO','Color',[.3 .3 .3],'InvertHardcopy','off','position',[1043 654 517 684]);
set(gcf, 'MenuBar', 'none');
set(gcf, 'ToolBar', 'none');
drawnow;
    
end

function makeplot(x,ox)
% plot the function output (f(x)) on top of the thing we're ditting (Y)
%
%
global aopt

[Y,y] = GetStates(x);

if ~isfield(aopt,'oerror')
    aopt.oerror = spm_vec(Y) - spm_vec(y);
end

former_error = aopt.oerror;
new_error    = spm_vec(Y) - spm_vec(y);

if length(y)==1 && length(Y) == 1 && isnumeric(y)
    ax        = gca;
    ax.YColor = [1 1 1];
    ax.XColor = [1 1 1];
    ax.Color  = [.3 .3 .3];hold on;
    
    % memory based error trace when y==e
    aopt.history = [aopt.history y];
    plot(aopt.history,'wo');hold on;
    plot(aopt.history,'w');hold off;grid on;
    ylabel('Error^2');xlabel('Step'); 
    title('AO: System Identification: Error','color','w','fontsize',18);
    drawnow;
else
%if iscell(Y)
    s(1) = subplot(311);
    plot(spm_cat(Y),'w:','linewidth',3); hold on;
    plot(spm_cat(y),     'linewidth',3,'Color',[1 .7 .7]); hold off;
    grid on;grid minor;title('AO: System Identification','color','w','fontsize',18);
    s(1).YColor = [1 1 1];
    s(1).XColor = [1 1 1];
    s(1).Color  = [.3 .3 .3];

    s(2) = subplot(312);
    %bar([former_error new_error]);
    plot(former_error,'w--','linewidth',3); hold on;
    plot(new_error,'linewidth',3,'Color',[1 .7 .7]); hold off;
    grid on;grid minor;title('Error Change','color','w','fontsize',18);
    s(2).YColor = [1 1 1];
    s(2).XColor = [1 1 1];
    s(2).Color  = [.3 .3 .3];
    
    
    s(3) = subplot(313);
    bar([ x(:)-ox(:) ],'FaceColor',[1 .7 .7],'EdgeColor','w');
    title('Parameter Change','color','w','fontsize',18);
    ax = gca;
    ax.XGrid = 'off';
    ax.YGrid = 'on';
    s(3).YColor = [1 1 1];
    s(3).XColor = [1 1 1];
    s(3).Color  = [.3 .3 .3];
    drawnow;
%end
end

aopt.oerror = new_error;

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

function [e,J,er] = obj(x0)
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
    
    % if square precision matrix was supplied, use this objective
    ey  = spm_vec(Y) - spm_vec(y);
    qh  = Q*(ey*ey')*Q';
    e   = sum(qh(:).^2);    
    
else
    
    % sse: sum of error squared
    e  = sum( (spm_vec(Y) - spm_vec(y)).^2 );
    
    % mse: mean squared error
    %e = (norm(spm_vec(Y)-spm_vec(y),2).^2)/numel(spm_vec(Y));
    
    % rmse: root mean squaree error
    %e = ( (norm(spm_vec(Y)-spm_vec(y),2).^2)/numel(spm_vec(Y)) ).^(1/2);
    
    % 1 - r^2 (bc. minimisation routine == maximisation)
    %e = 1 - ( corr( spm_vec(Y), spm_vec(y) ).^2 );
    
    % combination:
    %SSE = sum( (spm_vec(Y) - spm_vec(y)).^2 );
    %R2  = 1 - ( corr( spm_vec(Y), spm_vec(y) ).^2 );
    %e   = SSE + R2;
    
    %ey  = spm_vec(Y) - spm_vec(y);
    %qh  = real(ey)*real(ey') + imag(ey)*imag(ey');
    %e   = sum(qh(:).^2);
end

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
    
    if ~mimo; [J,ip] = jaco(@obj,x0,V,0,Ord);    ... df[e]   /dx [MISO]
    else;     [J,ip] = jaco(@inter,x0,V,0,Ord);  ... df[e(k)]/dx [MIMO]
    end

    %[J,ip] = jaco(IS,P',V,0,Ord);   ... df/dx
    %J = repmat(V,[1 size(J,2)])./J;
end


end



% 
% Pseudo-code on how it works:
%--------------------------------------------------------------------------
% - while improving:
%   * compute parameter gradients (1st/2nd order partial derivatives)
%   * derive descent path & initial step
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
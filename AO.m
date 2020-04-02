function [X,F,Cp,PP,Hist] = AO(fun,x0,V,y,maxit,inner_loop,Q,criterion,min_df,...
                                order,writelog,objective,ba,im,da,step_method)
% A Bayesian gradient/curvature descent-based optimisation, primarily for 
% model fitting [system identification & parameter estimation]. Objective
% function minimises free energy or the SSE.
%
% Fit multivariate linear/nonlinear models of the forms:
%   Y0 = f(x) + e   (e.g. state-space models) ..or
%   e  = f(x)       (e.g. f() is the objective function)
%
% Y0 = empirical data (to fit)
% f  = model (function)
% x  = model parameters/inputs to be optimised (treated as Gaussian
% distributions with variance V)
%
% ** Do >> AO('help') for an overview of the optimser.
%
% Usage 1: to minimise a model fitting problem of the form:
%--------------------------------------------------------------------------
%   y    = f(p)
%   e    = (data - y)            ... 
%   F(p) = log evidence(y) - divergence(p)   ... (Free Energy Objective Fun)
%
% the usage is:
%   [X,F,Cp,Pp,Hist] = AO(fun,x0,V,data,maxit,inner_loop,Q,crit,min_df,ordr,writelog,obj)
%
% minimum usage (using defaults):
%   [X,F] = AO(fun,x0,V,[y])
%
% INPUTS:
% fun        = function handle / anonymous function
% x0         = starting points (vector input to fun, mean of Gauss)
% V          = variances controlling each element of x0 (var of Gauss)
% data       = Y0 / the data to fit, for computing the objective function
% maxit      = number of iterations (def=128) to restart descent
% inner_loop = num iters to continue on a specific descent (def=9999)
% Q          = optional precision matrix (*DONT USE, [])
% crit       = objective convergence value 
% min_df     = minimum change in function value (Error) to continue
%              (set to -1 to switch off)
% order      = [-1, 0, 1, 2, 3, 4, 5] - ** see jaco.m for opts **
% writelog   = flag to write progress to command window (0) or a txt log (1)
% obj        = 'sse' 'free_energy' 'mse' 'rmse' 'logevidence' (def 'fe')
% ba         = BayesAdjust flag (def=0): curb parameter step by P(p) 
% im         = Include momentum (def=0): multiple steps in same dir=bigger step
% da         = DivergencAdjust (def=0): adjust step size based on divergence
% step_meth  = 1, 2 or 3 (def=3): 1=large, variance-controlled steps,
%                                 2=small GaussNewton steps
%                                 3=smaller (vanilla) steps (still var controlled)
%
% OUTPUTS:
% X   = posterior parameters
% F   = fit value (depending on objective function specified)
% CP  = parameter covariance
% Pp  = posterior probabilites
% H   = history
%
% Notes: 
% (1) in my testing minimising SSE seems to produce the best overall fits but
% free energy (log evidence - divergence) converges much faster. 
% (2) the free energy equation is
%       L(1) = spm_logdet(iS)*nq/2  - real(e'*iS*e)/2 - ny*log(8*atan(1))/2;  ...
%       L(2) = spm_logdet(ipC*Cp)/2 - p'*ipC*p/2;
%       F    = -sum(L);
%
% Usage 2: minimise objective problems of the form:
%--------------------------------------------------------------------------
%   e = f(x)
%
% note: set y=0 if f(x) returns the error/objective to be minimised
% usage:
%   [X,F] = AO(fun,x0,V,0,maxit,inner_loop, ... ) 
%
% For overview of optimisation method, do AO('help')
%
% Still to implement:
% - add momentum parameter
% - add Nesterov accelerated descent
%
% See also ao_glm AO_DCM jaco AO
%
% AS2019
% alexandershaw4@gmail.com
global aopt

% Print the description of steps and exit
%--------------------------------------------------------------------------
if nargin == 1 && strcmp(lower(fun),'help')
    PrintHelp(); return;
end

% Inputs & Defaults...
%--------------------------------------------------------------------------
if nargin < 16 || isempty(step_method);step_method = 3;   end
if nargin < 15 || isempty(da);         da = 0;            end
if nargin < 14 || isempty(im);         im = 0;            end
if nargin < 13 || isempty(ba);         ba = 0;            end
if nargin < 12 || isempty(objective);  objective = 'sse'; end
if nargin < 11 || isempty(writelog);   writelog = 0;      end   
if nargin < 10 || isempty(order);      order = 2;         end
if nargin < 9  || isempty(min_df);     min_df = 0;        end
if nargin < 8  || isempty(criterion);  criterion = 1e-2;  end
if nargin < 7  || isempty(Q);          Q = 1;             end
if nargin < 6  || isempty(inner_loop); inner_loop = 9999; end
if nargin < 5  || isempty(maxit);      maxit = 128;       end
if nargin < 4  || isempty(y);          y = 0;             end

% Set up log if requested
persistent loc ;
if writelog
    name = datestr(now); name(name==' ') = '_';
    name = [char(fun) '_' name '.txt'];
    loc  = fopen(name,'w');
else
    loc = 1;
end

% check functions, inputs, options...
%--------------------------------------------------------------------------
%aopt         = [];       % reset
aopt.order   = order;    % first or second order derivatives [-1,0,1,2]
aopt.fun     = fun;      % (objective?) function handle
aopt.y       = y(:);     % truth / data to fit
aopt.pp      = x0(:);    % starting parameters
aopt.Q       = Q;        % precision matrix: e = Q*(ey*ey')
aopt.history = [];       % error history when y=e & arg min y = f(x)
aopt.memory  = 0;        % incorporate previous gradients when recomputing
aopt.fixedstepderiv = 1; % fixed or adjusted step for derivative calculation
aopt.ObjectiveMethod = objective; % 'sse' 'fe' 'mse' 'rmse' (def sse)

BayesAdjust = ba; % Bayes-esque adjustment (constraint) of the GD-predicted parameters
                  % (converges slower but might be more careful)
IncMomentum = im; % Observe and use momentum data            
DivAdjust   = da; % Divergence adjustment

% % if no prior guess for parameters step sizes (variances) find step sizes
% % whereby each parameter has similar effect size w.r.t. error
% if nargin < 3 || isempty(V)
%     % iterates: v = v./abs(J)
%     V = FindOptimumStep(x0,V);
% end

% parameter and step vectors
x0  = full(x0(:));
V   = full(V(:));
v   = V;
pC  = diag(V);

% variance (in reduced space)
%--------------------------------------------------------------------------
%V     = spm_svd(pC);
V     = eye(length(x0));    %turn off svd 
pC    = V'*pC*V;
ipC   = inv(spm_cat(spm_diag({pC})));
red   = diag(pC);

aopt.pC  = V*red;      % store for derivative & objective function access
aopt.ipC = ipC; 

% initial objective value
[e0]       = obj(x0);
n          = 0;
iterate    = true;
doplot     = 1;
Vb         = V;

% initial error plot(s)
%--------------------------------------------------------------------------
if doplot
    setfig(); makeplot(x0,x0);
end

% initialise counters
%--------------------------------------------------------------------------
n_reject_consec = 0;
search          = 0;

% parameters (in reduced space)
%--------------------------------------------------------------------------
np    = size(V,2); 
p     = [V'*x0];
ip    = (1:np)';
Ep    = V*p(ip);

dff          = []; % tracks changes in error over iterations
localminflag = 0;  % triggers when stuck in local minima

if BayesAdjust; fprintf('Using BayesAdjust option\n'); end
if IncMomentum; fprintf('Using Momentum option\n');    end
if DivAdjust  ; fprintf('Using Divergence option\n');  end
fprintf('Using step-method %d\n',step_method);

% print start point - to console or logbook (loc)
refdate(loc);
pupdate(loc,n,0,e0,e0,'start:');

% start loop
%==========================================================================
while iterate
    
    % counter
    %----------------------------------------------------------------------
    n = n + 1;    tic;
   
    pupdate(loc,n,0,e0,e0,'grdnts',toc);
    
    aopt.pp = x0;
    
    % compute gradients & search directions
    %----------------------------------------------------------------------
    aopt.updatej = true;
    [e0,df0] = obj( V*x0(ip) );
    e0       = obj( V*x0(ip) );
        
    %[df0,e0] = spm_diff(@obj,x0,1);
         
    % print end of gradient computation (just so we know it's finished)
    pupdate(loc,n,0,e0,e0,'-fini-',toc); 
    
    % initial search direction (steepest) and slope
    %----------------------------------------------------------------------  
    search_method = step_method;
    
    switch search_method
        case 1
            
            J      = -df0';
            dFdpp  = -(J'*J);
            
            % Initial step
            x3  = V*red(ip)./(1-dFdpp);
                        
            % Leading (gradient) components
            [uu,ss,vv] = spm_svd(x3);
            nc = min(find(cumsum(diag(full(ss)))./sum(diag(ss))>=.95));
            x3 = full(uu(:,1:nc)*ss(1:nc,1:nc)*vv(:,1:nc)');
        
        case 2
            
            J     = -df0';
            dFdp  = -real(J'*e0);
            dFdpp = -real(J'*J);
            
        case 3
            
            J      = -df0;
            dFdpp  = -(J'*J);
            
            % Initial step (Rasmussen method)
            x3  = V*(4*red(ip))./(1-dFdpp);
    end
    
    % Log start of iteration
    Hist.e(n) = e0;
    Hist.p{n} = x0;
    Hist.J{n} = df0;
    
    % make copies of error and param set for inner while loops
    x1  = x0;
    e1  = e0;
        
    % start counters
    improve = true;
    nfun    = 0;
    
    % iterative descent on this slope
    %======================================================================
    while improve
        
        % descend while de < e1
        nfun = nfun + 1;
                                        
        % Parameter Step
        %------------------------------------------------------------------
        if search_method == 1
            %dx    = (V*x1(ip)+x3*J');
            dx    = (V*x1(ip)+ x3*J'); 
        elseif search_method == 2 
            ddx   = spm_dx(dFdpp,dFdp,{red})';
            dx    = x1 - ddx;
        elseif search_method == 3
            dx    = ( V*x1(ip) + (x3.*J) ); % (Rasmussen w/ diff p steps)  
        end
                
        % (option) Momentum inclusion
        %------------------------------------------------------------------
        if n > 2 && IncMomentum
            % The idea here is that we can have more confidence in
            % parameters that are repeatedly updated in the same direction,
            % so we can take bigger steps for those parameters
            imom = sum( diff(full(spm_cat(Hist.p))')' > 0 ,2);
            dmom = sum( diff(full(spm_cat(Hist.p))')' < 0 ,2);
            
            timom = imom >= (2);
            tdmom = dmom >= (2);
            
            moments = (timom .* imom) + (tdmom .* dmom);
            
            if any(moments)
               % parameter update
               ddx = dx - x1;
               dx  = dx + ( ddx .* (moments./n) );
            end
        end
        
        % (option) Probability constraint
        %------------------------------------------------------------------
        if BayesAdjust
            % Probabilities of these (predicted) values actually belonging to
            % the prior distribution as a bound on parameter step
            % (with arbitrary threshold)
            ddx = dx - x1;
            ppx = spm_Ncdf(abs(x0),abs(dx),sqrt(red)); ppx(ppx<.2) = 0.2;
                    
            % parameter update
            dx = x1 + (ddx.*ppx);
            
            % mock some distributions to visualise changes
            for i = 1:length(x1)
                pd(i)   = makedist('normal','mu',abs(x1(i)),'sigma', ( red(i) ));
                pr(i,:) = pdf(pd(i),abs(x1(i))-10:abs(x1(i))+10);
                po(i,:) = pdf(pd(i),abs(dx(i))-10:abs(dx(i))+10);
            end
            BayesPlot(-10:10,pr,po);
            
        end
        
        % (option) Divergence adjustment
        %------------------------------------------------------------------
        for i = 1:length(x1)
            pd(i)  = makedist('normal','mu', (x1(i)),'sigma', ( red(i) ));
            pdt(i) = 1-cdf( pd(i), (x1(i)));
            pt(i)  = 1-cdf( pd(i), (dx(i)));
        end
        if DivAdjust
            
            % Divergence of the prob distribution 
            PQ  = pt(:).*log( pt(:)./pdt(:) );  PQ(isnan(PQ)) = 0;
            iPQ = 1./(1 - PQ);

            % parameter update
            ddx = dx(:) - x1(:);
            dx  = x1(:) + ddx.*iPQ(:);
        end
        
        
        % Evaluate parameter(s) step (ascent)
        %------------------------------------------------------------------
        if obj(dx) < obj(x1)
            % Don't perform checks, assume all f(dx[i]) <= e1
            gp  = ones(1,length(x0));
            gpi = 1:length(x0);
            de  = obj(dx);
            DFE = ones(1,length(x0))*de; 

        else
            % Assess each new (extrapolated) parameter estimate individually, 
            % update only improvers
            for nip = 1:length(dx)
                XX       = V*x0;
                XX(nip)  = dx(nip);
                DFE(nip) = obj(real(XX));
            end

            % Identify improver-parameters
            gp  = double(DFE < e0); % e0
            gpi = find(gp);

            % Only update select parameters
            ddx        = V*x0;
            ddx(gpi)   = dx(gpi);
            dx         = ddx;
            de         = obj(dx);
        end
        
%         % Check the new parameter estimates (dx)?
%         Check = 1;
%         
%         if Check == 1
%                         
%             % Assess each new (extrapolated) parameter estimate individually, 
%             % update only improvers
%             for nip = 1:length(dx)
%                 XX       = V*x0;
%                 XX(nip)  = dx(nip);
%                 DFE(nip) = obj(real(XX));
%             end
% 
%             % Identify improver-parameters
%             gp  = double(DFE < e0); % e0
%             gpi = find(gp);
% 
%             % Only update select parameters
%             ddx        = V*x0;
%             ddx(gpi)   = dx(gpi);
%             dx         = ddx;
%             de         = obj(dx);
%             
%         elseif Check == 0
%             
%             % Don't perform checks, assume all f(dx[i]) <= e1
%             gp  = ones(1,length(x0));
%             gpi = 1:length(x0);
%             de  = obj(dx);
%             DFE = ones(1,length(x0))*de; 
%             
%         end
        
        % Tolerance on update error as function of iteration number
        % - this can be helpful in functions with lots of local minima
        % i.e. bad step required before improvement
        %etol = e1 * ( ( 0.5./n ) ./(nfun.^2) );
        etol = 0; % none
        
        if de  < ( obj(x1) + etol )
            
            if nfun == 1; pupdate(loc,n,0,de,e1,'improv',toc); end
            
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
        
        % upper limit on the length of this loop (force recompute dfdx)
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
                
        % (like a line search?)
        %==================================================================
        
        %if the system is (locally?) linear, and we know what dp caused de
        %------------------------------------------------------------------
        exploit = true;
        nexpl   = 0;
        pupdate(loc,n,nexpl,e1,e0,'descnd',toc);
        while exploit
            if obj(V*(x1+(-dp./-df))) < e1
                x1    = V*(x1+(-dp./-df));
                e1    = obj(real(x1));
                x1    = V'*real(x1);
                nexpl = nexpl + 1;
            else
                exploit = false;
                pupdate(loc,n,nexpl,e1,e0,'finish',toc);
            end
            
            % upper limit on the length of this loop: no don't do this
            if nexpl == (inner_loop)
                exploit = false;
            end
        end
        
        % Update best-so-far estimates
        e0 = e1;
        x0 = x1;
            
        % print & plots success
        %------------------------------------------------------------------
        pupdate(loc,n,nfun,e1,e0,'accept',toc);
        if doplot; makeplot(V*x0(ip),x1); end
        n_reject_consec = 0;
        dff = [dff df];
    else
        
        % if didn't improve: what to do?
        %------------------------------------------------------------------
        pupdate(loc,n,nfun,e1,e0,'select',toc);             
        
        % sample from improvers params in dx
        %------------------------------------------------------------------
        thisgood = gp*0;
        if any(gp)
            
            % sort good params by improvement amount *OR probability*
            %--------------------------------------------------------------
            %[~,PO] = sort(DFE(gpi),'descend'); % or ascend? ..
            %[~,PO] = sort(pt(gpi),'descend');
            
            % update p's causing biggest improvment in fe while maintaining highest P(p)
            [~,PO] = sort(pt(gpi).*DFE(gpi),'ascend'); % DFE(gpi) = dp that cause imrpvoements
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
                    pupdate(loc,n,nfun,e0,e0,'accept',toc);
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
            
            pupdate(loc,n,nfun,e0,e0,'reject',toc);
            
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
        if aopt.order == 2; ldf = 800; dftol = 0.0001; end
        if aopt.order == 0; ldf = 30; dftol = 0.002; end
        if aopt.order <  0; ldf = 30; dftol = 0.002; end
        if aopt.order >  2; ldf = 800; dftol = 0.0001; end
        
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
        %fprintf(loc,'I think we''re stuck...stopping\n');
        
        % return current best
        X = V*real(x0(ip));
        F = e0;
                
        % covariance estimation
        J       = df0;
        Pp      = real(J*1*J');
        Pp      = V'*Pp*V;
        Cp      = spm_inv(Pp + ipC);
        
        PP = BayesInf(x0,Ep,diag(red));
        
        if writelog;fclose(loc);end
        return;
    end
    
    % if 3 fails, reset the reduction term (based on the specified variance)
    if n_reject_consec == 3
        pupdate(loc,n,nfun,e1,e0,'resetv');
        %red = red ./ max(red(:));
        red     = diag(pC);
        aopt.pC = V*red;
    end
    
    % stop at max iterations
    if n == maxit
        fprintf(loc,'Reached max iterations: stopping\n');
        
        % return current best
        X = V*real(x0(ip));
        F = e0;
                
        % covariance estimation
        Cp = aopt.Cp;
        PP = BayesInf(x0,Ep,diag(red));
        
        if writelog;fclose(loc);end
        return;
    end
    
    % check for convergence
    if e0 <= criterion
        fprintf(loc,'Convergence.\n');
        
        % return current best
        X = V*real(x0(ip));
        F = e0;
                
        % covariance estimation
        Cp = aopt.Cp;
        PP = BayesInf(x0,Ep,diag(red));
        
        if writelog;fclose(loc);end
        return;
    end
    
    % give up after 10 failed iterations
    if n_reject_consec == 5
        fprintf(loc,'Failed to converge...\n');
        
            % return current best
            X = V*real(x0(ip));
            F = e0;

            % covariance estimation
            Cp = aopt.Cp;
            PP = BayesInf(x0,Ep,diag(red));

            if writelog;fclose(loc);end
            return;
     end
end
    
end

function refdate(loc)
fprintf(loc,'\n');

fprintf(loc,'| ITERATION     | FUN EVAL | CURRENT ERROR     | BEST ERROR SO FAR  | ACTION | TIME\n');
fprintf(loc,'|---------------|----------|-------------------|--------------------|--------|-------------\n');

end

function prinfo(loc,it,nfun,nc,ncs)

fprintf(loc,'| Main It: %04i | nf: %04i | Selecting components: %01i of %01i\n',it,nfun,nc,ncs);

end

function pupdate(loc,it,nfun,err,best,action,varargin)

if nargin >= 7
    n = varargin{1};
    fprintf(loc,'| Main It: %04i | nf: %04i | Err: %04i | Best: %04i | %s | %d\n',it,nfun,err,best,action,n);
else
    fprintf(loc,'| Main It: %04i | nf: %04i | Err: %04i | Best: %04i | %s |\n',it,nfun,err,best,action);
end

end

function setfig()

figure('Name','AO','Color',[.3 .3 .3],'InvertHardcopy','off','position',[1088 122 442 914]);
set(gcf, 'MenuBar', 'none');
set(gcf, 'ToolBar', 'none');
drawnow;
    
end

function BayesPlot(x,pr,po)

s(3) = subplot(3,2,5);
imagesc(pr);
s(4) = subplot(3,2,6);
imagesc(po);

%bar([ x(:)-ox(:) ],'FaceColor',[1 .7 .7],'EdgeColor','w');
%title('Parameter Change','color','w','fontsize',18);

ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'on';
s(3).YColor = [1 1 1];
s(3).XColor = [1 1 1];
s(3).Color  = [.3 .3 .3];
s(4).YColor = [1 1 1];
s(4).XColor = [1 1 1];
s(4).Color  = [.3 .3 .3];
drawnow;
end

function makeplot(x,ox)
% plot the function output (f(x)) on top of the thing we're ditting (Y)
%
%
global aopt

[Y,y] = GetStates(x);

% Restrict plots to real values only - just for clarity really
% (this doesn't mean the actual model output is not complex)
Y = spm_unvec( real(spm_vec(Y)), Y);
y = spm_unvec( real(spm_vec(y)), y);

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
% elseif iscell(Y) && all( size(Y{1}) > 1)
%     % Matrix representation (imagesc)
%     py = spm_unvec( spm_vec(y), Y);
%     subplot(321);imagesc(Y{1});
%     subplot(322);imagesc(py{1});
%     title('AO: System Identification','color','w','fontsize',18);
%     
%     s(2) = subplot(3,2,[3 4]);
%     %bar([former_error new_error]);
%     plot(former_error,'w--','linewidth',3); hold on;
%     plot(new_error,'linewidth',3,'Color',[1 .7 .7]); hold off;
%     grid on;grid minor;title('Error Change','color','w','fontsize',18);
%     s(2).YColor = [1 1 1];
%     s(2).XColor = [1 1 1];
%     s(2).Color  = [.3 .3 .3];
%     
%     
%     s(3) = subplot(3,2,[5 6]);
%     bar([ x(:)-ox(:) ],'FaceColor',[1 .7 .7],'EdgeColor','w');
%     title('Parameter Change','color','w','fontsize',18);
%     ax = gca;
%     ax.XGrid = 'off';
%     ax.YGrid = 'on';
%     s(3).YColor = [1 1 1];
%     s(3).XColor = [1 1 1];
%     s(3).Color  = [.3 .3 .3];
%     drawnow;
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

try    y  = IS(P); 
catch; y  = spm_vec(aopt.y)*0;
end
Y  = aopt.y;

end

function [e,J,er,mp,Cp] = obj(x0)
% - compute the objective function - i.e. the sqaured error to minimise
% - also returns the parameter Jacobian,  error (vector), model prediction
% (vector) and covariance
%

global aopt

% if ~isfield(aopt,'computeiCp')
%     % Compute inverse covariance - on first call trigger this, but it gets 
%     % switched off during objective calls during derivative calculation
%     aopt.computeiCp = 1;
% end

method = aopt.ObjectiveMethod;

IS = aopt.fun;
P  = x0(:)';

try    y  = IS(P); 
catch; y  = spm_vec(aopt.y)*0+inf;
end

Y  = aopt.y;
Q  = aopt.Q;

   
if ~isfield(aopt,'J')
    aopt.J = ones(length(x0),length(spm_vec(y)));
end
if isfield(aopt,'J') && isvector(aopt.J)
    aopt.J = repmat(aopt.J,[1 length(spm_vec(y))]);
end


%switch lower(method)
%    case {'free_energy','fe','freeenergy','logevidence'};

        % Free Energy Objective Function: F(p) = log evidence - divergence
        %----------------------------------------------------------------------
        Q  = spm_Ce(1*ones(1,length(spm_vec(y))));
        h  = sparse(length(Q),1) - log(var(spm_vec(Y))) + 4;
        iS = sparse(0);

        for i  = 1:length(Q)
            iS = iS + Q{i}*(exp(-32) + exp(h(i)));
        end

        ny  = length(spm_vec(y));
        nq  = ny ./ length(Q);
        e   = spm_vec(Y) - spm_vec(y);
        ipC = aopt.ipC;
        warning off; % don't warn abour singularity

        %if aopt.computeiCp
            Cp  = spm_inv( (aopt.J*iS*aopt.J') + ipC );
        %else
        %    Cp = aopt.Cp;
        %end

        warning on
        p   = ( x0(:) - aopt.pp(:) );

        if any(isnan(Cp(:))) 
            Cp = Cp;
        end

        L(1) = spm_logdet(iS)*nq/2  - real(e'*iS*e)/2 - ny*log(8*atan(1))/2;            ...
        L(2) = spm_logdet(ipC*Cp)/2 - p'*ipC*p/2;
       %L(3) = spm_logdet(ihC*Ch)/2 - d'*ihC*d/2; % no hyperparameters
       
       aopt.Cp = Cp;
       
switch lower(method)
    case {'free_energy','fe','freeenergy','logevidence'};
        F    = sum(L);
        e    = (-F);

        %aopt.Cp = Cp;
        %aopt.Q  = iS;

        if strcmp(lower(method),'logevidence')
            % for log evidence, ignore the parameter term
            % its actually still an SSE measure really
            F = L(1);
            e = -F;
        end
    
        % Other Objective Functions
        %------------------------------------------------------------------ 
        case 'sse'
            % sse: sum of error squared
            e  = sum( (spm_vec(Y) - spm_vec(y) ).^2 ); e = abs(e);
            
        case 'sse2' % sse robust to complex systems
            e  = sum(sum( ( spm_vec(Y)-spm_vec(y)').^2 ));
            e  = real(e) + imag(e);
    
        case 'mse'
            % mse: mean squared error
            e = (norm(spm_vec(Y)-spm_vec(y),2).^2)/numel(spm_vec(Y));

        case 'rmse'
            % rmse: root mean squaree error
            e = ( (norm(spm_vec(Y)-spm_vec(y),2).^2)/numel(spm_vec(Y)) ).^(1/2);

        case {'correlation','corr','cor','r2'}
            % 1 - r^2 (bc. minimisation routine == maximisation)
            e = 1 - ( corr( spm_vec(Y), spm_vec(y) ).^2 );

        case 'combination'
            % combination:
            SSE = sum( ((spm_vec(Y) - spm_vec(y)).^2)./sum(spm_vec(Y)) );
            R2  = 1 - abs( corr( spm_vec(Y), spm_vec(y) ).^2 );
            e   = SSE + R2;

        case {'logistic' 'lr'}
            % logistic optimisation 
            e = -( spm_vec(Y)'*log(spm_vec(y)) + (1 - spm_vec(Y))'*log(1-spm_vec(y)) );


            % complex output models
            %ey  = spm_vec(Y) - spm_vec(y);
            %qh  = real(ey)*real(ey') + imag(ey)*imag(ey');
            %e   = sum(qh(:).^2);
end
%end

% error along output vector
er = spm_vec(y) - spm_vec(Y);
mp = spm_vec(y);

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
    
    if aopt.fixedstepderiv == 1
        V = (~~V)*1e-3;
        %V = (~~V)*exp(-8);
    end
    
    %aopt.computeiCp = 0; % don't re-invert covariance for each p of dfdp
    
    [J,ip] = jaco(@obj,x0,V,0,Ord);   ... df[e]   /dx [MISO]

    %aopt.computeiCp = 1;
    
    % store for objective function
    if  aopt.updatej
        aopt.J       = J;
        aopt.updatej = false;     % (when triggered always switch off)
    end
    
    % accumulate gradients / memory of gradients
    if aopt.memory
        try
            J       = J + (aopt.pJ/2) ;
            aopt.pJ = J;
        catch
            aopt.pJ = J;
        end
    end
    
    %[J,ip] = jaco(IS,P',V,0,Ord);   ... df/dx
    %J = repmat(V,[1 size(J,2)])./J;
end


end

function [Pp,dP] = BayesInf(Ep,pE,Cp)
% Returns the conditional probability of parameters, given 
% posteriors, priors and  covariance
%
% AS

Qp = Ep;
Ep = spm_vec(Ep);
pE = spm_vec(pE);

if isstruct(Cp)
    Cp = diag(spm_vec(Cp));
end

dP = spm_unvec(Ep-pE,Qp);

% Bayesian inference {threshold = prior} 
warning('off','SPM:negativeVariance');
dp  = spm_vec(Ep) - spm_vec(pE);
Pp  = spm_unvec(1 - spm_Ncdf(0,abs(dp),diag(Cp)),Qp);
warning('on', 'SPM:negativeVariance');
 
end

function V = FindOptimumStep(x0,v)
global aopt

fprintf('Auto computing parameter step sizes...(wait)\n');tic;

% initialise at 1/8 - arbitrary
% this is equivalent to:
% dx[i] = x[i] + ( x[i]*1/8 )
%
% using the n-th order numerical derivatives as a measure of parameter
% effect size, find starting step sizes whereby all parameters are
% effective / equally balanced
if nargin < 2 || isempty(v)
    v = ones( size(x0) )/8;
end
n = 0;

aopt.ipC   = inv(spm_cat(spm_diag({diag(v)})));

[J,ip] = jaco(@obj,x0,v,0,aopt.order);
n = 0;

while var(J) > 0.15
    vj = v./abs(J);
    vj(isinf(vj))=0;
    vj(isnan(vj))=0;
    vj = full(vj);
    v  = pinv( vj )' ;
    [J,ip] = jaco(@obj,x0,v,0,aopt.order);
    n = n + 1;
end

V = v;

fprintf('Finished computing step sizes in %d iterations (%d s)\n',n,round(toc));

end

function PrintHelp()

fprintf(['AO implements a gradient descent optimisation that incorporates \n' ...
    'curvature information (like a GaussNewton). Each parameter of f() is \n' ...
    'treated as a Gaussian distribution with variance v. Step sizes are controlled \n' ...
    'by the variance term and calculated using standard method. \n' ...
    'Additional constraint options can be included (e.g. Divergence based). \n'  ...
    'When the full gradient prediction doesnt improve the objective, the routine\n' ...
    'picks a subset of parameters that do. This selection is based on the probability\n' ...
    'of the (GD predicted) new parameter value coming from the prior distribution.\n' ...
    '\nIn model fitting scenarios, the code is set up so that you pass the model\n'...
    'function (fun), parameters and also the data you want to fit. The advantage of\n'...
    'this is that the algo can compute the objective function. This is necessary\n'...
    'if you want to minimising free energy (but also has SSE, MSE, RMSE etc).\n' ...
    '\nOutputs are the posteriors (means), objective value (F), (co)variance (CP),\n' ...
    'posterior probabilities (Pp) and a History structure (Hist) that contains the\n'...
    'parameters, objective values and gradients from each iteration of the algorithm.\n' ...
    '\nThe code makes use of the fabulous SPM toolbox functions for things like\n' ...
    'vectorising and un-vectorising - so SPM is a dependency. This means that\n' ...
    'the data you''re fitting (y in AO(fun,p,v,y) ) and the output of fun(p)\n'...
    'can be of whatever data type you like (vector, matrix, cell etc).\n' ...
    '\nIf you want to speed up the algorithm, look around line 211 and change\n'...
    'search_method from 3 to 1. This method extrapolates further and can fit data\n'...
    'quite a bit faster (and with better fits). However, it is also prone to\n'...
    'pushing parameters to extremes, which is often bad in model fitting when you\n'...
    'plan to make some parameter inference.\n']);



end
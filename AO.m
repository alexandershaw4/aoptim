function [X,F,Cp,PP,Hist] = AO(funopts)
% A gradient/curvature descent optimisation routine, designed primarily 
% for nonlinear model fitting / system identification & parameter estimation. 
% 
% The objective function minimises the free energy (~ ELBO) or the SSE.
%
% Fit multivariate nonlinear models of the forms:
% 1)  Y0 = f(x) + e   (e.g. generative models) ..or
% 2)  e  = f(x)       (e.g. f() is the objective function)
%
% To optimise models using the FE, you need your code set up as per (1).
%
% Y0 = empirical data (to fit)
% f  = model (function)
% x  = model parameters/inputs to be optimised 
%      (treated as Gaussians with variance V)
%
% Given the model function (f), parameters (x) and data (y), AO computes the
% objective function, F:
%
%   F(x) = log evidence(e,y) - divergence(x) 
%
% For a multivariate function f(x) where x = [p1 .. pn]' the ascent scheme
% is:
%
%   x[p,t+1] = x[p,t] +         a[p]          *-dFdx[p]
%
% Note the use of different step sizes, a[p], for each parameter.
% Optionally, a second GD can be computed to find the best step size a[p]:
%
%   x[p,t+1] = x[p,t] + ( a[p] + b*-dFda[p] ) *-dFdx[p] ... where b = 1e-4
%
% For the new step sizes, the Armijo-Goldstein condition is enforced. The
% secondary optimisation (of a) is only invoked is the full gradient
% prediction initially inproved F, since it is computationally intensive.
%
% dFdx[p] are the partial derivatives of F, w.r.t parameters, p. (Note F = 
% the objective function and not 'f' - your function). See jaco.m for options, 
% although by default these are computed using a finite difference 
% approximation of the curvature, which retains the sign of the gradient:
%
% f0 = F(x[p]+h) 
% fx = F(x[p]  )
% f1 = F(x[p]-h) 
%
% j(p,:) =        (f0 - f1) / 2h  
%          ----------------------------
%            (f0 - 2 * fx + f1) / h^2  
%
% nb. - depending on the specified step method, a[] is computed from j & V
% using variations on:      (where V is the variance of each param)
% 
%  J      = -j ;
%  dFdpp  = -(J'*J);
%  a      = (V)./(1-dFdpp);   
% 
% If step_method = 3, dFdpp is a small number and this reduces to a simple
% scale on the stepsize - i.e. a = V./scale. However, if step_method = 1, J
% is transposed, such that dFdpp is np*np. This leads to much bigger
% parameter steps & can be useful when your start positions are a long way
% from the solution. In either case, note that the variance term V[p] on the
% parameter distribution is a scaled version of the step size. 
%
% For each iteration of the ascent:
% 
%   dx[p]  = x[p] + a[p]*-dFdx
%
% There are 4 possible outcomes for dx. In order:
%   
%   (1.) F(dx[1:np]) < F(x[1:np]) - i.e. all dx improve F ... keep going.
%   (2.) If (1.) fails, evaluate each dx[ip] and form subset, R, to update:
%        F(dx[R]) < F(x) ... keep going.
%   (3.) If 2 fails, do a very selective update: 
%        - Compute prob of each dx[ip] coming from Gauss defined by x[ip] & V[ip].
%        - Sort params by Prob(ip)*Improvment(ip)
%        - Iteratively accept each individual param update in order until
%          no more improvement.
%   (4.) If 1:3 fail, then reject dx, expand V and recompute derivatives. 
%        Go to (1).
%
% Usage: to minimise a model fitting problem of the form:
%==========================================================================
%   y    = f(p)                              ... f = function, p = params
%   e    = (data - y)                        ... error = data - f(p)
%   F(p) = log evidence(e,y) - divergence(p) ... objective function F
%
% the long usage is:
%   [X,F,Cp,Pp,Hist] = AO(f,x0,V,data,maxit,inner_loop,Q,crit,min_df,ordr,...
%                                writelog,obj,ba,im,step_meth)
%
% minimum usage (using defaults):
%   [X,F] = AO(fun,x0,V,[y])
%
%
% INPUTS:
%-------------------------------------------------------------------------
% To call this function using an options structure (recommended), do this:
%-------------------------------------------------------------------------
% opts = AO('options');     % get the options struct
% opts.fun = @myfun         % fill in what you want...
% opts.x0  = [0 1];         % start param value
% opts.V   = [1 1]/8;       % variances for each param
% opts.y   = [...];         % data to fit - .e.g y = f(p) + e
% opts.macit       = 125;   % max num iterations 
% opts.step_method = 1;     % step method - 1 (big), 3 (small) and 4 (vanilla).
% opts.im    = 1;           % flag to include momentum of parameters
% opts.order = 2;           % partial derivate fun option (see jaco.m)
% opts.hyperparameters = 0; % do an grad asc on the precision (see spm_nlsi_GN)
% opts.BTLineSearch    = 1; % back tracking line search
% opts.criterion    = -300  % convergence threshold
%
% [X,F] = AO(opts);       % call the optimser, passing the opttions struct
%
% OUTPUTS:
%-------------------------------------------------------------------------
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
% *NOTE THAT, FOR FREE ENERGY OBJECTIVE, THE OUTPUT F-VALUE IS SIGN FLIPPED!
%
% *If the optimiser isn't working well, try making V smaller!
%
% See also ao_glm AO_DCM jaco AO
%
% AS2019
% alexandershaw4@gmail.com
global aopt

% Print the description of steps and exit
%--------------------------------------------------------------------------
if nargin == 1 && strcmp(lower(funopts),'help')
    PrintHelp(); return;
end
if nargin == 1 && strcmp(lower(funopts),'options');
    X = DefOpts; return;
end

% Inputs & Defaults...
%--------------------------------------------------------------------------
if isstruct(funopts)
   parseinputstruct(funopts);
else
   fprintf('You have to supply a funopts input struct now...\nTry AO(''options'')\n');
   return;
end

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
%aopt         = [];      % reset
aopt.order   = order;    % first or second order derivatives [-1,0,1,2]
aopt.fun     = fun;      % (objective?) function handle
aopt.y       = y(:);     % truth / data to fit
aopt.pp      = x0(:);    % starting parameters
aopt.Q       = Q;        % precision matrix: e = Q*(ey*ey')
aopt.history = [];       % error history when y=e & arg min y = f(x)
aopt.memory  = gradmemory;% incorporate previous gradients when recomputing
aopt.fixedstepderiv  = fsd;% fixed or adjusted step for derivative calculation
aopt.ObjectiveMethod = objective; % 'sse' 'fe' 'mse' 'rmse' (def sse)
aopt.hyperparameters = hyperparams;
aopt.forcels         = force_ls;  % force line search
aopt.smoothfun       = smoothfun; % auto smooth the error function
aopt.mimo            = ismimo;    % derivatives w.r.t multiple output fun
aopt.parallel        = doparallel;

BayesAdjust = ba; % Bayes-esque adjustment (constraint) of the GD-predicted parameters
                  % (converges slower but might be more careful)
IncMomentum = im; % Observe and use momentum data            

params.aopt = aopt;

% % if no prior guess for parameters step sizes (variances) find step sizes
% % whereby each parameter has similar effect size w.r.t. error
% if nargin < 3 || isempty(V)
%     V = FindOptimumStep(x0,V); % this func iterates: v = v./abs(J)
% end

% parameter and step vectors
x0  = full(x0(:));
V   = full(V(:));
v   = V;
pC  = diag(V);

% variance (in reduced space)
%--------------------------------------------------------------------------
if length(find(diag(pC))) == length(pC)
    Vv = sparse(eye(length(pC)));
else
    Vv    = spm_svd(pC);
end

V     = eye(length(x0));    %turn off svd 
pC    = V'*pC*V;
ipC   = spm_inv(spm_cat(spm_diag({pC})));
red   = diag(pC);

aopt.updateh = true; % update hyperpriors
aopt.pC  = red;      % store for derivative & objective function access
aopt.ipC = ipC;      % store ^

params.aopt = aopt;

% initial objective value
[e0]       = obj(x0,params);
n          = 0;
iterate    = true;
Vb         = V;
    
% initial error plot(s)
%--------------------------------------------------------------------------
if doplot
    setfig(); makeplot(x0,x0,params);
end

% initialise counters
%--------------------------------------------------------------------------
n_reject_consec = 0;
search          = 0;

% Initial probability threshold for inclusion (i.e. update) of a parameter
Initial_JPDtol = 1e-10;
JPDtol = Initial_JPDtol;

% parameters (in reduced space)
%--------------------------------------------------------------------------
np    = size(V,2); 
p     = [V'*x0];
ip    = (1:np)';
Ep    = V*p(ip);

dff          = []; % tracks changes in error over iterations
localminflag = 0;  % triggers when stuck in local minima

% print options before we start printing updates
if BayesAdjust; fprintf('Using BayesAdjust option\n'); end
if IncMomentum; fprintf('Using Momentum option\n');    end
fprintf('Using step-method: %d\n',step_method);
fprintf('Using Jaco (gradient) option: %d\n',order);
fprintf('User fun has %d varying parameters\n',length(find(red)));

% print start point - to console or logbook (loc)
refdate(loc);pupdate(loc,n,0,e0,e0,'start:');

% gen & store smoothing vec
if smoothfun
    smoothvec = round(linspace(6,1,maxit).^2);
end

if step_method == 0
    % step method can switch between 1 (big) and 3 (small) automatcically
     autostep = 1;
else autostep = 0;
end

% start optimisation loop
%==========================================================================
while iterate
    
    % counter
    %----------------------------------------------------------------------
    n = n + 1;    tic;
   
    pupdate(loc,n,0,e0,e0,'grdnts',toc);
    
    aopt.pp = x0;
    
    %  compute & store the smoothing function for this iteration
    if smoothfun
        aopt.R = @(x) spm_unvec(smooth(spm_vec(x),smoothvec(n)),x);
    end
    
    % compute gradients & search directions
    %----------------------------------------------------------------------
    aopt.updatej = true;
    aopt.updateh = true;
    params.aopt  = aopt;
    [e0,df0]  = obj( V*x0(ip),params );
    [e0,~,er] = obj( V*x0(ip),params );
    
    % Second order partial derivates w.r.t parameters, x[i], using:
    %
    % f0 = f(x[i]+h) 
    % fx = f(x[i]  )
    % f1 = f(x[i]-h) 
    %
    % j(i,:) =        (f0 - f1) / 2h  
    %          ----------------------------
    %            (f0 - 2 * fx + f1) / h^2  
    %
         
    %[df0,e0] = spm_diff(@obj,x0,1);
         
    % print end of gradient computation (just so we know it's finished)
    pupdate(loc,n,0,e0,e0,'g-done',toc); 
    
    % initial search direction (steepest) and slope
    %----------------------------------------------------------------------  
    if autostep
        if n < 3; 
            search_method = 1;
        else
           % auto switch the step size (method) based on mean negative error growth rate
           if (e0 ./ Hist.e(n-1)) > .9*mean([Hist.e e0] ./ [Hist.e(2:end) e0 e0])
                search_method = 3; 
           else search_method = 1;
           end
        end
        
    else
        search_method = step_method;
    end
    
    % compute step, a, in scheme: dx = x0 + a*-J
    [a,J] = compute_step(df0,red,e0,search_method); % a = (1/64) / J = -df0;
               
    % Log start of iteration (these are returned)
    Hist.e(n) = e0;
    Hist.p{n} = x0;
    Hist.J{n} = df0;
    
    % Track the variance of the partial derivatives over iterations
    if n >= 3
        switchoff = find( var(squeeze(cat(3,Hist.J{:}))')*Vv < 1e-12 );
        if any(switchoff)
            offvec = ones(1,size(Vv,2));
            offvec(switchoff) = 0;
            red = red.*(Vv*offvec');
            aopt.ipC = spm_inv(spm_diag(red));
        end
    end
    
    % make copies of error and param set for inner while loops
    x1  = x0;
    e1  = e0;
        
    % start counters
    improve = true;
    nfun    = 0;
    
    % iterative descent on this (gradient) trajectory
    %======================================================================
    while improve
        
        % descend while de < e1
        nfun = nfun + 1;
                                        
        % Compute The Parameter Step (from gradients and step sizes):
        % % x[p,t+1] = x[p,t] + a[p]*-dfdx[p] 
        %------------------------------------------------------------------
        dx = compute_dx(x1,a,J,red,search_method);  % dx = x1 + ( a * J );  
           
        % Optionally now compute a gradient descent on the step size, a:
        % -----------------------------------------------------------------
        % x[p,t+1] = x[p,t] +         a[p]          *-dfdx[p]
        % 
        % x[p,t+1] = x[p,t] + ( a[p] + b*-dfda[p] ) *-dfdx[p]
        %
        % where b (the step of the step) is fixed at 1e-4
        %
        % Note that optimising 'a' isn't just about speeding up the ascent,
        % because a is also the variance term on the parameters
        % distributions, this is also optimising the variance...
        OptimiseStepSize = BTLineSearch;
                
        if OptimiseStepSize && nfun == 1 && obj(dx,params) < obj(x1,params) && search_method ~= 4
            
            pupdate(loc,n,0,e1,e1,'grdstp',toc); 
            aopt.updateh = false;
            params.aopt  = aopt;
            br  = red;
            
            % Optimise a!
            % - Do a gradient descent on the step size!        
            afun = @(red) obj( compute_dx(x1,compute_step(df0,red,e0,search_method),...
                                    J,red,search_method),params );

            % Compute the gradient w.r.t steps sizes
            %[agrad] = jaco(afun,red,(~~red)*1e-4,0,2);
            [agrad] = jaco(afun,red,red/8,0,1);
            
            % condition it
            agrad(isinf(agrad))=0;
            
            % Backtracking Line Search:
            %    f(p + g(a)*-y) <= f(p + a*-y)
            
            chn = 0; loop = true;
            while loop
            
                chn   = chn + 1;
                d_red = red + ( (1e-3)*-agrad );

                % Update only stable new step sizes
                bad = unique([ find(isnan(d_red)); ...
                               find(isinf(d_red)); ...
                               find(d_red < 0) ]);
                d_red(bad) = red(bad);
                red        = d_red;

                % Final ddx
                [da,~]  = compute_step(df0,red,e0,search_method);
                ddx     = compute_dx(x1,da,J,red,search_method);
                
                if obj(ddx,params) < obj(dx,params)
                    if chn == 1
                        pupdate(loc,n,0,obj(ddx,params),e1,'BkLnSr',toc);
                    end
                    dx = ddx;
                    a  = da;
                    br = red;   
                    
                    % Update the (inverse) variance used in computing F
                    aopt.ipC   = spm_inv(spm_cat(spm_diag({diag(red)})));
                else
                    red  = br;
                    loop = false;
                end
                if chn == 20
                    loop = 0;
                end
            end
            
            aopt.updateh = true;
            
        end
        
        % The following options are essentially 'E-steps' in an Expectation
        % Maximisation routine - i.e. they estimate the missing variables
        % (parameter indices & values) that should be optimised in this iteration
        %==================================================================
        
        % Computes the probabilities of each (predicted) new parameter
        % coming from the same distribution defined by the prior (last best)
        % - We're going to use this as the prior probability of each param step   
        pdt  = zeros(1,length(x1));
        pt   = pdt;
        ured = diag(aopt.Cp); % latest estimate of v
        for i = 1:length(x1)
            if red(i)
                vv     = real(sqrt( ured(i) ));
                if vv <= 0 || isnan(vv) || isinf(vv); vv = 1/64; end
                
                pd(i)  = makedist('normal','mu', real(x1(i)),'sigma', vv);
                pdt(i) = (1-cdf( pd(i), real(x1(i))));
                pt(i)  = (1-cdf( pd(i), real(dx(i))));                                
            end
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
        

        % Given (gradient) predictions, dx[i..n], optimise obj(dx) 
        % Either by:
        % (1) just update all parameters
        % (2) update all parameters whose updated value improves obj
        % (3) update only parameters whose probability exceeds a threshold
        %------------------------------------------------------------------
        if obj(dx,params) < obj(x1,params) && ~BayesAdjust && ~aopt.forcels
            
            % Don't perform checks, assume all f(dx[i]) <= e1
            % i.e. full gradient prediction over parameters is good and we
            % don't want to be explicitly Bayesian about it ...
            gp  = ones(1,length(x0));
            gpi = 1:length(x0);
            de  = obj(V*dx,params);
            DFE = ones(1,length(x0))*de; 

        else
            % Assess each new parameter estimate (step) individually
            for nip = 1:length(dx)
                XX     = V*x0;
                if red(nip)
                    XX(nip)  = dx(nip);
                    DFE(nip) = obj(XX,params); % FE
                else
                    DFE(nip) = e0;
                end
            end
            
            DFE  = real(DFE(:));
           
            % Identify improver-parameters
            gp  = double(DFE < e0); % e0
            gpi = find(gp);
            
            if ~BayesAdjust
                % If the full gradient prediction over parameters did not
                % improve, but the BayesAdjust option is not selected, then
                % only update parameters showing improvements in objective func
                
                ddx        = V*x0;
                ddx(gpi)   = dx(gpi);
                dx         = ddx;
                de         = obj(dx,params);
                                
            else
                % If the BayesAdjust option is selected, perform a sort of 
                % Maximum likelihood estimation: find optimum set of 
                % parameters' steps that maximise param joint probabilities^
                pupdate(loc,n,0,e1,e1,'bayes ',toc);
                
                % Now we have (prior) probabilities of updated params 
                % and 'likelihoods', compute probabilities of (updated) params
                PP = pt(:) .* rescale(-DFE);
                
                [~,o]  = sort(PP(:),'descend');
                selpar = o( find(cumprod(PP(o)) > JPDtol) );
                              
                newp         = x1;
                newp(selpar) = dx(selpar);
                dx           = newp(:);                
                de           = obj(V*dx,params);
                probplot(PP,JPDtol);
                
            end            
        end
                
        
        % Tolerance on update error as function of iteration number
        % - this can be helpful in functions with lots of local minima
        % i.e. bad step required before improvement
        %etol = e1 * ( ( 0.5./n ) ./(nfun.^2) );
        etol = 0; % none
        
        if de  < ( obj(x1,params) + etol )
            
            % If the objective function has improved...
            if nfun == 1; pupdate(loc,n,0,de,e1,'improv',toc); end
            
            % update the error & the (reduced) parameter set
            %--------------------------------------------------------------
            df  = e1 - de;
            e1  = de;
            x1  = V'*dx;
            dff = [dff df];
        else
            % If it hasn't improved, flag to stop this loop...
            improve = false;            
        end
        
        % upper limit on the length of this loop (force recompute dfdx)
        if nfun >= inner_loop
            improve = false;
        end
        
    end  % end while improve... ends iter descent on this trajectory
      
    
    % ignore complex parameter values - for most functions, yes
    %----------------------------------------------------------------------
    x1 = (x1); % (put 'real' here)
        
    % evaluate - accept/reject - plot - adjust rate
    %======================================================================
    if e1 < e0 && ~aopt.forcels   % Improvement...
        
        % compute deltas & accept new parameters and error
        %------------------------------------------------------------------
        df =  e0 - e1;
        dp =  x0 - x1;
        x0 = -dp + x0;
        e0 =  e1;
                
        % Extrapolate...
        %==================================================================        
        
        % we know what param-step caused what improvement, so try again...
        %------------------------------------------------------------------
        exploit = true;
        nexpl   = 0;
        pupdate(loc,n,nexpl,e1,e0,'descnd',toc);
                
        while exploit
            % local linear extrapolation
            %extrapx = V*(x1+(-dp./-df));
            extrapx = V*(x1 + ((-dp).*((df-e1)./e1)).^(1+nexpl) );
            if obj(extrapx,params) < e1
                %dp    = dp + (extrapx-x1);
                x1    = extrapx(:);
                e1    = obj(x1,params);
                %df    = df + (e0-e1);
                nexpl = nexpl + 1;
            else
                % if this didn't work, just stop and move on to accepting
                % the best set from the while loop above
                exploit = false;
                pupdate(loc,n,nexpl,e1,e0,'finish',toc);
            end
            
            % upper limit on the length of this loop: no don't do this
            if nexpl == (inner_loop)
                exploit = false;
            end
        end
        
%         % after best full steps, try some smaller steps using selection of
%         % best performing parameters
%         [these,Ip] = sort(dp,'descend');
%         Ip = Ip(1:length(find(these)));
%         
%         parsel = true;
%         while parsel
%             
%             for jp = 1:length(Ip)
%                 x2         = x1;
%                 x2(Ip(jp)) = x1(Ip(jp)) - dp(Ip(jp));
%                 
%                 if obj(x2) < obj(x1)
%                     x1 = x2;
%                     e1 = obj(x1);
%                 else
%                     pfail(jp)=1;
%                 end
%             end
%             if sum(pfail)>=length(Ip)
%                 parsel=false;
%             end
%         end
        
        % Update best-so-far estimates
        e0 = e1;
        x0 = x1;
            
        % print & plots success
        %------------------------------------------------------------------
        pupdate(loc,n,nfun,e1,e0,'accept',toc);      % print update
        if doplot; makeplot(V*x0(ip),aopt.pp,params); end   % update plots
        
        n_reject_consec = 0;              % monitors consec rejections
        dff             = [dff df];       % monitors change in f over itrs
        JPDtol          = Initial_JPDtol; % resets prob threshold for update

    else
        
        % *if didn't improve: perform much more selective parameter update
        %==================================================================
        pupdate(loc,n,nfun,e1,e0,'parsel',toc);             
        
        % select only parameters whose steps improved the objective
        %------------------------------------------------------------------
        thisgood = gp*0; % track whether any of selection get used
        if any(gp)
            
            % sort good params by (improvement amount) * (probability)
            %--------------------------------------------------------------
            % update p's causing biggest improvment in fe while maintaining highest P(p)
            %[~,PO] = sort(pt(gpi).*rescale(-DFE(gpi)),'descend');
            [~,PO] = sort(rescale(-DFE(gpi)),'descend');
            
            %[~,PO] = sort([SpXm(:).*SKLD(:)],'descend');
            %gpi = 1:length(x0);
            
            % loop the good params in selected (PO) order
            %--------------------------------------------------------------
            improve1 = 1;
            nimp = 0 ;
            while improve1
                thisgood = gp*0; % tracks which params are updated below
                nimp     = nimp + 1;
                % evaluate the 'good' parameters
                for i  = 1:length(gpi)
                    %xnew             = real(V*x0);
                    xnew             = V*x0;
                    xnew(gpi(PO(i))) = dx(gpi(PO(i)));
                    enew             = obj(xnew,params);
                    % accept new error and parameters and continue
                    if enew < e0 && nimp < round(inner_loop)
                        dff = [dff (e0-enew)];
                        x0  = V'*(xnew);
                        e0  = enew;
                        thisgood(gpi(PO(i))) = 1;
                    end
                end
                
                % Ok, now assess whether any parameters were accepted from
                % this selective search....
                if any(thisgood)

                    % print & plot update
                    pupdate(loc,n,nfun,e0,e0,'accept',toc);
                    if doplot; makeplot(V*x0,x1,params); end

                    % update step size for these params
                    %red = red(:) + ( red(:).*thisgood(:) );      
                   
                    % reset rejection counter
                    n_reject_consec = 0;
                    
                    % reset JPDtol
                    JPDtol = Initial_JPDtol;

                else
                    % If we made it to here, then neither the full gradient
                    % predicted update, nore the selective search, managed
                    % to improve the objective. So:
                    
                    % reduce param steps (vars) and go back to main loop
                    red = red*.8;
                    %red = red * 1.2;
                    
                    % halt this while loop
                    improve1 = 0;
                    
                    % keep counting rejections
                    n_reject_consec = n_reject_consec + 1;
                    
                end
                
                % update global store of V
                aopt.pC = V*red;
            end
        else
            % Don't panic; it's ok & sometimes necessary to end up here 
            
            % If we get here, then there were not gradient steps across any
            % parameters which improved the objective! So...
            pupdate(loc,n,nfun,e0,e0,'reject',toc);
            
            % reduce step and go back to main loop
            %red = red*.8;
            red = red*1.4; % our (prior) 'variances' are probably too small
            
            % update global store of V
            aopt.pC = V*red;
            % keep counting rejections
            n_reject_consec = n_reject_consec + 1;
            
            % loosen inclusion threshold on param probability (temporarily)
            % i.e. specified variances aren't making sense.
            % (applies to Bayes option only)
            if BayesAdjust
                JPDtol = JPDtol * 1e-10;
                pupdate(loc,n,nfun,e0,e0,'TolAdj',toc);
            end
        end
                            
    end
    
    % stopping criteria, rules etc.
    %======================================================================
    if min_df == -1
        ldf   = inf;
        dftol = inf;
    elseif min_df ~= 0
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
        fprintf(loc,'I think we''re stuck...stopping.\n');
        
        % return current best
        X = V*(x0(ip));
        F = e0;
                
        % covariance estimation
        Cp = aopt.Cp;
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
        fprintf(loc,'Reached maximum iterations: stopping.\n');
        
        % return current best
        X = V*(x0(ip));
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
        X = V*(x0(ip));
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
            X = V*(x0(ip));
            F = e0;

            % covariance estimation
            Cp = aopt.Cp;
            PP = BayesInf(x0,Ep,diag(red));

            if writelog;fclose(loc);end
            return;
     end
end
    
end

% Subfunctions: Plotting & printing updates to the console / log...
%==========================================================================

function refdate(loc)
fprintf(loc,'\n');

fprintf(loc,'| ITERATION     | FUN EVAL | CURRENT F         | BEST F SO FAR      | ACTION | TIME\n');
fprintf(loc,'|---------------|----------|-------------------|--------------------|--------|-------------\n');

end

function s = prinfo(loc,it,nfun,nc,ncs)

s = sprintf(loc,'| Main It: %04i | nf: %04i | Selecting components: %01i of %01i\n',it,nfun,nc,ncs);
fprintf(loc,'| Main It: %04i | nf: %04i | Selecting components: %01i of %01i\n',it,nfun,nc,ncs);

end

function s = pupdate(loc,it,nfun,err,best,action,varargin)
persistent tx
if nargin >= 7
    n = varargin{1};
    fprintf(loc,'| Main It: %04i | nf: %04i | Err: %04i | Best: %04i | %s | %d\n',it,nfun,err,best,action,n);
    st=sprintf('\n| PROGRESS SUMMARY:\n|-----------------------------\n| Main Iteration: %04i \n| Num FunEval: %04i \n| Curr F: %04i \n| Best F: %04i \n| Status: %s \n| Time: %d\n',it,nfun,err,best,action,n);

else
    fprintf(loc,'| Main It: %04i | nf: %04i | Err: %04i | Best: %04i | %s |\n',it,nfun,err,best,action);
    st = sprintf('\n| PROGRESS SUMMARY:\n|-----------------------------\n| Main It: %04i \n| Num FunEval: %04i \n| Curr F: %04i \n| Best F: %04i \n| Status: %s \n\n',it,nfun,err,best,action);
end

s = subplot(4,3,3);
delete(tx);
tx = text(0,.5,st,'FontSize',18,'Color','w');
ax = gca;
set(ax,'visible','off');
ax.XGrid = 'off';
ax.YGrid = 'on';
s.YColor = [1 1 1];
s.XColor = [1 1 1];
s.Color  = [.3 .3 .3];
drawnow;


end

function setfig()

%figure('Name','AO','Color',[.3 .3 .3],'InvertHardcopy','off','position',[1088 122 442 914]);
figure('Name','AO','Color',[.3 .3 .3],'InvertHardcopy','off','position',[1116 140 1310 809]);
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

function probplot(growth,thresh)

growth(isnan(growth))=0;
these = find(growth>thresh);
s = subplot(414);
bar(growth,'FaceColor',[1 .7 .7],'EdgeColor','w');
%plot(growth,'w','linewidth',3);hold on
%plot(these,growth(these),'linewidth',3,'Color',[1 .7 .7]);
title('Parameter Step Probability','color','w','fontsize',18);
ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'on';
s.YColor = [1 1 1];
s.XColor = [1 1 1];
s.Color  = [.3 .3 .3];
drawnow;hold off;

end

function makeplot(x,ox,params)
% plot the function output (f(x)) on top of the thing we're ditting (Y)
%
%
global aopt

[Y,y] = GetStates(x,params);

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
    %s(1) = subplot(411);
    s(1) = subplot(4,3,[1 2]);
    
    plot(spm_vec(Y),'w:','linewidth',3); hold on;
    plot(spm_vec(y),     'linewidth',3,'Color',[1 .7 .7]); hold off;
    grid on;grid minor;title('AO System Identification: Current Best','color','w','fontsize',18);
    s(1).YColor = [1 1 1];
    s(1).XColor = [1 1 1];
    s(1).Color  = [.3 .3 .3];

    %s(2) = subplot(412);
    s(2) = subplot(4,3,[4]);
    %bar([former_error new_error]);
    plot(former_error,'w--','linewidth',3); hold on;
    plot(new_error,'linewidth',3,'Color',[1 .7 .7]); hold off;
    grid on;grid minor;title('Error Change','color','w','fontsize',18);
    s(2).YColor = [1 1 1];
    s(2).XColor = [1 1 1];
    s(2).Color  = [.3 .3 .3];
    
    
    %s(3) = subplot(413);
    s(3) = subplot(4,3,5);
    bar(real([ x(:)-ox(:) ]),'FaceColor',[1 .7 .7],'EdgeColor','w');
    title('Parameter Change','color','w','fontsize',18);
    ax = gca;
    ax.XGrid = 'off';
    ax.YGrid = 'on';
    s(3).YColor = [1 1 1];
    s(3).XColor = [1 1 1];
    s(3).Color  = [.3 .3 .3];
    drawnow;
    
    s(4) = subplot(4,3,[7 8]);
    plot(spm_vec(Y),'w:','linewidth',3);
    hold on;
    plot(spm_vec(Y)-aopt.oerror,'linewidth',3,'Color',[1 .7 .7]); hold off;
    grid on;grid minor;
    title('Last Best','color','w','fontsize',18);
    ax = gca;
    ax.XGrid = 'off';
    ax.YGrid = 'on';
    s(4).YColor = [1 1 1];
    s(4).XColor = [1 1 1];
    s(4).Color  = [.3 .3 .3];
    drawnow;
    
    
%end
end

aopt.oerror = new_error;

end

% Subfunctions: Fetching states & the objective function
%==========================================================================

function [Y,y] = GetStates(x,params)
% - evaluates the model and returns it along with the stored data Y
%

%global aopt
aopt = params.aopt;

IS = aopt.fun;
P  = x(:)';

try    y  = IS(P); 
catch; y  = spm_vec(aopt.y)*0;
end
Y  = aopt.y;

end

function [e,J,er,mp,Cp,L,params] = obj(x0,params)
% - compute the objective function - i.e. the sqaured error to minimise
% - also returns the parameter Jacobian,  error (vector), model prediction
% (vector) and covariance
%

if ~params.aopt.parallel
    global aopt;
else
    aopt=params.aopt;
end

% if ~isfield(aopt,'computeiCp')
%     % Compute inverse covariance - on first call trigger this, but it gets 
%     % switched off during objective calls during derivative calculation
%     aopt.computeiCp = 1;
% end

method = aopt.ObjectiveMethod;

IS = aopt.fun;
P  = x0(:)';

% Evalulate f(X)
warning off
try    y  = IS(P); 
catch; y  = spm_vec(aopt.y)*0+inf;
end
warning on;

% Data & precision
Y  = aopt.y;
Q  = aopt.Q;

% Apply smoothing operator if supplied
if isfield(aopt,'smoothfun') && isfield(aopt,'R') && aopt.smoothfun
    y = aopt.R(y);
    Y = aopt.R(Y);
end

% Check / complete the derivative matrix (for the covariance)
%--------------------------------------------------------------------------
if ~isfield(aopt,'J')
    aopt.J = ones(length(x0),length(spm_vec(y)));
end
if isfield(aopt,'J') && isvector(aopt.J) && length(x0) > 1
    aopt.J = repmat(aopt.J,[1 length(spm_vec(y))]);
end

% Free Energy Objective Function: F(p) = log evidence - divergence
%--------------------------------------------------------------------------
Q  = spm_Ce(1*ones(1,length(spm_vec(y)))); %

if aopt.mimo
    Q  = {AGenQ(spm_vec(Y))};
end

if ~isfield(aopt,'h');
    h  = sparse(length(Q),1) - log(var(spm_vec(Y))) + 4;
else
    h = aopt.h;
end

% If the system output >1D (e.g. a power spectum), prioritise shifts along x
% using an auto-generated smooth precision operator
%----------------------------------------------------------------------------
% if aopt.mimo
%     ny    = length(spm_vec(y));
%     [~,l] = findpeaks(spm_vec(Y),'NPeaks',12);
%     FS    = ones(ny,1)./ny;
%     %FS    = linspace(1,4,length(y))';
%     %FS    = FS .* rescale(( (1 - cos(2*pi*[1:ny]'/(ny + 1)))/2 ),.5,1);
%     FS(l) = 4;
%     FS    = AGenQ(FS);
%     Q     = {FS};
% end
% 
% for i = 1:ny
%     Q{i} = sparse(i,i,FS(i,i),ny,ny);
% end

%Q  = {AGenQ(atcm.fun.aenvelope(spm_vec(Y),30))};

if isinf(h)
    h = 1/8;
end

iS = sparse(0);

for i  = 1:length(Q)
    iS = iS + Q{i}*(exp(-32) + exp(h(i)));
end

ny  = length(spm_vec(y));
nq  = ny ./ length(Q);
e   = spm_vec(Y) - spm_vec(y);

ipC = aopt.ipC;

warning off;                                % suppress singularity warnings
Cp  = spm_inv( (aopt.J*iS*aopt.J') + ipC );
%Cp  = spm_inv( ipC );
warning on

p   = ( x0(:) - aopt.pp(:) );

if any(isnan(Cp(:))) 
    Cp = Cp;
end

if aopt.hyperparameters
    % pulled directly from SPM's spm_nlsi_GN.m ...
    
    % ascent on h / precision {M-step}
    %==========================================================================
    clear P;
    nh  = length(Q);
    S   = spm_inv(iS);
    ihC = speye(nh,nh)*exp(4);
    hE  = sparse(nh,1) - log(var(spm_vec(Y))) + 4;
    for i = 1:nh
        P{i}   = Q{i}*exp(h(i));
        PS{i}  = P{i}*S;
        P{i}   = kron(speye(nq),P{i});
        JPJ{i} = real(aopt.J*P{i}*aopt.J');
    end

    % derivatives: dLdh 
    %------------------------------------------------------------------
    for i = 1:nh
        dFdh(i,1)      =   trace(PS{i})*nq/2 ...
            - real(e'*P{i}*e)/2 ...
            - spm_trace(Cp,JPJ{i})/2;
        for j = i:nh
            dFdhh(i,j) = - spm_trace(PS{i},PS{j})*nq/2;
            dFdhh(j,i) =   dFdhh(i,j);
        end
    end

    % add hyperpriors
    %------------------------------------------------------------------
    d     = h     - hE;
    dFdh  = dFdh  - ihC*d;
    dFdhh = dFdhh - ihC;
    Ch    = spm_inv(-dFdhh);

    % update ReML estimate
    %------------------------------------------------------------------
    warning off;
    dh    = spm_dx(dFdhh,dFdh,{4});
    dh    = min(max(dh,-1),1);
    warning on;
    h     = h  + dh;

    if aopt.updateh
        aopt.h = h;
    end
end % end of if hyperparams (from spm) ... 

%L(1) = spm_logdet(iS)*nq/2  - sum( sum(real(e'.*iS.*e)/2) ) - ny*log(8*atan(1))/2;
L(1) = spm_logdet(iS)*nq/2  - real(e'*iS*e)/2 - ny*log(8*atan(1))/2;            ...
L(2) = spm_logdet(ipC*Cp)/2 - p'*ipC*p/2;

if aopt.hyperparameters
    L(3) = spm_logdet(ihC*Ch)/2 - d'*ihC*d/2; % no hyperparameters
end

try aopt.Cp = aopt.Cp + Cp;
catch
    aopt.Cp = Cp;
end
    aopt.iS = iS;
              
       
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
er = ( spm_vec(y) - spm_vec(Y) ).^2;
mp = spm_vec(y);

%params.aopt = aopt;

% this wraps obj to return only the third output for MIMOs
% when we want the derivatives w.r.t. each output 
function er  = inter(x0,params)
    [~,~,~,er] = obj(x0,params);
end

J = [];

% this hands off to jaco, which computes the derivatives
% - note this can be used differently for multi-output systems
if nargout == 2 || nargout == 7
    V    = aopt.pC;
    Ord  = aopt.order; 
    
    if aopt.fixedstepderiv == 1
        V = (~~V)*1e-3;
       %V = (~~V)*exp(-8);
    else
        %V = V*1e-2;
    end
    
    %aopt.computeiCp = 0; % don't re-invert covariance for each p of dfdp
    aopt.updateh = 0;
    
    if ~aopt.mimo 
        if ~aopt.parallel
            [J,ip] = jaco(@obj,x0,V,0,Ord,[],{params});   ... df[e]   /dx [MISO]
        else
            [J,ip] = jacopar(@obj,x0,V,0,Ord,{params});   ... df[e]   /dx [MISO] 
        end
    else
        nout   = 4;
        [J,ip] = jaco_mimo(@obj,x0,V,0,Ord,nout);
        J0     = cat(2,J{:,1})';
        J      = cat(2,J{:,nout})';
    end
    
    aopt.updateh = 1;
    
    % store for objective function
    if  aopt.updatej
        aopt.J       = J;
        aopt.updatej = false;     % (when triggered always switch off)
    end
    
    % accumulate gradients / memory of gradients
    if aopt.memory
        try
            %J       = J + (aopt.pJ/2) ;
            J       = ( J + aopt.pJ ) ./2;
            aopt.pJ = J;
        catch
            aopt.pJ = J;
        end
    end
    
    if aopt.mimo
        J = spm_vec(J0);
    end
    
    params.aopt = aopt;
    
    %[J,ip] = jaco(IS,P',V,0,Ord);   ... df/dx
    %J = repmat(V,[1 size(J,2)])./J;
end


end

function J = compute_step_J(df0,red,e0,step_method)
% wrapper on the function below to just return the second output!
    [x3,J] = compute_step(df0,red,e0,step_method);
end

function [x3,J] = compute_step(df0,red,e0,step_method)
% Given the gradients (df0) & parameter variances (red)
% compute the step, 'a' , in:
%
%  dx = x + a*-J
%
% in most cases this is some linear transform of red.
%
global aopt

search_method = step_method;

switch search_method
    case 1
        
        J      = -df0';
        dFdpp  = -(J'*J);
        
        %Initial step (red is expected reduction in obj fun)
        %x3  = red./(1-dFdpp);
        
        % Compatibility with older matlabs
        x3  = repmat(red,[1 length(red)])./(1-dFdpp);
        
        %Leading (gradient) components
        [uu,ss,vv] = spm_svd(x3);
        nc = min(find(cumsum(diag(full(ss)))./sum(diag(ss))>=.95));
        x3 = full(uu(:,1:nc)*ss(1:nc,1:nc)*vv(:,1:nc)');
        
    case 2
        
        J     = -df0';
        dFdp  = -real(aopt.J*aopt.iS*repmat(e0,[ length(aopt.iS),1]) ) - aopt.ipC*aopt.pp;
        dFdpp = -real(aopt.J*aopt.iS*aopt.J')  - aopt.ipC;
        
        % bit hacky but means i don't have to change the function
        x3 = dFdp;
        J  = dFdpp;

    case 3
        
        J      = -df0;
        dFdpp  = -(J'*J);
        
        %Initial step (Rasmussen method)
        x3  = (4*red)./(1-dFdpp);
        
    case 4
        
        J  = -df0;
        %x3 = (1/64);
        x3 = red(:);
        
end

end

function dx = compute_dx(x1,a,J,red,search_method)
%  given start point x1, step a and (directional) gradient J, compute dx:
%
% dx = x1 + a*J
%
% (note gradient has already been negative signed)
global aopt

if search_method == 1
    dx    = x1 + (a*J');                 % When a is a matrix
elseif search_method == 2
    dFdp  = a;
    dFdpp = J;
    ddx   = spm_dx(dFdpp,dFdp,{red})';    % The SPM way
    ddx   = ddx(:);
    dx    = x1 + ddx;    
elseif search_method == 3                % Rasmussen w/ varying p steps
    dx    = x1 + (a.*J);                 
elseif search_method == 4                % Flat/generic descent
    dx    = x1 + (a.*J);
end


end

function y = rescale(x)
% for older matlabs where rescale function isn't a built in
y =  (x - min(x) ) / ( max(x) - min(x) );

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
% effective / equally balanced w.r.t F
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

function X = DefOpts()
% Returns an empty options structure
X.step_method = 3;
X.im          = 1;
X.ba          = 0;
X.objective   = 'fe';
X.writelog    = 0;
X.order       = 2;
X.min_df      = 0;
X.criterion   = 1e-3;
X.Q           = [];
X.inner_loop  = 9999;
X.maxit       = 128;
X.y           = 0;
X.V           = [];
X.x0          = [];
X.fun         = [];

X.hyperparams  = 0;
X.BTLineSearch = 1;
X.force_ls     = 0;
X.doplot       = 1;
X.smoothfun    = 0;
X.ismimo       = 0;
X.gradmemory   = 0;
X.doparallel   = 0;
X.fsd          = 1;
end

function parseinputstruct(opts)
% Gets the user supplied options structure and assigns the options to the
% correct variables in the AO base workspace

fprintf('User supplied options / config structure...\n');

def = DefOpts();
opt = fieldnames(def);

% complete the options 
for i = 1:length(opt)
    if isfield(opts,opt{i})
        def.(opt{i}) = opts.(opt{i});
    end
end

% send it back to the caller (AO.m)
for i = 1:length(opt)
    assignin('caller',opt{i},def.(opt{i}));
end

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
    '\nIf using the BayesAdjust option = 1, this selection routine entails MLE\n' ...
    'to find the optimum set of parameter steps that maximise their approx joint probabilities.\n'...
    '\nIn model fitting scenarios, the code is set up so that you pass the model\n'...
    'function (fun), parameters and also the data you want to fit. The advantage of\n'...
    'this is that the algo can compute the objective function. This is necessary\n'...
    'if you want to minimise free energy (but also has SSE, MSE, RMSE etc).\n' ...
    '\nOutputs are the posteriors (means), objective value (F), (co)variance (CP),\n' ...
    'posterior probabilities (Pp) and a History structure (Hist) that contains the\n'...
    'parameters, objective values and gradients from each iteration of the algorithm.\n' ...
    '\nThe code makes use of the fabulous SPM toolbox functions for things like\n' ...
    'vectorising and un-vectorising - so SPM is a dependency. This means that\n' ...
    'the data you''re fitting (y in AO(fun,p,v,y) ) and the output of fun(p)\n'...
    'can be of whatever data type you like (vector, matrix, cell etc).\n' ...
    '\nIf you want to speed up the algorithm, change search_method from 3 to 1.\n'...
    'This method extrapolates further and can fit data quite a bit faster \n'...
    '(and with better fits). However, it is also prone to pushing parameters\n'...
    'to extremes, which is often bad in model fitting when you plan to make\n'...
    'some parameter inference.\n']);



end

% Other experiment stuff
%---------------------------------------------



        % (option) Probability constraint
        %------------------------------------------------------------------
%         if BayesAdjust
%             % This probably isn't sensible, use the JPD method instead.
%             % Probabilities of these (predicted) values actually belonging to
%             % the prior distribution as a bound on parameter step
%             % (with arbitrary threshold)
%             ddx = dx - x1;
%             ppx = spm_Ncdf(abs(x0),abs(dx),sqrt(red)); ppx(ppx<.2) = 0.2;
%                     
%             % Parameter update
%             dx = x1 + (ddx.*ppx);
%             
%             % mock some distributions to visualise changes
%             for i = 1:length(x1)
%                 pd(i)   = makedist('normal','mu',abs(x1(i)),'sigma', ( red(i) ));
%                 pr(i,:) = pdf(pd(i),abs(x1(i))-10:abs(x1(i))+10);
%                 po(i,:) = pdf(pd(i),abs(dx(i))-10:abs(dx(i))+10);
%             end
%             BayesPlot(-10:10,pr,po);
%             
%         end

        % (option) Divergence adjustment
        %------------------------------------------------------------------
%         if DivAdjust
%             % This probably isn't a good idea actually
%             % Divergence of the prob distribution 
%             PQ  = pt(:).*log( pt(:)./pdt(:) );  PQ(isnan(PQ)) = 0;
%             iPQ = 1./(1 - PQ);
% 
%             % Parameter update
%             ddx = dx(:) - x1(:);
%             dx  = x1(:) + ddx.*iPQ(:);
%         end

        % (option) Param selection using maximum joint-probability estimates
        %------------------------------------------------------------------
%         JPD = BayesAdjust;
%         if JPD
%             % Compute JPs from rank sorted p(P)s            
%             [~,o]  = sort(pt(:),'descend');
%             for ns = 1:length(pt)
%                 pjpd(ns) = prod( pt(o(1:ns)) ) * ns;    
%             end
%             
%             % Compute 'growth rate' of (prob ranked) parameter inclusions
%             growth(1) = 1;
%             for i = 2:length(pjpd)
%                 growth(i) = pjpd(i) ./ growth(i-1);
%             end
%             
%             % JPDtol is a hyperparameter             
%             % Starting with the highest probability parameter estimate
%             % (resulting from the GD), it asks, what is the effect of
%             % also updating the parameter with the next highest probability
%             % ... and so on until we hit a small threshold param, JPDtol.
%             selpar       = o( find( growth(:) > JPDtol ));            
%             newp         = x1;
%             newp(selpar) = dx(selpar);
%             dx           = newp(:);
%             
%             probplot(pjpd,JPDtol);
%                                     
%         end
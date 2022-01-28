# aoptim

AO is an optimisation algorithm for parameter estimation of nonlinear dynamical and state-space models. 

It performs a 2nd order gradient (curvature) descent on a free-energy objective function using a sort of EM framework that incorporates MLE, line search, momentum acceleration and other options that implicitly try to minimise the error while maximising evidence and parameter probabilities. 

It is inspired by theoretical models of neuronal computation - i.e. the way the brain optimises its own 'parameters' to minimise its objective. In this sense, when applied to relevant problems, it is an AI.


```
% A gradient/curvature descent optimisation routine, designed primarily 
% for nonlinear model fitting / system identification & parameter estimation. 
% 
% The objective function minimises the free energy (~ ELBO - evidence lower bound).
%
% Y0 = f(x) + e  
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
%   x[p,t+1] = x[p,t] + a[p] *-dFdx[p]
%
% Note the use of different step sizes, a[p], for each parameter.
% Optionally, low dimensional hyperparameter tuning can be used to find 
% the best step size a[p], by setting step_method = 6;
%
%   x[p,t+1] = x[p,t] + (b*V) *-dFdx[p]  ... where b is optimised using fminsearch
%
% where V maps between a (reduced) parameter subspace and the full vector.
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
%  dFdpp  = approx -(J'*J); -- however, we normalise out the full magnitude 
%                              of the gradient using a LDL decomposition,
%                              such that;
%  dFdpp  = -L*(D./sum(D))*L' 
%  a      = (V)./(1-dFdpp);   
% 
% If step_method = 3, dFdpp is a small number and this reduces to a simple
% scale on the stepsize - i.e. a = V./scale. However, if step_method = 1, J
% is transposed, such that dFdpp is np*np. This leads to much bigger
% parameter steps & can be useful when your start positions are a long way
% from the solution. In either case, note that the variance term V[p] on the
% parameter distribution is a scaled version of the step size. 
% 
% Alternatively, step_method=6 invokes hyperparameter tuning of the step
% size.
%
% By default momentum is included (opts.im=1). The idea is that we can 
% have more confidence in parameters that are repeatedly updated in the 
% same direction, so we can take bigger steps for those parameters as the
% optimisation progresses.
%
% For each iteration of the ascent:
% 
%   dx[p]  = x[p] + a[p]*-dFdx
%
% With the MLE setting, the probability of each predicted dx[p] coming 
% from x[p] is computed (Pp) and incorporated into a NL-WLS implementation of 
% MLE:
%           
%   b(s+1) = b(s) - (J'*Pp*J)^-1*J'*Pp*r(s)
%   dx     = x - (a*b)
%  
% The {free energy} objective function minimised is
%==========================================================================
%  L(1) = spm_logdet(iS)*nq/2  - real(e'*iS*e)/2 - ny*log(8*atan(1))/2; % complexity minus accuracy of states
%  L(2) = spm_logdet(ipC*Cp)/2 - p'*ipC*p/2;                            % complexity minus accuracy of parameters
%  L(3) = spm_logdet(ihC*Ch)/2 - d'*ihC*d/2;                            % complexity minus accuracy of precision (hyperparameter)
%  F    = -sum(L);
%
% *Alternatively, set opts.objective to 'rmse' 'sse' 'euclidean' ... 
%
% INPUTS:
%-------------------------------------------------------------------------
% To call this function using an options structure:
%-------------------------------------------------------------------------
% opts = AO('options');     % get the options struct
% opts.fun = @myfun         % fill in what you want...
% opts.x0  = [0 1];         % start param value
% opts.V   = [1 1]/8;       % variances for each param
% opts.y   = [...];         % data to fit - .e.g y = f(p) + e
% opts.maxit       = 125;   % max num iterations 
% opts.step_method = 1;     % step method - 1 (big), 3 (small) and 4 (vanilla)**.
% opts.im    = 1;           % flag to include momentum of parameters
% opts.order = 2;           % partial derivate fun option (see jaco.m): 1 or 2
% opts.hyperparameters = 0; % flag to do a grad ascent on the precision (3rd term in FE)
% opts.criterion    = -300  % convergence threshold
% opts.mleselect    = 0;    % maximum likelihood param selection
% opts.objective    = 'fe'; % objective fun: 'free energy', 'logevidence','sse', ...
% opts.writelog     = 0;    % flag to write logbook instead of to console
% opts.Q   = [];            % square matrix precision operator ( size=length(y) )
% opts.inner_loop = 10;     % limit on number of repeats on same decent (between grad comp)
% opts.DoMLE = 0;           % flag to perform a sort of MLE via WLS
% opts.force_ls = 0;        % flag to force a line search every time
% opts.ismimo = 0;          % compute dFdx of a multi-output function, not just objective value
% opts.gradmemory = 0;      % remember previous gradients
% opts.doparallel = 0;      % compute gradients using parfor
% opts.fsd = 1;             % use fixed step for derivative computation
% opts.allow_worsen = 0;    % allow objective to get worse sometimes
% opts.doimagesc = 0;       % if data/model outputs are matrix (not vector), plot as such
% opts.EnforcePriorProb = 0;% force the parameter updates to strictly adhere to prior distribution
% opts.FS = []              % feat selection function: FS(y)
% opts.userplotfun = [];    % inject a user plot function into the main display
% opts.corrweight = 1;      % weight error term by correlation
%
% [X,F,Cp,PP,Hist] = AO(opts);       % call the optimser, passing the options struct
%
% OUTPUTS:
%-------------------------------------------------------------------------
% X   = posterior parameters
% F   = fit value (depending on objective function specified)
% CP  = parameter covariance
% Pp  = posterior probabilites
% H   = history
%
% *If the optimiser isn't working well, try making V smaller!
%
% References
%-------------------------------------------------------------------------
%
% "Computing the objective function in DCM" Stephan, Friston & Penny
% https://www.fil.ion.ucl.ac.uk/spm/doc/papers/stephan_DCM_ObjFcn_tr05.pdf
%
% "The free energy principal: a rough guide to the brain?" Friston
% https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20-%20a%20rough%20guide%20to%20the%20brain.pdf
%
% "Likelihood and Bayesian Inference And Computation" Gelman & Hill 
% http://www.stat.columbia.edu/~gelman/arm/chap18.pdf
%
% For the nonlinear least squares MLE / Gauss Newton:
% https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
%
% For an explanation of momentum in gradient methods:
% https://distill.pub/2017/momentum/
%
% Approximation of derivaives by finite difference methods:
% https://www.ljll.math.upmc.fr/frey/cours/UdC/ma691/ma691_ch6.pdf
%
% For an explanation of normalised gradients in gradient descent
% https://jermwatt.github.io/machine_learning_refined/notes/3_First_order_methods/3_9_Normalized.html
%
% AS2019/2020/2021
% alexandershaw4@gmail.com

```

Here's a video of the the optimiser solving a system of nonlinear differential equations that describe a mean-field neural mass model - fitting it's spectral output to some real data:

![screenshot](OptimisationGIF.gif)

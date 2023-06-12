# aoptim

AO is an optimisation algorithm for parameter estimation of nonlinear dynamical and state-space models. 

It performs a 2nd order gradient (curvature) descent on a free-energy objective function using a sort of EM framework that incorporates MLE, line search, momentum acceleration and other options that implicitly try to minimise the error while maximising evidence and parameter probabilities. 

It is inspired by theoretical models of neuronal computation - i.e. the way the brain optimises its own 'parameters' to minimise its objective. In this sense, when applied to relevant problems, it is an AI.

![overview](AO_optim_overview.png)

```
Install:
addpath(genpath('~/Downloads/aoptim/'))
```

```
% Getting started with default options:

op = AO('options')

op.fun = func;          % function/model f(x0)
op.x0  = x0(:);      % start values: x0
op.y   = Y(:);       % data we're fitting (for computation of objective fun, e.g. e = Y - f(x)
op.V   = V(:);       % variance / step for each parameter, e.g. ones(length(x0),1)/8

op.objective='gauss'; % select smooth Gaussian error function

% run it:
[X,F,CV,~,Hi] = AO(op); 


% change objective to 'gaussmap' for MAP estimation
```



from AO.m Help:

```
% A (Bayesian) gradient (curvature) descent optimisation routine, designed primarily 
% for parameter estimation in nonlinear models.
%
% The algorithm combines a Newton-like gradient routine (& optionally
% MAP and ML) with line search and an optimisation on the composition of update 
% parameters from a combination of the gradient flow and memories of previous updates.
% It further implements an exponential cost function hyperparameter which
% is tuned independently, and an iteratively updating precision operator.
% 
% General idea: I have some data Y0 and a model (f) with some parameters x.
% Y0 = f(x) + e  
%
% For a multivariate function f(x) where x = [p1 .. pn]' a gradient descent scheme
% is:
%
%   x[p,t+1] = x[p,t] + a[p] *-dFdx[p]
%
% Note the use of different step sizes, a[p], for each parameter.
% Optionally, low dimensional hyperparameter tuning can be used to find 
% the best step size a[p], by setting step_method = 6;
%
%   x[p,t+1] = x[p,t] + (b*V) *-dFdx[p]  ... where b is optimised independently
%
% See jaco.m for options, although by default these are computed using a 
% finite difference approximation of the curvature, which retains the sign of the gradient:
%
% f0 = F(x[p]+h) 
% fx = F(x[p]  )
% f1 = F(x[p]-h) 
%
% j(p,:) =        (f0 - f1) / 2h  
%          ----------------------------
%            (f0 - 2 * fx + f1) / h^2  
%
% The algorithm computes the objective function itself based on user
% option; to retreive an empty options structure, do:
%
% opts = AO('options')
%
% Compulsory arguments are: (full list of optionals below)
%
% opts.y     = data to fit
% opts.fun   = model or function f(x)  
% opts.x0    = parameter vector (initial guesses) for x in f(x)
% opts.V     = Var vector, with initial variance for each elemtn of x
% opts.objective = the objective function selected from:
% {'sse' 'mse' 'rmse' 'mvgkl' 'gauss' 'gaussmap' 'gaussq' 'jsd' 'euclidean' 'gkld'}
%
% then to run the optmisation, pass the opts structure back into AO with
% these outputs:
%
%  [X,F,Cp,PP,Hist,params] = AO(opts)
%
% Optional "step" methods (def 9: normal fixed step GD):
% -- step_method = 1 invokes steepest descent
% -- step_method = 3 or 4 invokes a vanilla dx = x + a*-J descent
% -- step_method = 6 invokes hyperparameter tuning of the step size.
% -- step_method = 7 invokes an eigen decomp of the Jacobian matrix
% -- step_method = 8 converts the routine to a mirror descent with
%    Bregman proximity term.
%
% By default momentum is included (opts.im=1). The idea is that we can 
% have more confidence in parameters that are repeatedly updated in the 
% same direction, so we can take bigger steps for those parameters as the
% optimisation progresses.
%
% NOTE - the default option [2022] is now to use a regularised Newton routine:
%---------------------------------------------------------------------------
% 
%  dx = x + inv(H*L*H)*-J
%
% whewre H is the Hessian and J the jacobian (dFdx). L is a regularisation
% term that is optimised independently. 
%
% The (best and default) objective function is you are unsure, is 'gauss'
% which is simply a smooth (approx Gaussian) error function, or 'gaussq'
% which is similar to gauss but implements a sort of pca. 
%
% If you want true MAP estimates (or just to be Bayesian), use 'gaussmap'
% which implements a MAP routine: 
%
%  log(f(X|p)) + log(g(p))
%
%
% Other important stuff to know:
% -------------------------------
% if your function f(x) generates a vector output (not a single value),
% then you can compute the partial gradients along each oputput, which is
% necessary for proper implementation of some functions e.g. GaussNewton;
% flag:
%
% opts.ismimo = 1;
%
% The gradient computation can be done in parallel if you have a cluster or
% multicore computer, set:
%
% opts.doparallel = 1;
%
% Set opts.hypertune = 1 to append an exponential cost function to the chosen 
% objective function. This is defined as:
% 
% c = t * exp(1/t * data - pred)
% 
% where t is a (temperature) hyperparameter controlled through a separate gradient
% descent routine.
%
% ALSO SET:
%
% opts.memory_optimise = 1; to optimise the weighting of dx on the gradient flow and recent memories 
% opts.opts.rungekutta = 1; to invoke a runge-kutta optimisation locally around the gradient predicted dx
% opts.updateQ = 1; to update the error weighting on the precision matrix 
%  
% The {fe / free energy} objective function minimised is
%==========================================================================
%  L(1) = spm_logdet(iS)*nq/2  - real(e'*iS*e)/2 - ny*log(8*atan(1))/2; % complexity minus accuracy of states
%  L(2) = spm_logdet(ipC*Cp)/2 - p'*ipC*p/2;                            % complexity minus accuracy of parameters
%  L(3) = spm_logdet(ihC*Ch)/2 - d'*ihC*d/2;                            % complexity minus accuracy of precision (hyperparameter)
%  F    = -sum(L);
%
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
% opts.WeightByProbability = 0; % weight parameter step update by prob
% opts.faster = 0;          % make faster by limiting loops
% opts.factorise_gradients = 1; % factorise gradients
% opts.sample_mvn = 0;      % when  stuck sample from a multivariate gauss
% opts.do_gpr = 0;          % try to refine estimates using a Gauss process
% opts.normalise_gradients=0;% normalise gradients to a pdf
% opts.isGaussNewton = 0;   % explicitly make the update a Gauss-Newton
% opts.do_poly=0;           % same as do_gpr but using polynomials
% opts.steps_choice = [];   % use multiple step methods and pick best on-line
% opts.hypertune  = 1;      % tune a hyperparameter using exponential cost
% opts.memory_optimise = 1; % switch on memory (recommend)
% opts.rungekutta = 1;      % RK line search (recommend)
% opts.updateQ = 1;         % update precision on each iteration(recommend)
% opts.DoMAP = 0;           % maximum a posteriori
% opts.DoMAP_Bayes = 0;     % Bayesian MAP (recommend, make sure mimo=1)
% opts.crit = [0 0 0 0 0 0 0 0];
% opts.save_constant = 0;   % save .mat on each iteration
% opts.nocheck = 0;         % skip some checks to make faster
% opts.isQR = 0;            % use QR decomp to solve for dx 
% opts.NatGrad = 0;         % work in natural gradients
% opts.variance_estimation = 0; % try to estimate variances
% opts.gradtol = 1e-4;      % minimum tol on dFdx
% opts.isGaussNewtonReg=1;  % try to use DampedGaussNewton steps
% opts.orthogradient=1;     % orthogonalise jacobian/partial gradients
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


# aoptim

AO implements a gradient descent optimisation that incorporates 
curvature information (like a GaussNewton). Each parameter of f() is 
treated as a Gaussian distribution with variance v. Step sizes are controlled 
by the variance term and calculated using standard method. 
Additional constraint options can be included (e.g. Divergence based). 
When the full gradient prediction doesnt improve the objective, the routine
picks a subset of parameters that do. This selection is based on the probability
of the (GD predicted) new parameter value coming from the prior distribution.

If using the BayesAdjust option = 1, this selection routine entails MLE
to find the optimum set of parameter steps that maximise their approx joint probabilities.

In model fitting scenarios, the code is set up so that you pass the model
function (fun), parameters and also the data you want to fit. The advantage of
this is that the algo can compute the objective function. This is necessary
if you want to minimise free energy (but also has SSE, MSE, RMSE etc).

Outputs are the posteriors (means), objective value (F), (co)variance (CP),
posterior probabilities (Pp) and a History structure (Hist) that contains the
parameters, objective values and gradients from each iteration of the algorithm.

The code makes use of the fabulous SPM toolbox functions for things like
vectorising and un-vectorising - so SPM is a dependency. This means that
the data you're fitting (y in AO(fun,p,v,y) ) and the output of fun(p)
can be of whatever data type you like (vector, matrix, cell, struct etc).

If you want to speed up the algorithm, change search_method from 3 to 1.
This method extrapolates further and can fit data quite a bit faster 
(and with better fits). However, it is also prone to pushing parameters
to extremes, which is often bad in model fitting when you plan to make
some parameter inference.

```
A curvature descent based optimisation routine for system identification.
Designed for highly parameterised non-linear MIMO/MISO dynamical models.

*Also includes wrapper code for optimsing Dynamic Causal Models (DCMs) - see AO_DCM.m*

Fit differentiable models of the forms:

(1.)    Y0 = f(x0) + e   // model fitting, where truth, Y0, is known 
(2.)    e  = f(x0)       // generic function minimisation 


Generic Usages:

Model fitting:

[X,F,Cp,Pp,Hist] = AO(fun,x0,V,[y],[maxit],[inner_loop],[Q],[crit],[min_df],[ordr],[log],[obj],[ba],[im],[step_meth])

Generic function minimisation:

[X,F,Cp,Pp,Hist] = AO(fun,x0,V,[0],[maxit],[inner_loop],[Q],[crit],[min_df],[ordr],[log],[obj],[ba],[im],[step_meth])


fun = function handle to system of equations
x0  = parameter start points - i.e. fun(x0). Can be a vector if fun is a system.
V   = width of the distirbution for each element of x0 (& also step size). If unsure, try 1/8ths.
y   = the real data to fit, when model fitting. If fun returns the error to be minimised, set to 0.

maxit = (optional) max number of iterations, default 128.
inner_loop = (optional) max num iterations descending using same strategy.
Q = (optional) output precision matrix for MIMO systems, i.e. SSE = Q*(ey*ey')*Q';
crit = (optional) convergence crierion, default 1e-2.
min_df = (optional) minimum change in function value to continue, default 0.
order = (optional) order of derivatives (see jaco.m), default 2 (curvature).
writelog = (optional) write steps to log (==1) instead of console (==0, def)
objective = (optional) error function: 'sse' 'mse' 'rmse' or 'fe' (free energy) 
ba = (optional) make an explicitly Bayesian adjustment to the parameter steps
im = (optional) consider momentum of parameters over iterations
step_meth = (optional) change the default step method: def=3, or try 1 for bigger steps.

Notes:

How well it works can be highly dependent on the step size / variances (V).
In my testing minimising SSE seems to produce the best overall fits but free energy (log evidence - divergence) converges much faster, so maybe start with that.

Here's a video of the the optimiser solving a system of nonlinear differential equations that describe a mean-field neural mass model - fitting it's spectral output to some real data:
```
![screenshot](OptimisationGIF.gif)

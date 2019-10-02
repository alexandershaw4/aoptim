% Example script for using AO optimiser.
%
% Here we'll fit a model of the form Y = f(x) + e.
% The optimisation will entail minimising 'e' by find the right 'x'.
%
% In this example, the 'model' is an exponential decay curve of t.^-2.
%
% AS

% Generate a function: here fun = t.^-x
%--------------------------------------------------------------------------
fun = @(x) (1:20).^-x;

% Generate a ground truth, where the exponent is 2, e..g t.^-2
%--------------------------------------------------------------------------
Y = fun(2);

% Run the optimiser with an itial guess of x = 1:
%--------------------------------------------------------------------------
[X,F] = AO(fun,1,1/32,Y,[],[],[],[],1e-16);

% Plot the outputs
%--------------------------------------------------------------------------
close;figure;t = 1:20;
plot(t,fun(2),'k:',t,fun(1),'c:*',t,fun(X),'m--*','linewidth',3);
title('Fitting Y = f(x) + e   ( where Y = x.^-2 )');
legend({'Data to fit: Y' 'Start point: f(x0)' 'Fitted point: f(X)'});
set(findall(gcf,'-property','FontSize'),'FontSize',20);
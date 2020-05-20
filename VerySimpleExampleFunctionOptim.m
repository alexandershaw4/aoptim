% TOY example function: the squarer...

y = [1 2 3 4 5 6 7 8]'/8;

x = [2 3 2 4 5 6 4 2]'/8;

f = @(x) x.^2;

opt = AO('options');
opt.fun  = f;
opt.x0 = x;
opt.V = (~~x)/8;
opt.step_method = 1;
opt.y = f(y);

[X,F] = AO(opt);

% Compare solution with optimiser prediction
[X(:) sqrt(y')]
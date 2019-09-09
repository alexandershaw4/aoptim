function e = ackley_fun(x)
% Ackley function
% f(x1, x2) = f(3, 0.5) = 0.

x1 = x(:,1);
x2 = x(:,2);

e = 20*(1 - exp(-0.2*sqrt(0.5*(x1.^2 + x2.^2))))...
            - exp(0.5*(cos(2*pi*x1) + cos(2*pi*x2))) + exp(1);

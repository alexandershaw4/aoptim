function y = rk4delays(fx,t,x,p,d,input)
% An implementatino of the 4th order Runge-Kutta integration algorithm but
% with the addition of state delays.
%
% Given an ode: dx/dt = f(x,[inp],p) 
%                       where x = states, p = parameters, inp = [opt] input vector
%
% and a vector of delays, d, integrates for period in t (and dt = diff(t))
% with interpolated delays.
%
% y = rk4delays(f,t,x,p,d,[input])
%
%
% Example system with and without delays
%----------------------------------------
% a = 1; b = 2; c = a*(b^2-1/(b^2));
% f  = @(x)[x(2)*x(3)-a*x(1),x(1)*(x(3)-c)-a*x(2),1-x(1)*x(2)];
% dt = 1/600;
% t  = (dt:dt:100);
% x  = [1 0 1];
% integrate
% y  = rk4delays(f,t,[1 0 1]',[],[0 0 0]);
% y1 = rk4delays(f,t,[1 0 1]',[],[4 4 4]/100);
% figure;
% scatter3(y(1,:),y(2,:),y(3,:));hold on; 
% scatter3(y1(1,:),y1(2,:),y1(3,:));
%
% AS2023

v = x;

if nargin < 6 || isempty(input)
    input = t*0;
    if ~isempty(p)
        f = @(x,input,p) fx(x,p);
    else
        f = @(x,input,p) fx(x);
    end
else
    f = fx;
end

dt = mean(diff(t));

for i = 1:length(t)
    if i == 1
        % 4-th order Runge-Kutta metho - first pass
        %--------------------------------------------------
        k1 = f(v          ,input(i),p);
        k2 = f(v+0.5*dt*k1,input(i),p);
        k3 = f(v-0.5*dt*k2,input(i),p);
        k4 = f(v+    dt*k3,input(i),p);
    
        dxdt      = (dt/6)*(k1+2*k2+2*k3+k4);
        v         = v + dxdt;
        y(:,i)    = v ;

    else
    
        % 4-th order Runge-Kutta method.
        %--------------------------------------------------
        k1 = f(v          ,input(i),p);
        k2 = f(v+0.5*dt*k1,input(i)+dt/2,p);
        k3 = f(v+0.5*dt*k2,input(i)+dt/2,p);
        [k4] = f(v+    dt*(k3),input(i),p);
    
        dxdt = (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
        v         = v + dxdt;
    
        % State Delays - interpolated
        %--------------------------------------------------
        L = (d);
    
        for j = 1:length(L)
            ti = real(L(j))/dt;
            if i > 1 && any(ti)
                pt = t(i) - ti;
                if pt > 0
                    v(j) = interp1(t(1:i), [y(j,1:i-1) v(j)]', pt);
                end
            end
        end
    
        % Full update
        %--------------------------------------------------
        y(:,i) =   v;
    
    end
end
function [X,F] = AO_basin_hop(funopts,T)
% a basin-hopping wrapper on AO optimisation


% initial optimisation
[z0,Fz0] = AO(funopts)


Tmin = 1e-8;    % Temperature minimum 
done = false;
maxloop = 100;
loop = 0;
while ~done
    
    xstep = funopts.x0 + 0.01*rand(1);
    
    funopts.x0 = xstep;
    
    % optimise
    [zstar,Fzstar] = AO(funopts)

    % accept?
    
    p = exp(-(Fzstar-Fz0)/T);   % acceptance probability
    if rand(1) <= p
        x0 = xstep;
        z0 = zstar;
        
        funopts.x0 = x0;
    end
    T = T*((0.95)^loop); % temperature update
    
    if T <= Tmin
        done = true; % stopping criteria
        xmin = z0;
    end
    
    loop = loop + 1;
    if loop >= maxloop
        fprintf('Loop: %i\n',loop);
        error('Loop exceeded maxloop');
    end
end
end


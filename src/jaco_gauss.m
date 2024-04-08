function J = jaco_gauss(f,x)

d   = exp(-8);
x0  = f(x);
J   = zeros(length(x),length(x0));
nx0 = length(x0);

parfor i = 1:length(x)
    
    dx    = x;
    lx    = x;
    ux    = x;

    dx(i) = dx(i) + d*dx(i);
    lx(i) = lx(i) - d*lx(i) * 2.45;
    ux(i) = ux(i) + d*ux(i) * 2.45;
    
    x2 = f(dx);
    x1 = f(lx);
    x3 = f(ux);

    %J(i,:) = x2 + (x1./2.45) - (x3./2.45);

    all = [x1 x2 x3];
    for j = 1:nx0
        pd = fitdist(all(j,:)','Normal');
        J(i,j) = pd.mu;
    end

end
    



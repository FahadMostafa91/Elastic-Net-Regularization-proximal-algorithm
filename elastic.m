
clear;
m=60;n=40;
A=randn(m,n);
b=randn(m,1);
x0=randn(n,1);
x=x0;
Q=A'*A;
L=eigs(Q,1);
tol=10^(-4);
k=1;
kmax=1000;
lambda=0.3;
alpha=0.01;
epsilon=10^(-5);
t=1/L;
diff=2*tol;


%Elastic Net Regularization proximal algorithm
x=x0;
cost_current=0.5*(norm(A*x-b)^2+(1-alpha)*lambda*norm(x)^2)+alpha*lambda*norm(x,1);
diff=2*tol;
k=1;
while diff>tol && k<kmax
    y=x-t*((A'*A+(1-alpha)*lambda*eye(n))*x-A'*b);
    xold=x;
    x=proximal_gradient_function_ridge(y,t*lambda*alpha);
    cost_old=cost_current;
    cost_current=0.5*(norm(A*x-b)^2+(1-alpha)*lambda*norm(x)^2)+alpha*lambda*norm(x,1);
    k=k+1;
    diff=cost_old-cost_current;
end;
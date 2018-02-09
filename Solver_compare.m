% Proximal gradient descent algorithm
% This solver is to solve the Huberized support vector machine problem of 
% min_{w,b} 1/N \Sigma_{i=1}^N h_{\delta}(y_i(b+x_i^T {\bf w}))+\lambda1 || w ||_1 + \lambda2/2 || w ||^2

% Input datasets: X, y, where y \in {-1,1}
% parameters:
% N: the number of iterations;
% delta: the smoothing paramters > 0;
% lambda1 and lambda2: regularization coefficient positive;
% beta: is to search the proper step size when gradient Lip-continuous is unknown, \beta is larger than 1/2 and generally can be 1
% alpha: The inital step size and will be adjusted automatically by \beta to ensure
% the descent of objective function;
% sigma1 and sigma2: the coefficents of increasing (decreasing) step size and they both are > 1
clc; close all; clear all;
% paramters
N=500;
delta=0.1; lambda1=0.01; lambda2=0.1;
beta=1; alpha0=0.1; 
sigma1=1 ;sigma2=2.8;
tau=1;
F_opt= 2.17;
% load datasets
load('gisette.mat', 'ytrain');
load('gisette.mat', 'Xtrain');
% load('E:\04_course\21_Topics in Optimization\Homework1\realsim.mat'); 
X=full(Xtrain);
Y=full(ytrain);
[samples, dim]=size(X);
% Initialization
winit=abs(randn(dim,1));
binit=0;
alpha=alpha0;
w=winit;
b=binit;
time=zeros(1,N);
time_acc=zeros(1,N);
F_old=0;   
his_obj_prox=zeros(N,1);
his_obj_acc_prox=zeros(N,1);
tstart_prox = tic;
for k=1:N 
    w_old = w;
    % update w
    % determin the set I1 and set I2  
    t=(X*w+repmat(b,samples,1)).*Y;
    I1=find(t>(1-delta) & t<=1);
    I2=find(t<=(1-delta));
    t1=(X(I1,:)*w+repmat(b,numel(I1),1)-Y(I1))/(samples*delta); 
    t2= -Y(I2)/samples;   
    Grad_w = sum(repmat(t1,[1,dim]).*X(I1,:),1)+  sum(repmat(t2,[1,dim]).*X(I2,:),1)+ lambda2*w';
    p=w - alpha * Grad_w'; 
    w = sign(p) .* max(0, abs(p) - alpha * lambda1);  
    % update b 
    t=(X*w+repmat(b,samples,1)).*Y;
    I1=find(t>(1-delta) & t<=1);
    I2=find(t<=(1-delta));
    t1=(X(I1,:)*w+repmat(b,numel(I1),1)-Y(I1))/(samples*delta);
    t2=  -Y(I2)/samples;  
    Grad_b= sum(t1) + sum(t2);
    b = b - alpha *(Grad_b);
    % the updated objective value F_new   
    F_new =  0;
    for i1=1:numel(I1)
        ti=Y(I1(i1))* (b+X(I1(i1),:)*w);
        F_new=F_new + (1-ti)^2/(2*delta*samples) ;
    end
    for i2=1:numel(I2)
        ti=Y(I2(i2))* (b+X(I2(i2),:)*w);
        F_new=F_new + (1-ti-delta/2)/samples;
    end  
    F_new = F_new + lambda1* norm(w,1) + lambda2/2 * norm(w,2)^2;  
    if mod(k,100)==0
        fprintf('The objective is %f after %d steps \n', F_new,k);  
    end 
    % backing search the step size alpha 
    eta = F_new - F_old - Grad_w*(w - w_old) - norm(w - w_old)^2/(2*beta*alpha)  ;
    F_old=F_new; 
    his_obj_prox(k)=F_new;
    if eta > 0
        alpha = alpha/sigma2;
    else
        alpha = sigma1*alpha;
    end  
end
time_prox= toc(tstart_prox);      
% run accelerate proximal gradient descent
w = winit;
b= binit;
w_bar = w; 
alpha =alpha0;
tstart_acc=tic;
for k=1:N 
    % determin the set I1 and set I2  
    t=(X*w_bar+repmat(b,samples,1)).*Y;
    I1=find(t>(1-delta) & t<=1);
    I2=find(t<=(1-delta));
    t1=(X(I1,:)*w_bar+repmat(b,numel(I1),1)-Y(I1))/(samples*delta);
    t2= -Y(I2)/samples;   
    Grad_wbar = sum(repmat(t1,[1,dim]).*X(I1,:),1)+  sum(repmat(t2,[1,dim]).*X(I2,:),1)+ lambda2*w_bar';
    p=w_bar - alpha * Grad_wbar';   
    w_old = w ; 
    w = sign(p) .* max(0, abs(p) - alpha * lambda1);
    % update b 
    t=(X*w+repmat(b,samples,1)).*Y;
    I1=find(t>(1-delta) & t<=1);
    I2=find(t<=(1-delta));
    t1=(X(I1,:)*w+repmat(b,numel(I1),1)-Y(I1))/(samples*delta);
    t2=  -Y(I2)/samples;  
    Grad_w = sum(repmat(t1,[1,dim]).*X(I1,:),1)+  sum(repmat(t2,[1,dim]).*X(I2,:),1)+ lambda2*w'; 
    Grad_b= sum(t1) + sum(t2);
    b = b - alpha *(Grad_b);
    % the updated objective value F_new   
    F_new =  0;
    for i1=1:numel(I1)
        ti=Y(I1(i1))* (b+X(I1(i1),:)*w);
        F_new=F_new + (1-ti)^2/(2*delta*samples) ;
    end
    for i2=1:numel(I2)
        ti=Y(I2(i2))* (b+X(I2(i2),:)*w);
        F_new=F_new + (1-ti-delta/2)/samples;
    end  
    F_new = F_new + lambda1* norm(w,1) + lambda2/2 * norm(w,2)^2;  
    if mod(k,100)==0
        fprintf('The objective is %f after %d steps \n', F_new,k);  
    end
%     fprintf('The time for this iteration is %d \n',toc)
    % backing search the step size alpha 
    eta = F_new - F_old - Grad_w*(w - w_old) - norm(w - w_old)^2/(2*beta*alpha)  ;
    F_old=F_new;
    his_obj_acc_prox(k)=F_new;
    if eta > 0
        alpha = alpha/sigma2;
    else
        alpha = sigma1*alpha;
    end 
    % update w_bar
    tau_old=tau;
    tau=(1+sqrt(1+4*tau^2))/2;
    gamma=(tau_old-1)/tau;  
    % update w_bar
    w_bar = w + (w - w_old)*gamma;  
    % reduce the oscillation
    if (F_new -F_old) > 0
        w_bar = w ;
        tau = tau_old;
        his_obj_acc_prox(k)=F_old;
    end 
end
 time_acc= toc(tstart_acc);       


 
% plot the results
close all;
plot(his_obj_prox-F_opt, 'b-','linewidth',2);
hold on;
plot(his_obj_acc_prox-F_opt,'k-','linewidth',2);
legend('Proximal Gradient Descent','Accelerated Proximal Gradient Descent')
xlabel('number of iterations')
ylabel('Distance to the optimal')
xlim([0,500])
set (gca,'fontsize',14)   
    
fprintf('Running time of proximal gradient descent algorithm is %f \n',time_prox );
fprintf('Running time of accelerated proximal gradient descent algorithm is %f \n',time_acc );



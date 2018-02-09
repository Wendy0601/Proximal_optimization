% clc ; clear all; close all;
% parameters:
% w=;
% b=;
% load gisette.mat data
% load('gisette.mat', 'Xtest');
% load('gisette.mat', 'ytest'); 
% [test_size,dim]=size(Xtest);
% Y_pre=( Xtest*w+repmat(b,test_size,1))./(abs(Xtest*w+repmat(b,test_size,1)));
% correct = numel(find(Y_pre==ytest));
% accuracy = correct / numel(ytest) 

% load('realsim.mat', 'Xtest','ytest');
load('E:\04_course\21_Topics in Optimization\Homework1\gisette.mat', 'Xtrain','ytrain','Xtest','ytest');
% Xtest=full(Xtest);
% ytest=full(ytest);
X=Xtest;
y=ytest;
[test_size,dim]=size(X);
Y_pre=( X*w+repmat(b,test_size,1))./(abs(X*w+repmat(b,test_size,1)));
correct = numel(find(Y_pre==y));
accuracy = correct / numel(y) 
function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

ls=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C=0.01;
sigma=0.01;
bC=.01;%to store best C
bSigma=0.01;%to store best sigma
model=svmTrain(X,y,C,@(x1, x2) gaussianKernel(x1, x2, sigma));
m1=mean(double(svmPredict(model, Xval) ~= yval));
for i=1:8
    for j=1:8
        C = ls(i);
        sigma = ls(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        m2=mean(double(predictions ~= yval));
        if m1>m2
            m1=m2;
            bC=C;
            bSigma=sigma;
        end
    end
end
C=bC;
sigma=bSigma;

% =========================================================================

end

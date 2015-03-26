function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(X); % number of training examples
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%% Part 1/2: Cost, J(theta)
H = sigmoid(X * theta);
sum1 = 1/m * (-y'*log(H) - (1-y)'*log(1-H));
% should not regularize the parameter theta_0
sum2 = (lambda/(2*m)) * sum(theta(2:end).^2);
J = sum1 + sum2;

%% Part 2/2: gradient of cost function
grad_0 = (1/m) * X(:,1)' * (H - y); % j=0 (special definition)
grad_rest = (1/m) * X(:,2:end)' * (H - y) + (lambda/m) * theta(2:end); % j>0
grad = [grad_0; grad_rest];
% =============================================================

end

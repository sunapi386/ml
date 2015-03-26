function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
[m, n] = size(X); % number of training examples and features


% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% cost function  
% J(theta) = 1/m sum i=1..m [-y*log(h_theta(x_i))
%            -(1-y)log(1-h_theta(x_i))
J_costs = zeros(m, 1); % m elements
for i=1:m
    h_theta = sigmoid(X(i,:)*theta); % g(z) (sigmoid)
    J_costs(i) = (-y(i)*log(h_theta)) - (1-y(i))*log(1-h_theta); 
end
J = (1/m) * sum(J_costs);

% gradient of cost vector
for j=1:n
    % for each j-th term of theta
    sum_term = 0;
    for i=1:m
        z = X(i,:)*theta;
        h_theta = sigmoid(z); % g(z) (sigmoid)
        sum_term = sum_term + (h_theta - y(i)) * X(i,j);
    end
    grad(j) = (1/m) * sum_term;
end


% =============================================================

end

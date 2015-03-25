function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2); % number of features

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% Width of X for generality

cost = 0;
for row = 1:m
    hx = theta'*X(row,1:n)'; % hypothesis h(x)
    cost = cost + (hx - y(row))^ 2;
end

J = 1/(2*m) * cost;

% =========================================================================

end

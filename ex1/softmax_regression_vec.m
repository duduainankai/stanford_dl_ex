function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  a = X;
  b = y;
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  
  %{
  %	非常慢的循环做法  训练一次50分钟
  theta=[theta, zeros(n,1)];
  r = exp(theta' * X);   % (n*k)' * (n*m) = (k*m)
  sumr = sum(r);         % 1*m
  
  for i = 1:m
    f += -(([1:num_classes] == y(i)) * log(r(:,i)/sumr(i)));
  end

  for i = 1:m
    p = r(:,i)/sumr(i);    % k*1
    a = -(X(:,i) * (([1:num_classes] == y(i)) - p'));
    g += a(:,1:end-1);
  end
  %}

  %比较快的向量化做法  训练一次6分钟
  theta=[theta, zeros(n,1)];
  h = exp(theta' * X);  % k*m        
  p = bsxfun(@rdivide,h,sum(h));  % k*m

  c = log(p);          % k*m
  i = sub2ind(size(c), y,1:size(c,2));  % y的值代表第i个样本的label k  从矩阵c的第i列中取出第k个值  返回的是编号 1*m
  values = c(i);       % 1*m

  f = -sum(values);


  a = [1:num_classes]' == y;  % k*m
  k = -X*(a - p)';
  g = k(:,1:end-1);


  g=g(:); % make gradient a vector for minFunc


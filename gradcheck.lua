local gradcheck = {}


function gradcheck.relative_error(x, y, h)
  h = h or 1e-12
  if torch.isTensor(x) and torch.isTensor(y) then
    local top = torch.abs(x - y)
    local bottom = torch.cmax(torch.abs(x) + torch.abs(y), h)
    return torch.max(torch.cdiv(top, bottom))
  else
    return math.abs(x - y) / math.max(math.abs(x) + math.abs(y), h)
  end
end


function gradcheck.numeric_gradient(f, x, df, eps)
  df = df or 1.0
  eps = eps or 1e-8
  local n = x:nElement()
  local x_flat = x:view(n)
  local dx_num = x.new(#x):zero()
  local dx_num_flat = dx_num:view(n)
  for i = 1, n do
    local orig = x_flat[i]
    
    x_flat[i] = orig + eps
    local pos = f(x)
    if torch.isTensor(df) then
      pos = pos:clone()
    end
    
    x_flat[i] = orig - eps
    local neg = f(x)
    if torch.isTensor(df) then
      neg = neg:clone()
    end
    
    local d = nil
    if torch.isTensor(df) then
      d = torch.dot(pos - neg, df) / (2 * eps)
    else
      d = df * (pos - neg) / (2 * eps)
    end
    
    dx_num_flat[i] = d
    x_flat[i] = orig
  end
  return dx_num
end
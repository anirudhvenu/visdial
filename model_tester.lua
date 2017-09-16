require 'torch'
require 'model'
require 'nn'
require 'nngraph'
require 'rnn'

local gradcheck = require 'gradcheck'

local tests = {}
local tester = torch.Tester()

------------------------------------------------------------------------
-- Input arguments and options
------------------------------------------------------------------------
local opt = require 'opts';
print(opt)

-- seed for reproducibility
torch.manualSeed(1234);

------------------------------------------------------------------------
-- Setting model parameters
------------------------------------------------------------------------
-- transfer all options to model
local modelParams = opt;

-- path to save the model
local modelPath = opt.savePath

------------------------------------------------------------------------
-- Setup the model
------------------------------------------------------------------------
require 'model'
local model = Model(modelParams);


local function gradCheck(opt, modelParams, dataloader)
  local dtype = 'torch.DoubleTensor'
  local opt = opt
  -- opt.vocab_size = 5
  -- opt.input_encoding_size = 4
  -- opt.rnn_size = 8
  -- opt.num_layers = 2
  -- opt.dropout = 0
  -- opt.seq_length = 7
  -- opt.batch_size = 6
 
  local model = Model(modelParams);
  local crit = model.criterion
  -- local lm = nn.LanguageModel(opt)
  -- local crit = nn.LanguageModelCriterion()
  model:type(dtype)
  crit:type(dtype)

  -- local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
  -- seq[{ {4, 7}, 1 }] = 0
  -- seq[{ {5, 7}, 4 }] = 0
  -- local imgs = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)

  local batch = dataloader:getTrainBatch(self.params);
  


  -- evaluate the analytic gradient
  local output = model:forward(batch)
  local loss = crit:forward(output, batch)
  local gradOutput = crit:backward(output, batch)
  local gradInput, dummy = unpack(model:backward(batch, gradOutput))

  -- create a loss function wrapper
  local function f(x)
    local output = model:forward(x)
    local loss = crit:forward(output)
    return loss
  end

  local gradInput_num = gradcheck.numeric_gradient(f, batch, 1, 1e-6)

  -- print(gradInput)
  -- print(gradInput_num)
  -- local g = gradInput:view(-1)
  -- local gn = gradInput_num:view(-1)
  -- for i=1,g:nElement() do
  --   local r = gradcheck.relative_error(g[i],gn[i])
  --   print(i, g[i], gn[i], r)
  -- end

  tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
  tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 5e-4)
end



tests.gradCheck = gradCheck

tester:add(tests)
tester:run()
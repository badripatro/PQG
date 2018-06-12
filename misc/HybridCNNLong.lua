-- input=26,200,17533-- SEQ_LENGTH, ALPHABATSIZE, BATCHLENGTH
local HybridCNNLong = {}
function HybridCNNLong.cnn(alphasize, emb_dim, dropout, avg, cnn_dim)
  dropout = dropout or 0.0
  avg = avg or 0
  cnn_dim = cnn_dim or 512

  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- Bag

  local net = nn.Sequential()
  net:add(nn.TemporalConvolution(alphasize, 256, 1))--2x26x384  -- paramater =17533x 256x1=26931072
  net:add(nn.Threshold())
  net:add(nn.TemporalConvolution(256, 512, 1))--2x26x512  -- paramater =256x 512=26931072
  net:add(nn.Threshold())
 
  local h1 = nn.SplitTable(2)(net(inputs[1]))

  local r2 = FixedRNN.rnn(26, avg, cnn_dim)(h1)
  out = nn.Linear(cnn_dim, emb_dim)(nn.Dropout(dropout)(r2))
  local outputs = {}
  table.insert(outputs, out)
  return nn.gModule(inputs, outputs)
end

return HybridCNNLong

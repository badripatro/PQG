------------------------------------------------------------------------------------
--  Torch Implementation of "Learning Semantic Sentence Embeddings using Pair-wise Discriminator"
--  ./train.sh
------------------------------------------------------------------------------------
require 'nn'
require 'torch'
require 'rnn'
require 'loadcaffe'
require 'optim' 
require 'misc.LanguageModel'
require 'misc.optim_updates'

local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
FixedRNN = require('misc.FixedGRU')
DocumentCNN = require('misc.HybridCNNLong')
require 'xlua'
-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Visual Question Answering model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_ques_h5','data/quora_data_prepro.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data/quora_data_prepro.json','path to the json file containing additional info and vocab')

-- starting point
cmd:option('-start_from', 'pretrained/model_epoch12.t7', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
--cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-feature_type', 'VGG', 'VGG or Residual')

-- Model settings
cmd:option('-batch_size',150,'what is theutils batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-input_encoding_size',512,'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-att_size',512,'size of sttention vector which refer to k in paper')
cmd:option('-emb_size',512,'the size after embeeding from onehot')
cmd:option('-rnn_layers',1,'number of the rnn layer')

-- Optimization
cmd:option('-optim','rmsprop','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',0.0008,'learning rate')--0.0001,--0.0002,--0.005
cmd:option('-learning_rate_decay_start', 5, 'at what epoch to start decaying learning rate? (-1 = dont)')--learning_rate_decay_start', 100,
cmd:option('-learning_rate_decay_every', 5, 'every how many epoch thereafter to drop LR by half?')---learning_rate_decay_every', 1500,
cmd:option('-momentum',0.9,'momentum')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')--optim_alpha',0.99
cmd:option('-optim_beta',0.999,'beta used for adam')--optim_beta',0.995
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator in rmsprop')
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-iterPerEpoch', 1250)
cmd:option('-drop_prob_lm', 0.5, 'strength of drop_prob_lm in the Language Model RNN')

-- Evaluation/Checkpointing
cmd:text('===>Save/Load Options')
cmd:option('-save',               'Results', 'save directory')
cmd:option('-checkpoint_dir', 'Results/checkpoints', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-language_eval', 1, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-val_images_use', 24800, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 2500, 'how often to save a model checkpoint?')
cmd:option('-losses_log_every', 200, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '1', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 1234, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-nGPU', 3, 'Number of GPUs to use by default')

--text encoder
cmd:option('-txtSize',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-cnn_dim',512,'the encoding size of each token in the vocabulary, and the image.')

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
print(opt)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid+1) -- note +1 because lua is 1-indexed
end
opt = cmd:parse(arg)

---------------------------------------------------------------------
--Step 4: create directory and log file
------------------------------------------------------------------
------------------------- Output files configuration -----------------
os.execute('mkdir -p ' .. opt.save) -- to create result folder  save folder
cmd:log(opt.save .. '/Log_cmdline.txt', opt)  --save log file in save folder
-- os.execute('cp ' .. opt.network .. '.lua ' .. opt.save)  -- to copy network to the save file path

-- to save model parameter
os.execute('mkdir -p ' .. opt.checkpoint_dir) 

-- to save log
local err_log_filename = paths.concat(opt.save,'ErrorProgress')
local err_log = optim.Logger(err_log_filename)

-- to save log
local errT_log_filename = paths.concat(opt.save,'ErrorProgress')
local errT_log = optim.Logger(errT_log_filename)

-- to save log
local lang_stats_filename = paths.concat(opt.save,'language_statstics')
local lang_stats_log = optim.Logger(lang_stats_filename)

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
-- dataloader
local dataloader = dofile('misc/dataloader.lua')
dataloader:initialize(opt)
collectgarbage()
--------------------------------------------------------------
local function weights_init(m)
  local name = torch.type(m)
  if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
  elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
  end
end
------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
local protos = {}
local loaded_checkpoint
local lmOpt
-- intialize language model
if string.len(opt.start_from) > 0 then
  -- load protos from file
  print('initializing weights from ' .. opt.start_from)
  loaded_checkpoint = torch.load(opt.start_from)
  lmOpt= loaded_checkpoint.lmOpt
else

  -- intialize language model
  lmOpt = {}
  lmOpt.vocab_size = dataloader:getVocabSize()
  lmOpt.input_encoding_size = opt.input_encoding_size
  lmOpt.rnn_size = opt.rnn_size
  lmOpt.num_layers = 1
  lmOpt.drop_prob_lm = opt.drop_prob_lm
  lmOpt.seq_length = dataloader:getSeqLength()
  lmOpt.batch_size = opt.batch_size 
  lmOpt.emb_size= opt.input_encoding_size
  lmOpt.hidden_size = opt.input_encoding_size
  lmOpt.att_size = opt.att_size
  lmOpt.num_layers = opt.rnn_layers
end
--------------------------------------------------------------------------
-- Model Defination
------------------------------------------------------------------------
-- Design Model From scratch
  print('Building the model from scratch...')
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------
-- Encoding Part 
protos.netE = DocumentCNN.cnn(lmOpt.vocab_size+1, opt.txtSize, 0, 1, opt.cnn_dim)
protos.netE:apply(weights_init)

-- Decoding Part 
protos.netD = nn.LanguageModel(lmOpt)

--- Convert decoder ouput to size of input vector dim
local  decoder_convert_net = nn.Sequential()
decoder_convert_net:add(nn.Narrow(1, 2, lmOpt.seq_length))
decoder_convert_net:add(nn.Transpose({1,2}))

-- criterion for the language model
protos.crit = nn.LanguageModelCriterion()
---------------------------------------------------------------------------
--Clone network
netT=protos.netE:clone('weight','bias', 'gradWeight','gradBias')
print('total number of parameters in protos.netE embedding net: ', protos.netE)
print('total number of parameters in netT embedding net: ', netT)


---------------------------------------------------------------------------------------
--print('model',protos)
print('vocab_size',lmOpt.vocab_size)--4223
print('seq_length',lmOpt.seq_length)

--------------------------------------------------------------------------
-- Shifting to GPU
------------------------------------------------------------------------
print('ship everything to GPU...')
-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
  decoder_convert_net=decoder_convert_net:cuda()
  netT=netT:cuda()
end
---------------------------------------------------------------------------
-- Declear variable
local input_txt_emb1 = torch.CudaTensor(opt.batch_size, opt.txtSize)
local input_txt_emb2 = torch.CudaTensor(opt.batch_size, opt.txtSize)

--------------------------------------------------------------------------
-- Get parameter
------------------------------------------------------------------------
local eparams, grad_eparams = protos.netE:getParameters()
local lparams, grad_lparams = protos.netD:getParameters()

--------------------------------------------------------------------------
-- Init parameter
------------------------------------------------------------------------
eparams:uniform(-0.1, 0.1)
lparams:uniform(-0.1, 0.1) 

--------------------------------------------------------------------------
-- Pretrained Weights
-----------------------------------------------------------------------
if string.len(opt.start_from) > 0 then
  print('Load the weight...')
  eparams:copy(loaded_checkpoint.eparams)
  lparams:copy(loaded_checkpoint.lparams) 

end

print('total number of parameters in Question embedding net: ', eparams:nElement())
assert(eparams:nElement() == grad_eparams:nElement())

print('total number of parameters of language Generating model ', lparams:nElement())
assert(lparams:nElement() == grad_lparams:nElement())

collectgarbage() 



---------------------------------------------------------------------------
--This part of the code is refered from :https://github.com/reedscot/icml2016
function JointEmbeddingLoss(feature_emb1, feature_emb2)
  local batch_size = feature_emb1:size(1)
  local score = torch.zeros(batch_size, batch_size)
  local grads_text1 = feature_emb1:clone():fill(0)
  local grads_text2 = feature_emb2:clone():fill(0)

  local loss = 0
  acc_smooth = 0.0
  acc_batch = 0.0
  for i = 1,batch_size do
    for j = 1,batch_size do
      score[{i,j}] = torch.dot(feature_emb2:narrow(1,i,1), feature_emb1:narrow(1,j,1))
    end
    local label_score = score[{i,i}]
    for j = 1,batch_size do
      if (i ~= j) then
        local cur_score = score[{i,j}]
        local thresh = cur_score - label_score + 1
        if (thresh > 0) then
          loss = loss + thresh
          local txt_diff = feature_emb1:narrow(1,j,1) - feature_emb1:narrow(1,i,1)
          grads_text2:narrow(1, i, 1):add(txt_diff)
          grads_text1:narrow(1, j, 1):add(feature_emb2:narrow(1,i,1))
          grads_text1:narrow(1, i, 1):add(-feature_emb2:narrow(1,i,1))
        end
      end 
    end
    local max_score, max_ix = score:narrow(1,i,1):max(2)
    if (max_ix[{1,1}] == i) then
      acc_batch = acc_batch + 1
    end
  end
  acc_batch = 100 * (acc_batch / batch_size)
  local denom = batch_size * batch_size
  local res = { [1] = grads_text1:div(denom),
                [2] = grads_text2:div(denom) }
  acc_smooth = 0.99 * acc_smooth + 0.01 * acc_batch
  return loss / denom, res
end

---------------------------------------------------------------------------
-- This is onehot representation of 26 word token into onehot vector of vocabulary size
-- This is used to convert 200x26 to 200x26x4224
function one_hot_tensor(input,vocab)
  output=torch.Tensor(input:size()[1],input:size()[2],vocab)
  function ints_to_one_hot(ints, width)
    local height = ints:size()[1]
    local zeros = torch.zeros(height, width)
    local indices = ints:view(-1, 1):long()
    local one_hot = zeros:scatter(2, indices, 1)
    return one_hot
  end
  local row = input:size()[1]
  for i=1,row do
     output[i]=ints_to_one_hot(input[i], vocab)
  end

  return output
end

---------------------------------------------------------------------------
-- This is used to convert 28x200x4224 to 200x28x4224
function decoder_output(input)
  local L,N,Mp1 = input:size(1), input:size(2), input:size(3)--l=28,n=200,mp1=17533
  local D = lmOpt.seq_length -- 26 her
  print('input',input:size())
  print('seq',D)
  assert(D == L-2, 'input Tensor should be 2 larger in time')
  local target=torch.Tensor(lmOpt.seq_length,opt.batch_size,lmOpt.vocab_size+1)
  seclectnet = input:narrow(1, 2,D) -- this is select first dim, from 2 to D(max_value)

  return target
end

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split)
  protos.netE:evaluate()	
  protos.netD:evaluate()

  dataloader:resetIterator(2)-- 2 for test and 1 for train

  local verbose = utils.getopt(evalopt, 'verbose', false) -- to enable the prints statement  entry.image_id, entry.caption
  local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local right_sum = 0
  local loss_text = 0
  total_num = dataloader:getDataNum(2) -- 2 for test and 1 for train-- this will provide total number of example in the image 

  local predictions = {}
  local vocab = dataloader:getVocab()
          
  while true do
    --local data = loader:getBatch{batch_size = opt.batch_size, split = split}
    local batch = dataloader:next_batch_eval(opt)
    --print('Ques_cap_id In eval batch[3]',batch[3])
    local data = {}
    data.questions=batch[1]
    data.label=batch[2]
    data.ques_id=batch[3]

    -------------------------------------------------------------------------------------
    n = n + data.questions:size(1)
    xlua.progress(n, total_num)

    local ques_feat= torch.CudaTensor(opt.batch_size, opt.txtSize)
    local decode_question= data.questions:t()-- bcz in langauage models checks assert(seq:size(1) == self.seq_length) os it should be 26 x 200
    -- bcz this language model needs dimension of size 26x200  
    local input_txt_onehot=one_hot_tensor(data.questions+1,lmOpt.vocab_size+1)--200x26x4224
    input_txt_onehot=input_txt_onehot:cuda()
    -------------------------------------------------------------------------------------------------------------------
    --Forward the question Encoder
    ques_feat:copy(protos.netE:forward(input_txt_onehot))
    -- forward the language model
    local logprobs = protos.netD:forward({ques_feat, decode_question}) -- data.questions=data.labels, img_feat=expanded_feats
    -- Change Dim   
    local decoder_output = decoder_convert_net:forward(logprobs) -- this is select first dim, from 2 to D(max_value)
    -- forward criterion
    local loss = protos.crit:forward(logprobs, decode_question)
    -- real txt
    input_txt_emb1:copy(netT:forward(decoder_output))--input_txt_raw1=logprobs-- twice forward , output will overwrite so we will use copy constructor
    -- get matching text embeddings
    input_txt_emb2:copy(netT:forward(input_txt_onehot))--input_txt_raw2=input_txt_onehot

    local errT, grads = JointEmbeddingLoss(input_txt_emb1, input_txt_emb2)
    -------------------------------------------------------------------------------------------------------------------
    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1
    loss_text = loss_text + errT

    -- forward the model to also get generated samples for each image
    local seq = protos.netD:sample(ques_feat)
    local sents = net_utils.decode_sequence(vocab, seq)
    for k=1,#sents do
        local entry = {image_id = data.ques_id[k], question = sents[k]} -- change here
        -- print('questions to be written to the val_predictions', sents[k])
        table.insert(predictions, entry) -- to save all the alements
        -------------------------------------------------------------------------
        -- for print log
        if verbose then
                print(string.format('image %s: %s', entry.image_id, entry.question))
        end
        ------------------------------------------------------------------------
    end
    -- print('length of sents ', #sents) -------checking 
    if n >= total_num then break end -- this is for complete val example , it should not be more than val total sample. otherwise , repetation example will save in json which will cause error in blue score evalution 
    if n >= opt.val_images_use then break end -- we've used enough images
   
  end
  ------------------------------------------------------------------------
  -- for blue,cider score
  local lang_stats
  if opt.language_eval == 1 then
          lang_stats = net_utils.language_eval(predictions, opt.id)
          local score_statistics = {epoch = epoch, statistics = lang_stats}
          print('Current language statistics',score_statistics)
  end
    ------------------------------------------------------------------------       
    -- write a (thin) json report-- for save image id and question print in json format
  local question_filename = string.format('%s/question_checkpoint_epoch%d', opt.checkpoint_dir, epoch)
  utils.write_json(question_filename .. '.json', predictions) -- for save image id and question print in json format
  print('wrote json checkpoint to ' .. question_filename .. '.json')

------------------------------------------------------------------------
  return loss_sum/loss_evals, predictions, lang_stats,loss_text/loss_evals

end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local iter = 0
local function lossFun()
	protos.netE:training()	
	protos.netD:training()
----------------------------------------------------------------------------
-- Forward pass
-----------------------------------------------------------------------------
  -- get batch of data  
  --local data = loader:getBatch{batch_size = opt.batch_size, split = 0}
  local batch = dataloader:next_batch(opt)        
  local data = {}
  data.questions=batch[1]--200x26
  data.label=batch[2]--200x26
  data.ques_id  = batch[3]

  local ques_feat= torch.CudaTensor(opt.batch_size, opt.txtSize)
  local decode_question= data.questions:t()-- bcz in langauage models checks assert(seq:size(1) == self.seq_length) os it should be 26 x 200
  -- bcz this language model needs dimension of size 26x200  
  local input_txt_onehot=one_hot_tensor(data.questions+1,lmOpt.vocab_size+1)--200x26x4224
  input_txt_onehot=input_txt_onehot:cuda()
  -------------------------------------------------------------------------------------------------------------------
  --Forward the question Encoder
  ques_feat:copy(protos.netE:forward(input_txt_onehot))
  -- forward the language model
  local logprobs = protos.netD:forward({ques_feat, decode_question}) -- data.questions=data.labels, img_feat=expanded_feats
  -- Chanage Dim
  local decoder_output = decoder_convert_net:forward(logprobs) -- this is select first dim, from 2 to D(max_value)
  -- forward the local criterion
  local loss = protos.crit:forward(logprobs, decode_question)
  -- real txt
  input_txt_emb1:copy(netT:forward(decoder_output))--input_txt_raw1=logprobs-- twice forward , output will overwrite so we will use copy constructor
  -- get matching text embeddings
  input_txt_emb2:copy(netT:forward(input_txt_onehot))--input_txt_raw2=input_txt_onehot
  -- Find Global loss
  local errT, grads = JointEmbeddingLoss(input_txt_emb1, input_txt_emb2)
        
-----------------------------------------------------------------------------
-- Backward pass
-----------------------------------------------------------------------------
  grad_eparams:zero()  
  grad_lparams:zero() 
  -- gradParametersT:zero()
-------------------------------------------------------------------------------------------------------------------
    netT:backward(input_txt_onehot, grads[2])-- twice backward, the gradient parameter value it will added not ovetwrite like forward output will overwrite
    netT:forward(decoder_output)
    local grad_text_enc=netT:backward(decoder_output, grads[1])
    local grad_text_enc_narrow=decoder_convert_net:backward(logprobs, grad_text_enc)
    -- backprop local criterion
    local dlogprobs = protos.crit:backward(logprobs, decode_question)
    local grad_text_encoding_total=dlogprobs+grad_text_enc_narrow
    -- backprop Decoder
    local d_lm_feats, ddummy = unpack(protos.netD:backward({ques_feat, decode_question}, grad_text_encoding_total))
    protos.netE:forward(input_txt_onehot)
    -- backprop question Encoder model
    local dummy_ques_feat= protos.netE:backward(input_txt_onehot, d_lm_feats)

---------------------------------------------------------------------------
  local losses = { total_loss = loss,errT=errT }
  return losses
end

-------------------------------------------------------------------------------
--Step 12:--Training Function
-------------------------------------------------------------------------------
local e_optim_state = {}  --- to mentain state in optim
local l_optim_state = {}  --- to mentain state in optim
local netT_optim_state={}

local grad_clip = 0.1
local timer = torch.Timer()
local decay_factor = math.exp(math.log(0.1)/opt.learning_rate_decay_every/opt.iterPerEpoch) -- for lr decay
local learning_rate = opt.learning_rate
-- local decay_factor =0.5

total_train_example = dataloader:getDataNum(1) -- for lr decay
train_nbatch=math.ceil(total_train_example /opt.batch_size)


function Train()
  count_sum=0  -- Cannt be make local bcz it is insisde the function and other function are using this.
  local iter=1	
  local ave_loss = 0  --for iter_log_print  train error
  err=0

  local ave_loss_t = 0  --for iter_log_print  train error
  err_t=0


	while iter <= train_nbatch do
		-- Training loss/gradient
		local losses = lossFun()
		err=err+ losses.total_loss
		ave_loss = ave_loss + losses.total_loss

    err_t=err_t+ losses.errT
    ave_loss_t = ave_loss_t + losses.errT
		---------------------------------------------------------
		-- decay the learning rate  
		if epoch % opt.learning_rate_decay_every ==0 then
                   learning_rate = learning_rate * decay_factor -- set the decayed rate
		end
    if epoch % 15 == 0 and iter < 10 then
        learning_rate = learning_rate * 0.99999 *decay_factor -- set the decayed rate
    end
		---------------------------------------------------------
		if iter % opt.losses_log_every == 0 then
			ave_loss = ave_loss / opt.losses_log_every

      ave_loss_t = ave_loss_t / opt.losses_log_every
			print(string.format('epoch:%d  iter %d: %f, %f,%f, %f', epoch, iter, ave_loss,ave_loss_t,learning_rate, timer:time().real))
			ave_loss = 0
      ave_loss_t = 0
		end
		---------------------------------------------------------
    -- perform a parameter update
    --this will update only netT
    -- adam(parametersT, gradParametersT, 0.0002, opt.optim_alpha, 0.5, opt.optim_epsilon, netT_optim_state)

    -- perform a parameter update
		if opt.optim == 'sgd' then
			sgdm(eparams, grad_eparams, learning_rate, opt.momentum, e_optim_state)
      sgdm(lparams, grad_lparams, learning_rate, opt.momentum, l_optim_state)
		elseif opt.optim == 'rmsprop' then
			rmsprop(eparams, grad_eparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, e_optim_state)			
			rmsprop(lparams, grad_lparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, l_optim_state)      		                		        				
		else
			error('bad option opt.optim')
		end
		---------------------------------------------------------
		iter = iter + 1
    if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
    if loss0 == nil then loss0 = losses.total_loss end
    if losses.total_loss > loss0 * 20 then
      print('loss seems to be exploding, quitting.')
      break
    end

	end
	return err/train_nbatch,err_t/train_nbatch
end
-------------------------------------------------------------------------------
--Step 13:--Log Function
-------------------------------------------------------------------------------
function printlog(epoch,ErrTrain,ErrTest,ErrTrainT,ErrTestT)
  ------------------------------------------------------------------------------
  -- log plot
  paths.mkdir(opt.save)
  err_log:add{['Training Error']= ErrTrain, ['Test Error'] = ErrTest}
  err_log:style{['Training Error'] = '-', ['Test Error'] = '-'}
  err_log:plot()

  errT_log:add{['Training Error text']= ErrTrainT, ['Test Error text'] = ErrTestT}
  errT_log:style{['Training Error text'] = '-', ['Test Error text'] = '-'}
  errT_log:plot()
  ---------------------------------------------------------------------------------
  if paths.filep(opt.save..'/ErrorProgress.eps') or paths.filep(opt.save..'/accuracyProgress.eps') then
    -----------------------------------------------------------------------------------------------------------
    -- convert .eps file as .png file
    local base64im
    do
      os.execute(('convert -density 200 %s/ErrorProgress.eps %s/ErrorProgress.png'):format(opt.save,opt.save))
      os.execute(('openssl base64 -in %s/ErrorProgress.png -out %s/ErrorProgress.base64'):format(opt.save,opt.save))
      local f = io.open(opt.save..'/ErrorProgress.base64')
      if f then base64im = f:read'*all' end
    end
    
    -----------------------------------------------------------------------------------------------------------------------
    -- to display in .html file
    local file = io.open(opt.save..'/report.html','w')
    file:write('<h5>Training data size:  '..total_train_example ..'\n')
    file:write('<h5>Validation data size:  '..total_num ..'\n')
    file:write('<h5>batchSize:  '..opt.batch_size..'\n')
    file:write('<h5>LR:  '..opt.learning_rate..'\n')
    file:write('<h5>optimization:  '..opt.optim..'\n')
    file:write('<h5>drop_prob_lm:  '..opt.drop_prob_lm..'\n')


    file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(opt.save,epoch,base64im))
  
  --[[  for k,v in pairs(optim_state) do
      if torch.type(v) == 'number' then
        file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
    end --]]

    file:write'</table><pre>\n'
    file:write'</pre></body></html>'
    file:close()
  end
--[[
  if opt.visualize then
    require 'image'
    local weights = EmbeddingNet:get(1).weight:clone()
    --win = image.display(weights,5,nil,nil,nil,win)
    image.saveJPG(paths.concat(opt.save,'Filters_epoch'.. epoch .. '.jpg'), image.toDisplayTensor(weights))
  end
--]]
  return 1  
end

local best_score_Bleu_1
local best_score_Bleu_2
local best_score_Bleu_3
local best_score_Bleu_4
local best_score_ROUGE_L
local best_score_METEOR
local best_score_CIDEr

-------------------------------------------------------------------------------
--Step 14:-- Main loop
-------------------------------------------------------------------------------
epoch = 1  -- made gloobal ,bcz inside training function, it is used
print '\n==> Starting Training\n'
while epoch ~= opt.epoch do

  print('Epoch ' .. epoch)
  local ErrTrain,ErrNetT = Train()
  print('Checkpointing. Calculating validation accuracy..')
  local val_loss, val_predictions, lang_stats,val_lossT = eval_split(2)
  print('------------------------------------------------------------------------')
  print('Training Error:  ', ErrTrain ,'Validation loss: ', val_loss)
  print('Training Error text:  ', ErrNetT ,'Validation loss text: ', val_lossT)

  local result=printlog(epoch,ErrTrain,val_loss,ErrNetT,val_lossT)
	-----------------------------------------------------------
  -- To print best score
  local current_score_Bleu_1
  local current_score_Bleu_2
  local current_score_Bleu_3
  local current_score_Bleu_4
  local current_score_ROUGE_L
  local current_score_METEOR
  local current_score_CIDEr
  --local current_score_SPICE
       
        
  if lang_stats then
    -- use CIDEr score for deciding how well we did                
    current_score_Bleu_1 = lang_stats['Bleu_1']
    current_score_Bleu_2 = lang_stats['Bleu_2']
    current_score_Bleu_3 = lang_stats['Bleu_3']
    current_score_Bleu_4 = lang_stats['Bleu_4']
    current_score_ROUGE_L = lang_stats['ROUGE_L']
    current_score_METEOR = lang_stats['METEOR']
    current_score_CIDEr = lang_stats['CIDEr']
    -- current_score_SPICE = lang_stats['SPICE']    
  else
    -- use the (negative) validation loss as a score          
    current_score_Bleu_1 = -val_loss
    current_score_Bleu_2 = -val_loss
    current_score_Bleu_3 = -val_loss
    current_score_Bleu_4 =-val_loss
    current_score_ROUGE_L = -val_loss
    current_score_METEOR = -val_loss
    current_score_CIDEr = -val_loss
    --current_score_SPICE = -val_loss
  end


        
    if best_score_Bleu_1 == nil or current_score_Bleu_1 > best_score_Bleu_1 then
        best_score_Bleu_1 = current_score_Bleu_1
    end

    if best_score_Bleu_2 == nil or current_score_Bleu_2 > best_score_Bleu_2 then
        best_score_Bleu_2 = current_score_Bleu_2
    end

    if best_score_Bleu_3 == nil or current_score_Bleu_3 > best_score_Bleu_3 then
        best_score_Bleu_3 = current_score_Bleu_3
    end

    if best_score_Bleu_4 == nil or current_score_Bleu_4 > best_score_Bleu_4 then
        best_score_Bleu_4 = current_score_Bleu_4
    end

    if best_score_ROUGE_L == nil or current_score_ROUGE_L > best_score_ROUGE_L then
        best_score_ROUGE_L = current_score_ROUGE_L
    end

    if best_score_METEOR == nil or current_score_METEOR > best_score_METEOR then
        best_score_METEOR = current_score_METEOR
    end

    if best_score_CIDEr == nil or current_score_CIDEr > best_score_CIDEr then
        best_score_CIDEr = current_score_CIDEr
    end

    --if best_score_SPICE == nil or current_score_SPICE > best_score_SPICE then
    --       best_score_SPICE = current_score_SPICE
    -- end

    print('-----------------------------------------------------------------------------------------')
    print('current_Bleu_1:', current_score_Bleu_1,'current_Bleu_2:', current_score_Bleu_2,'current_Bleu_3:', current_score_Bleu_3,'current_Bleu_4:', current_score_Bleu_4) 
    print('current_ROUGE_L:', current_score_ROUGE_L, 'current_METEOR:',current_score_METEOR, 'And current_CIDEr:',current_score_CIDEr) 
    print('-----------------------------------------------------------------------------------------')
    print('best_Bleu_1:', best_score_Bleu_1,'best_Bleu_2:', best_score_Bleu_2,'best_Bleu_3:', best_score_Bleu_3,'best_Bleu_4:', best_score_Bleu_4) 
    print('best_ROUGE_L:', best_score_ROUGE_L, 'best_METEOR:',best_score_METEOR, 'And best_CIDEr:',best_score_CIDEr) 
    print('-----------------------------------------------------------------------------------------')
    --print('Current language statistics',lang_stats)      
    ----------------------------------------------------------------------------------------
    -- for print log      
    lang_stats_log:add{['Bleu_1']= current_score_Bleu_1, ['Bleu_2'] = current_score_Bleu_2,['Bleu_3'] = current_score_Bleu_3,['Bleu_4'] = current_score_Bleu_4,['ROUGE_L'] = current_score_ROUGE_L,['METEOR'] = current_score_METEOR,['CIDEr'] = current_score_CIDEr}

    lang_stats_log:style{['Bleu_1']= '-', ['Bleu_2'] = '-',['Bleu_3'] = '-',['Bleu_4'] = '-',['ROUGE_L'] = '-',['METEOR'] = '-',['CIDEr'] = '-'}

    lang_stats_log:plot()	
    -----------------------------------------------------------------------------------

    ---------------------------------------------------------------------------------------------------------------------------------------------
    local model_save_filename = string.format('%s/model_epoch%d.t7', opt.checkpoint_dir, epoch)
    --if epoch % 100==0 then --dont save on very first iteration
    torch.save(model_save_filename, {eparams=eparams,lparams=lparams, lmOpt=lmOpt})  -- vocabulary mapping is included here, so we can use the checkpoint 
    --end
    print('Saving current checkpoint to:', model_save_filename)

	epoch = epoch+1
end

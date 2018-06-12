------------------------------------------------------------------------------
-- this is dataloader for sequential example reading
---------------------------------------------------------------------------------------
require 'hdf5'
cjson = require 'cjson'
utils = require 'misc/utils'

local dataloader = {}

function dataloader:initialize(opt)
    print('Reading ' .. opt.input_json)
    local file = io.open(opt.input_json, 'r')
    local text = file:read()
    file:close()
    local params = cjson.decode(text)
    for k,v in pairs(params) do self[k] = v end
    self['vocab_size'] = 0 for i,w in pairs(self['ix_to_word']) do self['vocab_size'] = self['vocab_size'] + 1 end
----------------------------------------------------------------------------------------------------------
        -- this is getting question information 
        print ('DataLoader loading h5 question file: ',opt.input_ques_h5)
        local qa_data = hdf5.open(opt.input_ques_h5, 'r')

       -- if split == 'train' then
       -- split is not required bcz here , u have chanaged variale name as like ques_train from ques and ques_test from ques which implecite indicate split
                -- question
                self['ques_train']      = qa_data:read('/ques_train'):all()
                self['ques_len_train']  = qa_data:read('ques_length_train'):all()
                --self['ques_train']      = utils.right_align(self['ques1_train'], self['ques1_length_train'])-- you will get  bad argument #1 to 'unpack' (table expected, got nil)--unpack(self.state[t-1])}
                self['ques_id_train']   = qa_data:read('/ques_cap_id_train'):all()


                -- label is also question
                self['label_train']       = qa_data:read('/ques1_train'):all()
                self['label_len_train']   = qa_data:read('ques1_length_train'):all()
                --self['label_train']       = utils.right_align(self['ques2_train'], self['ques2_length_train'])

                self['train_id']  = 1
                self.seq_length = self.ques_train:size(2)
                
                -- to print complete size of each split
                print('self[ques_train]:size(1)',self['ques_train']:size(1))
                
        --elseif split == 'test' then 
        -- split is not required bcz here , u have chanaged variale name as like ques_train from ques and ques_test from ques which implecite indicate split

                -- question
                self['ques_test']      = qa_data:read('/ques_test'):all()
                self['ques_len_test']  = qa_data:read('ques_length_test'):all()
                --self['ques_test']      = utils.right_align(self['ques_test'], self['ques_len_test'])
                self['ques_id_test']   = qa_data:read('/ques_cap_id_test'):all()

                -- label is also question
                self['label_test']       = qa_data:read('/ques1_test'):all()
                self['label_len_test']   = qa_data:read('ques1_length_test'):all()
                --self['label_test']       = utils.right_align(self['cap_test'], self['cap_len_test'])

                self['test_id']   = 1
                -- to print complete size of each split
                print('self[ques_test]:size(1)',self['ques_test']:size(1))
         --end
        qa_data:close()
end

function dataloader:next_batch(opt)
    local start_id = self['train_id'] -- start id , and it  it wiil be remember for next batch
    if start_id + opt.batch_size - 1 <= self['ques_train']:size(1) then 
        end_id = start_id + opt.batch_size - 1        
    else 
        -- start_id = self['train_id']
        -- end_id = self['ques_train']:size(1) - 1 
        -- print('end of epoch')   
        -- self['train_id'] =1  --reset train id to 1 
        print('end of epoch') -- rest(<battch size) of example are ignoreing, like last 130 example are igored if batch size is 200
        self['train_id'] =1  --reset train id to 1
        start_id = self['train_id']
        end_id = start_id + opt.batch_size - 1 
    end
 
    local qinds = torch.LongTensor(end_id - start_id + 1):fill(0) -- to keep track of  question index
    for i = 1, end_id - start_id + 1 do    
        qinds[i] = start_id + i - 1     -- this is required bcz, late batch size < opt.batch_size, so only remain part will be consider
    end

    local ques    = self['ques_train']:index(1, qinds)
    local label     = self['label_train']:index(1, qinds)
    local ques_id = self['ques_id_train']:index(1, qinds)

    if opt.gpuid >= 0 then
        ques   = ques:cuda()        
        label    = label:cuda()
    end
     
    self['train_id'] = self['train_id'] + end_id - start_id + 1   -- self['test_id']=  self.test_id both have same meaning
    return {ques,label,ques_id}
end

function dataloader:next_batch_eval(opt)
    local start_id = self['test_id']
    local end_id = math.min(start_id + opt.batch_size - 1, self['ques_test']:size(1))  --here it do sequential basic because it will check complete data set

    local qinds = torch.LongTensor(end_id - start_id + 1):fill(0)
  
    for i = 1, end_id - start_id + 1 do
        qinds[i] = start_id + i - 1
    end

    local ques    = self['ques_test']:index(1, qinds)
    local ques_id = self['ques_id_test']:index(1, qinds)
    local label     = self['label_test']:index(1, qinds)

    if opt.gpuid >= 0 then
      ques   = ques:cuda()
      label    = label:cuda()
    end
    -- print('Ques_cap_id AFTER',ques_id)
    self['test_id'] = self['test_id'] + end_id - start_id + 1   -- self['test_id']=  self.test_id both have same meaning

    return {ques,label,ques_id}
end
function dataloader:getVocab(opt)
     return self.ix_to_word
end

function dataloader:getVocabSize()
    return self['vocab_size'] -- or self.vocab_size
end

function dataloader:resetIterator(split)
        if split ==1 then 
                self['train_id'] = 1
        end
        if split ==2  then
                self['test_id']=1
        end
end


function dataloader:getDataNum(split)
        if split ==1 then 
               return self['ques_train']:size(1)
        end
        if split ==2  then
             return  self['ques_test']:size(1)
        end
end

function dataloader:getSeqLength()
  return self.seq_length
end


return dataloader

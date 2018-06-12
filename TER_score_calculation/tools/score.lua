require('torch')
local path = require('pl.path')

require('onmt.init')
local tokenizer = require('tools.utils.tokenizer')

local cmd = onmt.utils.ExtendedCmdLine.new('scorer.lua')

local options = {
  {
    '-scorer', 'bleu',
    [[Scorer to use.]],
    {
      enum = onmt.scorers.list
    }
  },
  {
    'rfilestem',
    'string',
    [[Reads the reference(s) from `name` or `name0`, `name1`, ...]],
    {
      valid = onmt.utils.ExtendedCmdLine.nonEmpty
    }
  },
  {
    '-sample',
    1,
    [[If > 1, number of samples for estimation of k-fold error margin (95% certitude) - 10 is a good value.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt()
    }
  },
  {
    '-order',
    4,
    [[Number of sample for estimation of error margin.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt()
    }
  },
  {
    '-tokenizer',
    'space',
    [[Tokenizer to use - space tokenization by default, `max` is corresponding to tokenizer -mode=aggressive,
      -segment_alphabet=Han,Kanbun,Katakana,Hiragana, -segment_alphabet_change. `max` is the best choice for all languages,
      detokenized translation, space applies for pre-tokenized corpus.]],
    {
      enum = { 'space', 'max' }
    }
  },

  {
    'hfilestem',
    'string',
    [[Reads the hypothessis(es) from `hyp`]],
    {
      valid = onmt.utils.ExtendedCmdLine.nonEmpty
    }
  },
 }

cmd:setCmdLineOptions(options)

onmt.utils.Logger.declareOpts(cmd)

local function main()
  local opt = cmd:parse(arg)

  local tok_options = { mode='space' }
  if opt.tokenizer == 'max' then
    tok_options['segment_alphabet'] = { 'Han', 'Kanbun', 'Katakana', 'Hiragana' }
    tok_options['segment_alphabet_change'] = true
    tok_options['mode'] = 'aggressive'
  end

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level, opt.log_tag)
  _G.hookManager = require('onmt.utils.HookManager').new()

local filenameHy=opt.hfilestem
print('filenameHy',filenameHy)
print('opt.rfilestem',opt.rfilestem)

  -- read the references
  local references = {}

  local function add_to_reference(filename)
    local file = io.open(filename)
    onmt.utils.Error.assert(file, "cannot open `" .. filename .. "`")
    local ref = {}
    while true do
      local line = file:read()
      if not line then break end
      table.insert(ref, tokenizer.tokenize(tok_options, line))
    end
    onmt.utils.Error.assert(#references==0 or #references[#references] == #ref, "all references do not have same line count")
    table.insert(references, ref)
  end

  local refid = 0

 print('path.exists(opt.rfilestem)',path.exists(opt.rfilestem))

  if path.exists(opt.rfilestem) then
    add_to_reference(opt.rfilestem)
   -- print('references after ',references)
  end

  if onmt.scorers.multi[opt.scorer] then
    while path.exists(opt.rfilestem .. refid) do
      add_to_reference(opt.rfilestem .. refid)
      refid = refid + 1
    end
  end

  local hyp = {}
  -- -- read from stdin
  -- while true do
  --   local line = io.read()
  --   if not line then break end
  --   table.insert(hyp, tokenizer.tokenize(tok_options, line))
  -- end

local fileHy = io.open(filenameHy)
-- print('fileHy',fileHy)
 --   onmt.utils.Error.assert(file, "cannot open `" .. fileHy .. "`")
    local refHy = {}
    while true do
      local line = fileHy:read()
      if not line then break end
      table.insert(refHy, tokenizer.tokenize(tok_options, line))
    end
   -- onmt.utils.Error.assert(#references==0 or #references[#references] == #refHy, "all references do not have same line count")
    table.insert(hyp, refHy)



 -- onmt.utils.Error.assert(#hyp==#references[1], "line count hyp/ref does not match")

  if not onmt.scorers.multi[opt.scorer] then
    references = references[1]
  end


  -- print('hyp',references)
  local score, format = onmt.scorers[opt.scorer](hyp[1], references, opt.order)
  score = string.format("%.2f", score*100)
  local margin = ''

 print('opt.sample',opt.sample)
  if opt.sample > 1 then
    local scores = torch.Tensor(opt.sample)
    for k = 1, opt.sample do
      -- extract 1/2 random sample
      local perm = torch.randperm(#hyp)
      local nhyp = {}
      local nref = {}
      for _ = 1, #references do
        table.insert(nref, {})
      end
      for i = 1, #hyp/2 do
        table.insert(nhyp, hyp[perm[i]])
        for j = 1, #references do
          table.insert(nref[j], references[j][perm[i]])
        end
      end
      scores[k] = onmt.scorers[opt.scorer](nhyp, nref, opt.order)
    end
    score = string.format("%.2f", torch.mean(scores)*100)
    margin = string.format("+/-%.2f", torch.std(scores)*1.96*100)
  end
  print(score, margin, format)
end

main()

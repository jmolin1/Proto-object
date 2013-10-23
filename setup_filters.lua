--function: setup filter for 10 x 10
function setupFilter(filter, filter_size)
   
   if torch.numel(filter) > 81 then
      print('ERROR - Filter is greater than 10 x 10!');
      return nil;

   elseif torch.numel(filter) == 81 then
      return torch.DoubleTensor(1,filter:size(1),filter:size(2)):copy(filter);

   else
      local curr_rows = filter:size(1);
      local curr_cols = filter:size(2);
      local pad_cols = (filter_size - curr_cols) / 2;
      local pad_rows = (filter_size - curr_rows) / 2;
      
      local padder = nn.SpatialZeroPadding(pad_cols,pad_cols,pad_rows,pad_rows)
      local filter = torch.DoubleTensor(1,filter:size(1),filter:size(2)):copy(filter)
      padder:forward(filter); 
      local new_filter = padder.output:clone();
      
      return new_filter;
   end
end

function createFilters(defaultParameters,MAX_FILTER_SIZE)

   ----------------------------------------------
   --SETUP FILTER VALUE IN SPATIAL CONVOLUTIONS--
   ----------------------------------------------
   local EVMSK = torch.DoubleTensor(1,defaultParameters.evenCellsPrs.numOri):zero();
   local ODDMSK = torch.DoubleTensor(1,defaultParameters.oddCellsPrs.numOri):zero();
   local GABORMSK = torch.DoubleTensor(1,defaultParameters.gaborPrs.numOri):zero();
   local CSMSK = 0;
   local VMSK1 = torch.DoubleTensor(1,defaultParameters.vmPrs.numOri):zero();
   local VMSK2 = torch.DoubleTensor(1,defaultParameters.vmPrs.numOri):zero();
   
   ---------------------------------
   --SETUP ALL FILTERS FOR NEUFLOW--
   ---------------------------------
   local NUM_OF_FILTERS = 1 + torch.numel(EVMSK) + torch.numel(ODDMSK) + torch.numel(GABORMSK) + torch.numel(VMSK1) + torch.numel(VMSK2);
   
   --network = nn.Sequential()
   local conv = nn.SpatialConvolution(1,NUM_OF_FILTERS,MAX_FILTER_SIZE,MAX_FILTER_SIZE,1,1);
   conv.bias = torch.zero(conv.bias)
   
   --Initialize filter_cnt
   local filter_cnt = 1;
   
   --Setup Even Pyramid mask (4 orientations)
   local prs_even = defaultParameters.evenCellsPrs;
   local Evmsk,Evmsk_final;
   for local_ori = 1,prs_even.numOri do
      Evmsk = makeEvenOrientationCells(prs_even.oris[local_ori],prs_even.lambda,prs_even.sigma,prs_even.gamma);
      Evmsk_final = setupFilter(torch.DoubleTensor(Evmsk),MAX_FILTER_SIZE);
      conv.weight[filter_cnt] = torch.DoubleTensor(Evmsk_final);
      EVMSK[1][local_ori] = filter_cnt;
      filter_cnt = filter_cnt + 1;
   end
   
   --Setup Odd Pyramid mask (4 orientations)
   local prs_odd = defaultParameters.oddCellsPrs;
   local Oddmsk1,Oddmsk2,Oddmsk_final;
   for local_ori = 1,prs_odd.numOri do
      Oddmsk1, Oddmsk2 = makeOddOrientationCells(prs_odd.oris[local_ori],prs_odd.lambda,prs_odd.sigma,prs_odd.gamma);
      Oddmsk_final = setupFilter(torch.DoubleTensor(Oddmsk1),MAX_FILTER_SIZE);
      conv.weight[filter_cnt] = torch.DoubleTensor(Oddmsk_final);
      ODDMSK[1][local_ori] = filter_cnt;
      filter_cnt = filter_cnt + 1;
   end
   
   --Setup Gabor/Orientation mask (4 orientations)
   local prs_gabor = defaultParameters.gaborPrs;
   local gabormsk,Gabormsk_final;
   for local_ori = 1,prs_gabor.numOri do
      Gabormsk = makeEvenOrientationCells(prs_gabor.oris[local_ori],prs_gabor.lambda,prs_gabor.sigma,prs_gabor.gamma);
      Gabormsk_final = setupFilter(torch.DoubleTensor(Gabormsk),MAX_FILTER_SIZE);
      conv.weight[filter_cnt] = torch.DoubleTensor(Gabormsk_final);
      GABORMSK[1][local_ori] = filter_cnt;
      filter_cnt = filter_cnt + 1;
   end
   
   --Setup Center Surround mask (1 center surround)
   local prs_cs = defaultParameters.csPrs;
   local CSmsk,CSmsk_final;
   CSmsk = makeCenterSurround(prs_cs.inner,prs_cs.outer);
   CSmsk_final = setupFilter(torch.DoubleTensor(CSmsk),MAX_FILTER_SIZE);
   conv.weight[filter_cnt] = torch.DoubleTensor(CSmsk_final);
   CSMSK = filter_cnt;
   filter_cnt = filter_cnt + 1;
   
   --Setup Von Mises masks (4 orientations, 2 types)
   local prs_vm = defaultParameters.vmPrs;
   local VMmsk1,VMmsk2,VMmsk1_final,VMmsk2_final;
   local dim1 = {};
   local idx = 1;
   for i=-3*prs_vm.R0,3*prs_vm.R0 do
      dim1[idx] = i;
      idx = idx + 1;
   end
   dim1 = torch.DoubleTensor(dim1);
   local dim2 = dim1:clone();
   
   for local_ori = 1,prs_vm.numOri do
      VMmsk1,VMmsk2 = makeVonMises(prs_vm.R0, prs_vm.oris[local_ori] + (math.pi / 2), dim1, dim2);
      
      VMmsk1_final = setupFilter(torch.DoubleTensor(VMmsk1),MAX_FILTER_SIZE);
      VMmsk2_final = setupFilter(torch.DoubleTensor(VMmsk2),MAX_FILTER_SIZE);
      conv.weight[filter_cnt] = torch.DoubleTensor(VMmsk1_final);
      VMSK1[1][local_ori] = filter_cnt;
      filter_cnt = filter_cnt + 1;
      
      conv.weight[filter_cnt] = torch.DoubleTensor(VMmsk2_final);
      VMSK2[1][local_ori] = filter_cnt;
      filter_cnt = filter_cnt + 1;
   end
   
   return conv,EVMSK,ODDMSK,GABORMSK,CSMSK,VMSK1,VMSK2;
end

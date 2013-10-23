---------------------------
--IMAGE-RELATED FUNCTIONS--
---------------------------

--
--function: Pre-Pad Image
--
function prePadImage(orig_img,kRows,kCols,padValue)
   local pad_rows = math.floor(kRows / 2);
   local pad_cols = math.floor(kCols / 2);
   
   --initially pad with zeros
   local padder = nn.SpatialZeroPadding(pad_cols,pad_cols,pad_rows,pad_rows)
   padder:forward(orig_img);
   local new_image = padder.output;
   
   --pad with padValue
   if (padValue ~= nil) then
      new_image[1][{{},{1,pad_cols}}]:fill(padValue);
      new_image[1][{{},{new_image:size(3) - pad_cols + 1,new_image:size(3)}}]:fill(padValue);
      new_image[1][{{1,pad_rows},{}}]:fill(padValue);
      new_image[1][{{new_image:size(2) - pad_rows + 1,new_image:size(2)},{}}]:fill(padValue);
      
   --pads with value of closest border pixel
   else
      local first_col = new_image[1][{{},{pad_cols + 1}}];
      local cols_to_pad = torch.expand(first_col,first_col:size(1),pad_cols);
      new_image[1][{{},{1,pad_cols}}] = cols_to_pad:clone();
      
      local last_col = new_image[1][{{},{new_image:size(3) - pad_cols}} ];
      cols_to_pad = torch.expand(last_col,last_col:size(1),pad_cols);
      new_image[1][{{},{new_image:size(3) - pad_cols + 1,new_image:size(3)}}] = cols_to_pad:clone();
      
      local first_row = new_image[1][{{pad_rows + 1},{}}];
      local rows_to_pad = torch.expand(first_row,pad_rows,first_row:size(2));
      new_image[1][{{1,pad_rows},{}}] = rows_to_pad:clone();
      
      local last_row = new_image[1][{{new_image:size(2) - pad_rows},{}}];
      rows_to_pad = torch.expand(last_row,pad_rows,last_row:size(2));
      new_image[1][{{new_image:size(2) - pad_rows + 1,new_image:size(2)},{}}] = rows_to_pad:clone();
   end

   return new_image;
end

--
--function: (Get) Mex Local Maxima
--
function mexLocalMaxima(data,thresh)
   --refData = torch.squeeze(torch.DoubleTensor(data)):clone();
   local refData = torch.squeeze(data);
   local temp_data = torch.squeeze(data);
   local end_row = refData:size()[1];
   local end_col = refData:size()[2];
   refData = refData[{{2,end_row-1},{2,end_col-1}}];

   local and_true_val = 5;
   local sum1 = torch.add( torch.ge(refData,temp_data[{{1,end_row-2},{2,end_col-1}}]), torch.ge(refData,temp_data[{{3,end_row},{2,end_col-1}}]));
   local sum2 = torch.add(sum1,torch.ge(refData,temp_data[{{2,end_row-1},{1,end_col-2}}]))
   local sum3 = torch.add(sum2, torch.ge(refData,temp_data[{{2,end_row-1},{3,end_col}}]));
   local sum4 = torch.add(sum3,torch.ge(refData,thresh));
   local localMax = torch.eq(sum4,and_true_val);
   local maxData = refData[localMax];
   
   local lm_avg;
   local lm_sum;
   local lm_num;

   if(torch.numel(maxData) > 0) then
      lm_avg = torch.mean(maxData);
      lm_sum = torch.sum(maxData);
      lm_num = torch.numel(maxData);
   else
      print("Error in Mex Local Maxima");
      lm_avg = 0;
      lm_sum = 0;
      lm_num = 0;
   end
   
   return lm_avg, lm_num, lm_sum;
end

--
--function: Max Normalize Local Max
--
function maxNormalizeLocalMax(data,minmax)
   if (minmax == nil) then
      minmax = torch.DoubleTensor({0,10});
   end

   local temp_data = torch.DoubleTensor(data);
   temp_data = clamp(temp_data,0);

   data = normalizeImage(temp_data,minmax);
   
   if (minmax[1] == minmax[2]) then
      thresh = 0.6;
   else
      thresh = minmax[1] + ((minmax[2] - minmax[1]) / 10);
   end
   
   local lm_avg;
   local lm_num;
   local lm_sum;

   lm_avg,lm_num,lm_sum = mexLocalMaxima(data,thresh);
   
   local result;

   if(lm_num > 1) then
      result = data * ((minmax[2] - lm_avg)^2);
   elseif (lm_num == 1) then
      result = data * (minmax[2]^2);
   else
      result = data;
   end

   return result;
end

--
--function: Normalize Image
--
function normalizeImage(im,range)
   if(range == nil) then
      range = torch.DoubleTensor({0,1});
   end
   if ( (range[1] == 0) and (range[2] == 0) ) then
      local res = torch.DoubleTensor(im);
      return res;
   else
      local mx = torch.max(im);
      local mn = torch.min(im);
      local res_im;

      if(mx == mn) then
         if mx == 0 then
            res_im = torch.mul(im,0);
         else
            res_im = im - mx + (0.5 * torch.sum(range));
         end
      else
         res_im = (torch.div((im - mn),(mx - mn)) * math.abs(range[2] - range[1])) + torch.min(range);
      end

      return res_im;
   end
end

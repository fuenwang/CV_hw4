% Starter code prepared by James Hays
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [bboxes, confidences, image_ids] = .... 
    run_detector(test_scn_path, w, b, feature_params)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression.
tic;
test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);
scale_factor = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05];
scale_num = length(scale_factor);
cell_size = feature_params.hog_cell_size;
temp_size = feature_params.template_size;
jump = temp_size / cell_size;
L = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;
for i = 1:length(test_scenes)
    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    cur_bboxes = zeros(0, 4);
    cur_confidences = zeros(0, 0);
    cur_image_ids = {};
    for scale = scale_factor
        %scale
        img_scale = imresize(img, scale);
        [height, width] = size(img_scale);
        img_hog = vl_hog(img_scale, cell_size);
        num_of_cell_row = size(img_hog, 1);
        num_of_cell_col = size(img_hog, 2);
        %num_of_window_row = floor(num_of_cell_row / (jump));
        %num_of_window_col = floor(num_of_cell_col / (jump));
        num_of_window_row = num_of_cell_row - jump + 1;
        num_of_window_col = num_of_cell_col - jump + 1;
        features = zeros(num_of_window_row* num_of_window_col, L);
        index = 1;
        row_map = zeros(1, num_of_window_row* num_of_window_col);
        col_map = zeros(1, num_of_window_row* num_of_window_col);
        for start_row = 1:num_of_window_row
            for start_col = 1:num_of_window_col
                end_row = start_row + jump - 1;
                end_col = start_col + jump - 1;
                patch = img_hog(start_row:end_row, start_col:end_col, :);
                %L
                %size(patch)
                features(index, :) = reshape(patch, 1, L);
                row_map(index) = (start_row-1) * jump + 1;
                col_map(index) = (start_col-1) * jump + 1;
                index = index + 1;
                %error('GG');
            end
        end
        index_vec = 1:(num_of_window_row*num_of_window_col);
        score_vec = features * w + b;
        binary = score_vec' > 0.9;
        %binary = score_vec' == score_vec';
        indice = index_vec(binary);
        scale_cur_y_min = row_map(indice)';
        scale_cur_x_min = col_map(indice)';
        %fprintf('H:%d      %d\n', height, scale_cur_y_min)
        %fprintf('W:%d      %d\n', width, scale_cur_x_min)
        scale_cur_y_max = scale_cur_y_min + temp_size - 1;
        scale_cur_x_max = scale_cur_x_min + temp_size - 1;
        scale_cur_bboxes = [scale_cur_x_min, scale_cur_y_min, scale_cur_x_max, scale_cur_y_max] / scale;
        scale_cur_confidences = score_vec(indice)';
        size(scale_cur_bboxes, 1);
        %{
        for k = 1:size(scale_cur_bboxes, 1)
            %cur_image_ids = strvcat(cur_image_ids, test_scenes(i).name);
            cur_image_ids(end+1,:) = {test_scenes(i).name};
        end
        %}
        cur_image_ids = vertcat(cur_image_ids, repmat({test_scenes(i).name}, size(scale_cur_bboxes, 1), 1));
        
        cur_bboxes = vertcat(cur_bboxes, scale_cur_bboxes);
        cur_confidences = vertcat(cur_confidences, scale_cur_confidences');
        %cur_bboxes(end+1, :) = scale_cur_bboxes;
        %cur_confidences(end+1, :) = scale_cur_confidences;
        %error('gg')
    end
    %You can delete all of this below.
    % Let's create 15 random detections per image
    %{
    cur_x_min = rand(15,1) * size(img,2);
    cur_y_min = rand(15,1) * size(img,1);
    cur_bboxes = [cur_x_min, cur_y_min, cur_x_min + rand(15,1) * 50, cur_y_min + rand(15,1) * 50];
    cur_confidences = rand(15,1) * 4 - 2; %confidences in the range [-2 2]
    cur_image_ids(1:15,1) = {test_scenes(i).name};
    %}
    %non_max_supr_bbox can actually get somewhat slow with thousands of
    %initial detections. You could pre-filter the detections by confidence,
    %e.g. a detection with confidence -1.1 will probably never be
    %meaningful. You probably _don't_ want to threshold at 0.0, though. You
    %can get higher recall with a lower threshold. You don't need to modify
    %anything in non_max_supr_bbox, but you can.
    %size(cur_confidences)
    %error('gg')
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));   
    %cur_image_ids
    %size(cur_bboxes)
    %size(cur_confidences)
    %is_maximum
    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);
 
    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
end
toc;





import copy
import logging

import numpy as np
import torch
import random
import cv2
from PIL import Image

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from pycocotools import mask as coco_mask
from pycocotools.mask import encode, decode, frPyObjects

import torchvision.transforms as transforms

from hyperseg.model.tracker.box_ops import box_xyxy_to_cxcywh, get_mask_box_from_json, get_mask_from_json

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)

def draw_circle(mask, center, radius):
    y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
    distance = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    mask[distance <= radius] = 1


def _get_dummy_anno(num_classes=-1, has_mask=True):
    anno = {
        "iscrowd": 0,
        "category_id": num_classes,
        "id": -1,
        "bbox": np.array([0, 0, 0, 0]),
        "bbox_mode": BoxMode.XYXY_ABS,
    }
    if has_mask:
        anno["segmentation"] = [np.array([0.0] * 6)]
    return anno



def filter_empty_instances_soft(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1 # invalid instances are marked with -1
    return instances




def enhance_with_circles(binary_mask, radius=5):
    if not isinstance(binary_mask, np.ndarray):
        binary_mask = np.array(binary_mask)

    binary_mask = binary_mask.astype(np.uint8)

    output_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    points = np.argwhere(binary_mask == 1)
    for point in points:
        draw_circle(output_mask, (point[0], point[1]), radius)
    return output_mask


def is_mask_non_empty(rle_mask):
    if rle_mask is None:
        return False
    binary_mask = decode(rle_mask)
    return binary_mask.sum() > 0


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(cfg):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    # if cfg.INPUT.RANDOM_FLIP != "none":
    #     augmentation.append(
    #         T.RandomFlip(
    #             horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
    #             vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
    #         )
    #     )

    augmentation.extend([
        # T.ResizeScale(
        #     min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        # ),
        T.ResizeShortestEdge(
            short_edge_length=image_size, max_size=image_size
        ),   
        # crop (image_size, image_size)
        T.FixedSizeCrop(crop_size=(image_size, image_size), seg_pad_value=0),
    ])

    return augmentation


def coco_video_build_transform_gen():
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """

    min_size = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800] # [320, 352, 392, 416, 448, 480, 512, 544, 576, 608, 640]

    max_size = 1333
    sample_style = 'choice'

    tfm_gens = []

    tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    
    return tfm_gens



def find_bounding_box(mask, margin=10):

    merge_masks = torch.sum(mask, dim=0, keepdim=False)
        
    coords = torch.nonzero(merge_masks, as_tuple=False)

    if coords.size(0) == 0:
        return [0, 0, 0, 0]
    
    y_min, x_min = torch.min(coords, dim=0)[0]
    y_max, x_max = torch.max(coords, dim=0)[0]

    y_min = max(y_min - margin, 1)
    x_min = max(x_min - margin, 1)
    y_max = min(y_max + margin, mask.shape[1] - 1)
    x_max = min(x_max + margin, mask.shape[2] - 1)
    
    return [x_min, y_min, x_max, y_max]


def get_boxes(instances, needing_convert=True):

    gt_boxes = instances.gt_boxes
    h, w = instances.image_size
    image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=gt_boxes.tensor.dtype)
    gt_boxes = gt_boxes.tensor / image_size_xyxy
    if needing_convert:
        gt_boxes = box_xyxy_to_cxcywh(gt_boxes)

    return gt_boxes

class COCOInstanceNewBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = build_transform_gen(cfg)
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.sampling_frame_range = 10
        self.sampling_interval = 1
        self.sampling_frame_num = 3
        self.sampling_frame_shuffle = False

        self.resize_transform = transforms.Resize((640, 640), interpolation=Image.NEAREST)

        self.coco_video_crop_gen = [
            T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop('absolute_range', crop_size=(384, 600)),
        ]
        self.coco_video_tfm_gens = coco_video_build_transform_gen()



    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def preprocess(self, dataset_dict, region_mask_type=None,clip_image_processor=None, mask_format='polygon', crop_frame=False, data_aug=False):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        if isinstance(dataset_dict["file_name"],str):
            image = utils.read_image(dataset_dict["file_name"], format='RGB')
        else:
            image = np.array(dataset_dict["file_name"])
        utils.check_image_size(dataset_dict, image)

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        if data_aug:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.coco_video_tfm_gens + self.tfm_gens, image)
            else:
                image, transforms = T.apply_transform_gens(
                    self.coco_video_tfm_gens[:-1] + self.coco_video_crop_gen + self.coco_video_tfm_gens[-1:] + self.tfm_gens, image)
        else:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)


        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["image"] = (image - self.pixel_mean) / self.pixel_std
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        dataset_dict['transforms'] = clip_image_processor
        region_masks = []

        if 'vp_image_file_name' in dataset_dict:
            
            vp_image_clip = cv2.imread(dataset_dict["vp_image_file_name"])
            vp_image_clip = cv2.cvtColor(vp_image_clip, cv2.COLOR_BGR2RGB)

            
            
            dataset_dict['vp_transforms'] = clip_image_processor
            vp_region_masks = []
            vp_fill_number = []
            vp_annotations = dataset_dict.pop("vp_annotations")

            for obj in vp_annotations:
                segm = obj['segmentation']
                mask = decode(segm)
                vp_region_masks.append(mask)

            # obj_num h w
            vp_region_masks = torch.cat([torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0) for mask in vp_region_masks])
            # print(vp_region_masks.shape)
            if crop_frame:
                vp_box = find_bounding_box(vp_region_masks, margin=10)
                x_min, y_min, x_max, y_max = vp_box
                # print(vp_box)
                vp_image_clip = vp_image_clip[y_min:y_max, x_min:x_max]
                vp_region_masks = vp_region_masks[:, y_min:y_max, x_min:x_max]

            vp_image_clip = clip_image_processor.preprocess(
                vp_image_clip, return_tensors="pt")["pixel_values"][0]
            dataset_dict["vp_image"] = vp_image_clip

            for vp_anno in vp_annotations:
                # vp_region_mask = vp_anno['segmentation']
                vp_fill_number.append(int(vp_anno['category_id']))
                # vp_region_masks.append(vp_region_mask)



        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                # Let's always keep mask
                # if not self.mask_on:
                #     anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            annotations = dataset_dict['annotations']

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            if len(annos) ==0:
                print('error')

            filter_annos = []

            if 'point_visual_prompt_mask' in annos[0]:
                if region_mask_type is None:
                    region_mask_type = ['point_visual_prompt_mask', 'mask_visual_prompt_mask', 'box_visual_prompt_mask',
                                        'scribble_visual_prompt_mask']
                # region_mask_type = ['mask_visual_prompt_mask', ]
                trial = 0
                while trial < 200:
                    filter_annos = []
                    region_masks = []
                    for anno in annos:
                        non_empty_masks = []
                        for mask_type in region_mask_type:
                            if is_mask_non_empty(anno[mask_type]):
                                non_empty_masks.append(mask_type)
                        # assert non_empty_masks, 'No visual prompt found in {}'.format(dataset_dict['file_name'])
                        if len(non_empty_masks) == 0:
                            continue
                        used_mask_type = random.choice(non_empty_masks)
                        region_mask = decode(anno[used_mask_type])
                        if used_mask_type in ['point_visual_prompt_mask', 'scribble_visual_prompt_mask']:
                            radius = 10 if used_mask_type == 'point_visual_prompt_mask' else 5
                            region_mask = enhance_with_circles(region_mask, radius)
                        # scale_region_mask = transforms.apply_segmentation(region_mask)
                        region_mask = clip_image_processor.preprocess(torch.from_numpy(np.ascontiguousarray(region_mask)).unsqueeze(0), do_rescale=False, do_normalize=False, return_tensors="pt")["pixel_values"][0]
                        # region_mask = torch.from_numpy(np.array(self.resize_transform(Image.fromarray(region_mask)))).unsqueeze(0)
               
                        if region_mask[0].nonzero().shape[0] <=0:
                            print('cur region mask is empty')
                            continue
                        region_masks.append(region_mask)
                        filter_annos.append(anno)
                    if len(filter_annos) == 0:
                        trial += 1
                        print(f'region trial {trial}')
                    else:
                        trial = 3000
            
                if len(region_masks) == 0:
                    return None
                
            if len(filter_annos) == 0:
                filter_annos = annos
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            # instances = utils.annotations_to_instances(annos, image_shape)
            instances = utils.annotations_to_instances(filter_annos, image_shape, mask_format='bitmask')
            if 'category_id' in filter_annos[0]:
                classes = [obj["category_id"] for obj in filter_annos]
                instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            if 'lvis_category_id' in filter_annos[0]:
                lvis_classes = [int(obj["lvis_category_id"]) for obj in annos]
                lvis_classes = torch.tensor(lvis_classes, dtype=torch.int64)
                instances.lvis_classes = lvis_classes
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            # non_empty_instance_mask = [len(obj.get('segmentation', [])) > 0 for obj in annos]
            non_empty_instance_mask = [len(obj.get('segmentation', [])) > 0 for obj in filter_annos]

            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                if hasattr(gt_masks,'polygons'):
                    gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                else:
                    gt_masks = gt_masks.tensor.to(dtype=torch.uint8)
                instances.gt_masks = gt_masks
            instances.gt_boxes = get_boxes(instances, needing_convert=True)

            if region_masks:
                len_region_mask, len_instance= len(region_masks), len(instances)
                if len_region_mask != len_instance:
                    return None
                # region_masks = [m for m, keep in zip(region_masks, non_empty_instance_mask) if keep]
                # assert len(region_masks) == len(instances), f'The number of region masks:{len_region_mask} must match the number of instances:{len_instance}'
                region_masks = torch.cat(region_masks)
                # region_masks = BitMasks(
                #     torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in region_masks])
                # )
                instances.region_masks = region_masks

            if 'vp_image_file_name' in dataset_dict:
                # vp_region_masks = BitMasks(
                #     torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in vp_region_masks])
                # )
                # vp_region_masks = torch.cat([torch.from_numpy(np.array(self.resize_transform(Image.fromarray(mask)))).unsqueeze(0) for mask in vp_region_masks])
               
                vp_region_masks = torch.cat([clip_image_processor.preprocess(mask.unsqueeze(0), do_rescale=False, do_normalize=False, return_tensors="pt")["pixel_values"][0] for mask in vp_region_masks])
                instances.vp_region_masks = vp_region_masks
                instances.vp_fill_number = torch.tensor(vp_fill_number, dtype=torch.int64)


            dataset_dict["instances"] = instances

        return dataset_dict




    def preprocess_reason(self, dataset_dict, mask_format='polygon', data_aug=False, is_train=True):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        image = utils.read_image(dataset_dict["file_name"], format='RGB')

        json_path = dataset_dict['json_path'] # .replace(".json", '_new.json')
        masks, sents, is_sentence = get_mask_from_json(json_path, image)

        dataset_dict['sentences'] = sents
        dataset_dict['height'], dataset_dict['width'] = image.shape[:2]
        


        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        if data_aug and not is_train:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.coco_video_tfm_gens + self.tfm_gens, image)
            else:
                image, transforms = T.apply_transform_gens(
                    self.coco_video_tfm_gens[:-1] + self.coco_video_crop_gen + self.coco_video_tfm_gens[-1:] + self.tfm_gens, image)
        else:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)


        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["image"] = (image - self.pixel_mean) / self.pixel_std
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        instances = Instances(image_shape)
        if is_train:
            masks = transforms.apply_segmentation(masks)
        masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in [masks]])
        )
        instances.gt_masks = masks
        instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        instances.gt_masks = masks.tensor.to(dtype=torch.uint8)

        classes = torch.tensor([1], dtype=torch.int64)
        instances.gt_classes = classes
            
        dataset_dict["instances"] = instances



        return dataset_dict

    def preprocess_revos_test(self, ori_dataset_dict, is_train=False, clip_image_processor=None,):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        
        video_length = ori_dataset_dict["length"]


        dataset_dict = copy.deepcopy(ori_dataset_dict)  # it will be modified by code below
        
        selected_idx = range(video_length)

        # selected_idx is a List of length self.sampling_frame_num

        dataset_dict["instances"] = []
        file_names = dataset_dict.pop("file_names", None) # List

        
        dataset_dict["image"] = []
        dataset_dict["padding_mask"] = []
        dataset_dict["file_name"] = []
        dataset_dict['transforms'] = []

        for frame_idx in selected_idx:
            dataset_dict["file_name"].append(file_names[frame_idx])

            # Read image
            image = utils.read_image(file_names[frame_idx], format='RGB')
            try:
                utils.check_image_size(dataset_dict, image)
            except:
                print(f'wrong image file: {file_names[frame_idx]}')
                return None
            
            origin_image_shape = image.shape[:2]
            padding_mask = np.ones(image.shape[:2])
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            # the crop transformation has default padding value 0 for segmentation
            padding_mask = transforms.apply_segmentation(padding_mask)
            padding_mask = ~ padding_mask.astype(bool)


            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            dataset_dict["image"].append((image - self.pixel_mean) / self.pixel_std)
            dataset_dict["padding_mask"].append(torch.as_tensor(np.ascontiguousarray(padding_mask)))
            dataset_dict['transforms'].append(transforms)

        dataset_dict["image"] = torch.stack(dataset_dict["image"], dim=0)
        dataset_dict["padding_mask"] = torch.stack(dataset_dict["padding_mask"], dim=0)

        return dataset_dict




    

def build_transform_gen_for_eval(cfg):
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    # if cfg.INPUT.RANDOM_FLIP != "none":
    #     augmentation.append(
    #         T.RandomFlip(
    #             horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
    #             vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
    #         )
    #     )

    augmentation.extend([
        T.ResizeShortestEdge(
            short_edge_length=image_size, max_size=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size), seg_pad_value=0),
    ])

    return augmentation


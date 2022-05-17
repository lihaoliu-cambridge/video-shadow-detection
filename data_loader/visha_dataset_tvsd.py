
import os
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
from torch.utils.data import Dataset
from .augmentations_tvsd import get_train_joint_transform, get_val_joint_transform, get_img_transform, get_target_transform, get_val_joint_transform


class ViSha_Dataset(Dataset):
    def __init__(self, mode: str, config: dict) -> None:
        self.config = config
        self.mode = mode
        self.is_training = (mode == "train")
        print("Dataloader Mode:", "training" if self.is_training else "testing")

        # configs
        self.data_root = config["data_root"]
        self.image_folder = config['image_folder']
        self.label_folder = config['label_folder']
        self.image_ext = config['image_ext']
        self.label_ext = config['label_ext']
        self.test_adjacent_length = config['test_adjacent_length']

        # transform
        if self.is_training:
            self.joint_transform = get_train_joint_transform(scale=config["scale"]) 
        else:
            self.joint_transform = get_val_joint_transform(scale=config["scale"])
        self.img_transform = get_img_transform()
        self.target_transform = get_target_transform()

        # get all frames from video datasets
        self.num_video_frame = 0
        self.frame_list = self.generate_images_from_video()
        print('Total video frames are {}.'.format(self.num_video_frame))

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, index):
        if self.is_training:
            # sample current frame from the video
            frame_path, gt_path, video_start_index, video_length = self.frame_list[index]
            
            # sample second frame from the same video
            query_index = np.random.randint(video_start_index, video_start_index + video_length)
            while query_index == index:
                query_index = np.random.randint(video_start_index, video_start_index + video_length)

            query_frame_path, query_gt_path, video_start_index_2, video_length_2 = self.frame_list[query_index]
            assert (video_start_index == video_start_index_2) and (video_length == video_length_2)

            # sample third frame from a different video
            while True:
                other_index = np.random.randint(0, self.__len__())
                if other_index < video_start_index or other_index > video_start_index + video_length - 1:
                    break
            other_frame_path, other_gt_path, video_start_index_3, _ = self.frame_list[other_index]
            assert video_start_index != video_start_index_3

            # read image and gt
            exemplar = Image.open(frame_path).convert('RGB')
            w, h = exemplar.size
            query = Image.open(query_frame_path).convert('RGB')
            query_w, query_h = query.size
            other = Image.open(other_frame_path).convert('RGB')
            other_w, other_h = other.size

            exemplar_gt = self.read_segmentation_mask(gt_path)
            query_gt = self.read_segmentation_mask(query_gt_path)
            other_gt = self.read_segmentation_mask(other_gt_path)

            # transformation
            manual_random = random.random()
            if self.joint_transform is not None:
                exemplar, exemplar_gt = self.joint_transform(exemplar, exemplar_gt, manual_random)
                query, query_gt = self.joint_transform(query, query_gt, manual_random)
                other, other_gt = self.joint_transform(other, other_gt)

            if self.img_transform is not None:
                exemplar = self.img_transform(exemplar)
                query = self.img_transform(query)
                other = self.img_transform(other)

            if self.target_transform is not None:
                exemplar_gt = self.target_transform(exemplar_gt)
                query_gt = self.target_transform(query_gt)
                other_gt = self.target_transform(other_gt)

            sample = {'exemplar': exemplar, 'exemplar_gt': exemplar_gt, "exemplar_path": frame_path, 'exemplar_gt_path': gt_path, 'exemplar_w': w, 'exemplar_h': h, 
                        'query': query, 'query_gt': query_gt, 'query_path': query_frame_path, 'query_gt_path': query_gt_path, 'query_w': query_w, 'query_h': query_h,
                        'other': other, 'other_gt': other_gt, 'other_path': other_frame_path, 'other_gt_path': other_gt_path, 'other_w': other_w, 'other_h': other_h,
                        }

            return sample
        else:
            manual_random = random.random()  # random for transformation

            # sample current frame from the video
            frame_path, gt_path, video_start_index, video_length = self.frame_list[index]
            exemplar = Image.open(frame_path).convert('RGB')
            w, h = exemplar.size
            exemplar_gt = self.read_segmentation_mask(gt_path)

            # sample adjacent query frame from the video
            adjacent_query_idx_list = self.get_adjacent_index(index, video_start_index, video_length, self.test_adjacent_length)
            
            adjacent_query_list = []
            adjacent_query_gt_list = []
            for query_idx in adjacent_query_idx_list:
                query_frame_path, query_gt_path, video_start_index_2, video_length_2 = self.frame_list[query_idx]
                assert (video_start_index == video_start_index_2) and (video_length == video_length_2)

                query = Image.open(query_frame_path).convert('RGB')
                adjacent_query_list.append(query)
                query_gt = self.read_segmentation_mask(query_gt_path)
                adjacent_query_gt_list.append(query_gt)
           
            # transformation
            manual_random = random.random()
            if self.joint_transform is not None:
                exemplar, exemplar_gt = self.joint_transform(exemplar, exemplar_gt, manual_random)
                for i in range(len(adjacent_query_list)):
                    adjacent_query_list[i], adjacent_query_gt_list[i] = self.joint_transform(adjacent_query_list[i], adjacent_query_gt_list[i], manual_random)
                
            if self.img_transform is not None:
                exemplar = self.img_transform(exemplar)
                for i in range(len(adjacent_query_list)):
                    adjacent_query_list[i] = self.img_transform(adjacent_query_list[i])

            if self.target_transform is not None:
                exemplar_gt = self.target_transform(exemplar_gt)
                for i in range(len(adjacent_query_list)):
                    adjacent_query_gt_list[i] = self.target_transform(adjacent_query_gt_list[i])
                
            sample = {'exemplar': exemplar, 'exemplar_gt': exemplar_gt, "exemplar_path": frame_path, 'exemplar_gt_path': gt_path, 'exemplar_w': w, 'exemplar_h': h, 'adjacent_length': self.test_adjacent_length, 'video_length': video_length}

            for i in range(len(adjacent_query_list)):
                sample["query_{}".format(str(i))] = adjacent_query_list[i]
                sample["query_gt_{}".format(str(i))] = adjacent_query_gt_list[i]
            
            return sample


    def generate_images_from_video(self):
        image_list = []
        video_list = os.listdir(os.path.join(self.data_root, self.mode, self.image_folder))
        # video_list = video_list if self.is_training else video_list[:10]
        for video in video_list:
            video_path = os.path.join(self.data_root, self.mode, self.image_folder, video)
            frame_list = [os.path.splitext(frame)[0] for frame in os.listdir(video_path) if frame.endswith(self.image_ext)]
            frame_list = self.sort_images(frame_list)
            for frame in frame_list:
                # frame_gt_idx_lenght: (frame, gt, video start index, length)
                frame_gt_idx_lenght = (os.path.join(self.data_root, self.mode, self.image_folder, video, frame + self.image_ext),
                                    os.path.join(self.data_root, self.mode, self.label_folder, video, frame + self.label_ext), 
                                    self.num_video_frame, 
                                    len(frame_list))
                image_list.append(frame_gt_idx_lenght)
            self.num_video_frame += len(frame_list)
        return image_list

    def get_adjacent_index(self, current_index, start_index, video_length, adjacent_length):
        query_index_list = [min(max(f, start_index), start_index+video_length-1) 
            for f in range(current_index - adjacent_length // 2, current_index + 1 + adjacent_length // 2) if f != current_index]
        return query_index_list

    def sort_images(self, frame_list):
        frame_int_list = [int(frame) for frame in frame_list]
        # sort img to 001, 002, 003...
        sort_index = [i for i, _ in sorted(enumerate(frame_int_list), key=lambda x: x[1])]
        return [frame_list[i] for i in sort_index]

    def read_segmentation_mask(self, gt_path):
        gt_pil = Image.open(gt_path).convert('L')
        gt_np = np.array(gt_pil)

        # some gt are store in RGB, whose values are not [0, 255]
        if len(np.unique(gt_np)) != 2:
            gt_np[gt_np != 0] = 255

        return Image.fromarray(gt_np)
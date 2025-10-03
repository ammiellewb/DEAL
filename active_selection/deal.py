import torch
import os
from datasets.base_dataset import BaseDataset
from datasets.transforms import get_transform
from utils.misc import get_topk_idxs, get_subset_paths, get_select_remain_paths
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from active_selection.uc_criterion import entropy, least_confidence, margin_sampling


class DEALSelector:

    def __init__(self, dataset, img_size, strategy, hard_levels):
        self.dataset = dataset
        self.img_size = img_size

        # map strategy abbreviations to full names
        strategy_mapping = {
            'DS': 'diff_score',
            'DE': 'diff_entropy'
        }
        self.strategy = strategy_mapping.get(strategy, strategy)
        self.hard_levels = hard_levels

        self.softmax = torch.nn.Softmax2d()

    @torch.no_grad()
    def select_next_batch(self, model, trainset, select_num):
        model.cuda()
        model.eval()

        # subset: 参与样本选择
        # remset: 在 subset 选出之后，再补充到 remain path 中
        # 50/50 split
        subset_img_paths, subset_target_paths, remset_img_paths, remset_target_paths = get_subset_paths(
            trainset.unlabel_img_paths, trainset.unlabel_target_paths, sub_ratio=0.5,
        )
        print('Candidate pool, processed through DEAL (subset_img_paths): ', len(subset_img_paths))
        print('Held-aside pool, so no scoring (remset_img_paths):', len(remset_img_paths))
        unlabelset = BaseDataset(subset_img_paths, subset_target_paths)  # load 时已将 bg_idx 统一为 255
        unlabelset.transform = get_transform('test', base_size=self.img_size)

        selection_batch_size = 8  # batch size for active selection
        dataloader = DataLoader(unlabelset,
                                batch_size=selection_batch_size, shuffle=False,
                                pin_memory=False, num_workers=4)

        scores = []
        tbar = tqdm(dataloader, desc='\r')
        tbar.set_description(f'{self.strategy}')
        

        
        for i, sample in enumerate(tbar):
            img = sample['img'].cuda()
            try:
                model_output = model(img)
                
                # Handle different model outputs
                if isinstance(model_output, tuple) and len(model_output) == 2:
                    output, diff_map = model_output
                    diff_map = diff_map.detach().cpu().numpy().squeeze(1)  # remove C=1, B,H,W
                else:
                    output = model_output
                    # Generate a dummy difficulty map from predictions
                    with torch.no_grad():
                        probs = self.softmax(output)
                        # Use entropy as difficulty proxy
                        entropy_map = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)
                        diff_map = entropy_map.detach().cpu().numpy().squeeze(1)  # B,H,W
                        
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue

            try:
                if self.strategy == 'diff_score':
                    probs = self.softmax(output)  # B,C,H,W
                    probs = np.transpose(probs.detach().cpu().numpy(), (0, 2, 3, 1))  # B,H,W,C
                    batch_scores = self.batch_diff_score(probs, diff_map)
                    
                    # Check for valid scores
                    valid_scores = [s for s in batch_scores if not (np.isnan(s) or np.isinf(s))]
                    if len(valid_scores) == 0:
                        print(f"Batch {i}: All DS scores are NaN/inf, skipping...")
                        continue
                        
                    scores += valid_scores
                    if i % 20 == 0:  # Print every 20th batch to avoid spam
                        print(f"Batch {i} DS scores: {[f'{s:.4f}' for s in batch_scores[:3]]}...")
                        # Report individual image scores with filenames
                        for j, score in enumerate(batch_scores):
                            img_idx = i * selection_batch_size + j
                            if img_idx < len(subset_img_paths):
                                img_name = os.path.basename(subset_img_paths[img_idx])
                                print(f"  Image {img_idx} DS score: {score:.6f} - {img_name}")
                            else:
                                print(f"  Image {img_idx} DS score: {score:.6f} - [index out of range]")
                            
                elif self.strategy == 'diff_entropy':
                    batch_scores = self.batch_diff_entropy(diff_map, hard_levels=self.hard_levels)
                    
                    # Check for valid scores
                    valid_scores = [s for s in batch_scores if not (np.isnan(s) or np.isinf(s))]
                    if len(valid_scores) == 0:
                        print(f"Batch {i}: All DE scores are NaN/inf, skipping...")
                        continue
                        
                    scores += valid_scores
                    if i % 20 == 0:  # Print every 20th batch to avoid spam
                        print(f"Batch {i} DE scores: {[f'{s:.4f}' for s in batch_scores[:3]]}...")
                        # Report individual image scores with filenames
                        for j, score in enumerate(batch_scores):
                            img_idx = i * selection_batch_size + j
                            if img_idx < len(subset_img_paths):
                                img_name = os.path.basename(subset_img_paths[img_idx])
                                print(f"  Image {img_idx} DE score: {score:.6f} - {img_name}")
                            else:
                                print(f"  Image {img_idx} DE score: {score:.6f} - [index out of range]")
                        
                            
            except Exception as e:
                print(f"Error calculating scores for batch {i}: {e}")
                continue

        print(f"\nTotal scores collected: {len(scores)}")
        
        # DEBUGGING
        if len(scores) == 0:
            print("\nERROR: No scores were collected! All batches failed or no valid scores generated.")
            print("This means either:")
            print("1. All model forward passes failed")
            print("2. All score calculations returned invalid values")
            print("3. Dataloader was empty or all batches were skipped")
            return

        print(f"Score range: [{min(scores):.6f}, {max(scores):.6f}]")
        print(f"Score statistics: mean={np.mean(scores):.6f}, std={np.std(scores):.6f}")
        print(f"Selecting top {select_num} from {len(scores)} images")
        
        select_idxs = get_topk_idxs(scores, select_num)
        print(f"Selected indices: {select_idxs}")

        # 从 subset 中选出样本
        select_img_paths, select_target_paths, remain_img_paths, remain_target_paths = get_select_remain_paths(
            subset_img_paths, subset_target_paths, select_idxs
        )
        # remain set 补充回去
        remain_img_paths += remset_img_paths
        remain_target_paths += remset_target_paths
        print('Selected images for labeling (select_img_paths): ', len(select_img_paths))
        print('Remaining unlabeled images (remain_img_paths): ', len(remain_img_paths))

        # 更新 DL, DU
        trainset.add_by_select_remain_paths(select_img_paths, select_target_paths,
                                            remain_img_paths, remain_target_paths)

    def batch_diff_score(self, probs, diff_maps, uc_criterion='none'):
        batch_scores = []
        for i in range(len(probs)):
            if uc_criterion == 'en':
                uc_map = entropy(probs[i])
            elif uc_criterion == 'ms':
                uc_map = margin_sampling(probs[i])
            elif uc_criterion == 'lc':
                uc_map = least_confidence(probs[i])
            elif uc_criterion == 'none':
                uc_map = 1.
            else:
                raise NotImplementedError
            batch_scores.append(np.mean(uc_map * diff_maps[i]))

        return batch_scores

    def batch_diff_entropy(self, diff_map, hard_levels):
        batch_scores = []
        for i in range(len(diff_map)):
            region_areas, score_ticks = np.histogram(diff_map[i], bins=hard_levels)
            probs = region_areas / region_areas.sum()
            entropy = -np.nansum(np.multiply(probs, np.log(probs + 1e-12)))
            batch_scores.append(entropy)

        return batch_scores

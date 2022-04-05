import bisect
import random

import numpy as np

from utils.umap_utils import find_ab_params, convert_distance_to_probability

MUST_LINK = 1
CANNOT_LINK = 0
SPREAD = 1
UN_SPREAD = 0
ACTIVE = 1
INACTIVE = 0


class LinkInfo:
    def __init__(self, links, link_spreads, finetune_epochs, pre_embeddings, min_dist):
        self.links = links
        self.link_uuid = links[:, 0]
        self.link_indices = links[:, 1:3]
        self.link_types = links[:, 3]
        self.link_activate = links[:, 4]
        self.link_spreads = link_spreads
        self.link_num = links.shape[0]
        self.new_link_num = self.link_num

        self.min_dist = min_dist
        self._a, self._b = find_ab_params(1.0, min_dist)

        self.finetune_epochs = finetune_epochs

        self.all_link_indices = None

        self.crs_indices = None
        self.crs_sims = None
        self.new_crs_indices = None
        self.new_crs_sims = None
        self.new_link_spreads = None

        self.iter_count = 0
        self.weaken_rate = 0.5

        self.construct_csr(pre_embeddings, links, link_spreads)

    def process_cur_links(self, links, link_spreads, pre_embeddings):
        link_uuid = links[:, 0]
        link_activates = links[:, 4]
        pre_link_uuid = self.link_uuid

        added_link_indices = []
        preserve_link_indices = []
        preserved_new_link_activate = []
        list_link_uuid = list(link_uuid)
        for i, uuid in enumerate(pre_link_uuid):
            idx = list_link_uuid.index(uuid) if uuid in list_link_uuid else -1
            if idx >= 0:
                preserve_link_indices.append(i)
                preserved_new_link_activate.append(link_activates[idx])

        for i, uuid in enumerate(link_uuid):
            if uuid not in pre_link_uuid:
                added_link_indices.append(i)

        preserve_link_uuid = pre_link_uuid

        added_links = links[added_link_indices] if len(added_link_indices) > 0 else []
        added_link_spreads = link_spreads[added_link_indices] if len(added_link_indices) > 0 else []

        self._preserve_old_links(preserve_link_uuid, preserve_link_indices)
        self._link_activate_change(preserved_new_link_activate)
        self.add_new_links(added_links, added_link_spreads, pre_embeddings)

    def add_new_links(self, added_links, added_link_spreads, pre_embeddings):
        self.new_link_num = len(added_link_spreads)
        if self.new_link_num <= 0:
            return
        self.link_num += self.new_link_num
        self.links = np.concatenate([self.links, added_links], axis=0)
        self.link_uuid = np.concatenate([self.link_uuid, added_links[:, 0]], axis=0)
        self.link_indices = np.concatenate([self.link_indices, added_links[:, 1:3]], axis=0)
        self.link_types = np.concatenate([self.link_types, added_links[:, 3]], axis=0)
        self.link_activate = np.concatenate([self.link_activate, added_links[:, 4]], axis=0)
        self.link_spreads = np.concatenate([self.link_spreads, added_link_spreads], axis=0)
        self.construct_csr(pre_embeddings, added_links, added_link_spreads)

    def _preserve_old_links(self, preserve_link_uuid, preserve_link_indices):
        self.link_num = len(preserve_link_indices)
        self.link_uuid = self.link_uuid[preserve_link_indices]
        self.link_indices = self.link_indices[preserve_link_indices]
        self.link_spreads = self.link_spreads[preserve_link_indices]
        self.link_types = self.link_types[preserve_link_indices]
        self.link_activate = self.link_activate[preserve_link_indices]

        reserved_crs_indices = []
        for i in range(len(self.crs_indices)):
            if self.crs_indices[i, 0] in preserve_link_uuid:
                reserved_crs_indices.append(i)

        self.crs_indices = self.crs_indices[reserved_crs_indices]
        self.crs_sims = self.crs_sims[reserved_crs_indices]

    def _link_activate_change(self, new_link_activate):

        weaken_indices = []
        zero_set_indices = []
        one_set_indices = []
        for i in range(self.link_num):
            pre_stat = self.link_activate[i]
            cur_stat = new_link_activate[i]
            if pre_stat == cur_stat:
                if pre_stat == ACTIVE:
                    weaken_indices.append(i)
                else:
                    zero_set_indices.append(i)
            else:
                if pre_stat == ACTIVE:
                    zero_set_indices.append(i)
                else:
                    one_set_indices.append(i)

        weaken_link_ids = self.link_uuid[weaken_indices] if len(weaken_indices) > 0 else []
        zero_set_link_ids = self.link_uuid[zero_set_indices] if len(zero_set_indices) > 0 else []
        one_set_link_ids = self.link_uuid[one_set_indices] if len(one_set_indices) > 0 else []
        self.weaken_old_links(weaken_link_ids, zero_set_link_ids, one_set_link_ids)

    def weaken_old_links(self, weaken_link_ids, zero_set_link_ids, one_set_link_ids):
        for i in range(self.crs_indices.shape[0]):
            link_uuid = self.crs_indices[i][0]

            if link_uuid in weaken_link_ids:
                pre_link_weight = self.crs_indices[i][-1]
                new_link_weight = pre_link_weight * self.weaken_rate
                if new_link_weight <= 0.05:
                    pass
            elif link_uuid in zero_set_link_ids:
                new_link_weight = 0
            else:
                new_link_weight = 1

            self.crs_indices[i][-1] = new_link_weight

    def construct_csr(self, embeddings, cur_new_links, cur_new_link_spreads):

        link_uuid = cur_new_links[:, 0]
        link_indices = cur_new_links[:, 1:3]
        link_types = cur_new_links[:, 3]

        self.new_link_spreads = np.array(cur_new_link_spreads, dtype=np.int)
        link_num = len(cur_new_link_spreads)
        cur_total_link_num = np.sum(cur_new_link_spreads) + np.sum(~np.array(cur_new_link_spreads, dtype=np.bool))
        if link_num <= 0:
            return

        crs_indices = np.ones((cur_total_link_num, 5), dtype=np.float)
        crs_sims = np.ones((cur_total_link_num, 5))

        count = 0
        for i in range(link_num):
            h_idx, t_idx = link_indices[i]
            uuid = link_uuid[i]
            cur_type = link_types[i]
            crs_indices[count] = [uuid, h_idx, t_idx, cur_type, 2]
            crs_sims[count] = [uuid, 1, 1, cur_type, 2]
            count += 1

        self.new_crs_indices = crs_indices
        self.new_crs_sims = crs_sims
        if self.crs_indices is None:
            self.crs_indices = crs_indices
            self.crs_sims = crs_sims
        else:
            self.crs_indices = np.concatenate([self.crs_indices, crs_indices], axis=0)
            self.crs_sims = np.concatenate([self.crs_sims, crs_sims], axis=0)


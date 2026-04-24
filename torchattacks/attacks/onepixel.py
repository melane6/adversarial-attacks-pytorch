import numpy as np
import logging

import torch
import torch.nn.functional as F

from ..attack import Attack
from ._differential_evolution import differential_evolution

logger = logging.getLogger(__name__)


class OnePixel(Attack):
    r"""
    Attack in the paper 'One pixel attack for fooling deep neural networks'
    [https://arxiv.org/abs/1710.08864]

    Modified from "https://github.com/DebangLi/one-pixel-attack-pytorch/" and 
    "https://github.com/sarathknv/adversarial-examples-pytorch/blob/master/one_pixel_attack/"

    Distance Measure : L0

    Arguments:
        model (nn.Module): model to attack.
        pixels (int): number of pixels to change (Default: 1)
        steps (int): number of steps. (Default: 10)
        popsize (int): population size, i.e. the number of candidate agents or "parents" in differential evolution (Default: 10)
        inf_batch (int): maximum batch size during inference (Default: 128)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.OnePixel(model, pixels=1, steps=10, popsize=10, inf_batch=128)
        >>> adv_images = attack(images, labels)

        >>> # With boolean mask to restrict attack to certain areas
        >>> mask = torch.zeros(224, 224, dtype=torch.bool)
        >>> mask[50:150, 60:160] = True  # Only attack this region
        >>> adv_images = attack(images, labels, mask=mask)

    """

    def __init__(self, model, pixels=1, steps=10, popsize=10, inf_batch=128):
        super().__init__("OnePixel", model)
        self.pixels = pixels
        self.steps = steps
        self.popsize = popsize
        self.inf_batch = inf_batch
        self.supported_mode = ["default", "targeted"]
        self.explanation = None
        self.responsibility = None
        self.mask_priority = None  # Priority weights for allowed coordinates

    def forward(self, images, labels, mask=None):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        batch_size, channel, height, width = images.shape

        allowed_coords = None
        priority_weights = None
        if mask is not None:
            # Handle two input formats: bool mask or (bool_mask, priority_weights)
            if isinstance(mask, (tuple, list)) and len(mask) == 2:
                mask_bool, mask_priority = mask
                mask = mask_bool
                logger.info(f"OnePixel: mask provided: Shape: {mask.shape}, dtype: {mask.dtype}")
                logger.info(f"OnePixel: mask priority provided: Shape: {mask_priority.shape}")
                if mask_priority is not None:
                    if isinstance(mask_priority, torch.Tensor):
                        mask_priority = mask_priority.detach().cpu().numpy()
                    priority_weights = mask_priority
                    logger.info(f"OnePixel: mask priority provided: Shape: {priority_weights.shape}")

            if isinstance(mask, torch.Tensor):
                mask = mask.to(torch.bool)
            else:
                mask = torch.tensor(mask, dtype=torch.bool)
            if mask.dtype != torch.bool:
                raise ValueError(f"Mask dtype {mask.dtype} must be torch.bool")

            mask_np = mask.detach().cpu().numpy()

            # make sure priority_weights has only values in the mask allowed region
            if priority_weights is not None:
                if priority_weights.shape != mask.shape:
                    raise ValueError(f"Priority weights shape {priority_weights.shape} must match mask shape {mask.shape}")
                priority_weights = np.where(mask_np, priority_weights, 0)

            allowed_coords = [np.argwhere(mask_np[i]) for i in range(batch_size)]
            logger.info("OnePixel: mask applied. Allowed pixels per image:")
            for i, coords in enumerate(allowed_coords):
                logger.info(f"  Image {i}: {coords.shape[0]} allowed pixels")
            # Check that each batch image has at least one allowed pixel
            for i, coords in enumerate(allowed_coords):
                if coords.shape[0] == 0:
                    raise ValueError(
                        f"Mask contains no True values for image {i}. Cannot attack empty region. "
                        f"Mask shape: {mask.shape}, non-zero count: {mask.sum().item()}"
                    )

            logger.info(
                "OnePixel: mask applied. Allowed pixels: %d / %d (%.2f%%)",
                allowed_coords[0].shape[0],
                height * width,
                100.0 * allowed_coords[0].shape[0] / (height * width),
            )
            self.mask_priority = priority_weights

        # BOUNDS setupo
        if allowed_coords is None:
            # (row, col) pairs - before for whole image
            bounds = [(0, height - 1), (0, width - 1)] + [(0, 1)] * channel
        else:
            # search over allowed coordinates
            num_allowed = allowed_coords[0].shape[0]
            bounds = [(0, num_allowed - 1)] + [(0, 1)] * channel

        bounds = bounds * self.pixels

        popmul = max(1, int(self.popsize / len(bounds)))

        logger.info(
            "OnePixel: batch_size=%d pixels=%d steps=%d popsize=%d inf_batch=%d bounds=%d",
            batch_size,
            self.pixels,
            self.steps,
            self.popsize,
            self.inf_batch,
            len(bounds),
        )

        adv_images = []
        for idx in range(batch_size):
            image, label = images[idx: idx + 1], labels[idx: idx + 1]

            image_allowed_coords = allowed_coords[idx] if allowed_coords is not None else None

            if self.targeted:
                target_label = target_labels[idx: idx + 1]
                gen = [0]

                def func(delta):
                    return self._loss(image, target_label, delta, image_allowed_coords)

                def callback(delta, convergence, image_idx=idx, gen_ref=gen):
                    gen_ref[0] += 1
                    logger.debug(
                        "OnePixel: image %d/%d generation %d convergence=%.6f",
                        image_idx + 1,
                        batch_size,
                        gen_ref[0],
                        convergence,
                    )
                    success = self._attack_success(image, target_label, delta, image_allowed_coords)
                    if success:
                        logger.debug(
                            "OnePixel: attack success for image %d at generation %d",
                            image_idx + 1,
                            gen_ref[0],
                        )
                    return success

            else:
                gen = [0]

                def func(delta):
                    return self._loss(image, label, delta, image_allowed_coords)

                def callback(delta, convergence, image_idx=idx, gen_ref=gen):
                    gen_ref[0] += 1
                    logger.debug(
                        "OnePixel: image %d/%d generation %d convergence=%.6f",
                        image_idx + 1,
                        batch_size,
                        gen_ref[0],
                        convergence,
                    )
                    success = self._attack_success(image, label, delta, image_allowed_coords)
                    if success:
                        logger.debug(
                            "OnePixel: attack success for image %d at generation %d",
                            image_idx + 1,
                            gen_ref[0],
                        )
                    return success

            delta = differential_evolution(
                func=func,
                bounds=bounds,
                callback=callback,
                maxiter=self.steps,
                popsize=popmul,
                init="random",
                recombination=1,
                atol=-1,
                polish=False,
            ).x
            delta = np.split(delta, len(delta) // len(bounds))
            adv_image = self._perturb(image, delta, image_allowed_coords)
            adv_images.append(adv_image)

        adv_images = torch.cat(adv_images)
        return adv_images

    def _loss(self, image, label, delta, allowed_coords=None):
        adv_images = self._perturb(image, delta, allowed_coords)
        prob = self._get_prob(adv_images)[:, label]
        if self.targeted:
            return 1 - prob  # If targeted, increase prob
        else:
            return prob  # If non-targeted, decrease prob

    def _attack_success(self, image, label, delta, allowed_coords=None):
        adv_image = self._perturb(image, delta, allowed_coords)
        prob = self._get_prob(adv_image)
        pre = np.argmax(prob)
        if self.targeted and (pre == label):
            return True
        elif (not self.targeted) and (pre != label):
            return True
        return False

    def _get_prob(self, images):
        with torch.no_grad():
            batches = torch.split(images, self.inf_batch)
            outs = []
            for batch in batches:
                out = self.get_logits(batch)
                outs.append(out)
        outs = torch.cat(outs)
        prob = F.softmax(outs, dim=1)
        return prob.detach().cpu().numpy()

    def _weight_index_mapping(self, base_idx, allowed_coords, priority_weights):
        """
        Map continuous DE index to discrete indices using priority weights.
        Pixels with higher priority are more likely to be selected.
        Uses inverse-transform sampling on normalized weights.

        Args:
            base_idx: Continuous index from DE [0, len(allowed_coords)-1]
            allowed_coords: Array of (row, col) coordinates
            priority_weights: Array of priority weights for each coordinate

        Returns:
            Weighted index into allowed_coords based on priority
        """
        num_coords = len(allowed_coords)
        # Normalize weights to [0, 1] range
        w = np.asarray(priority_weights).flatten()[:num_coords]
        if w.size == 0:
            return base_idx
        w_sum = np.sum(w)
        if w_sum <= 0:
            return base_idx  # Fallback: all weights zero
        w = w / w_sum  # Normalize to probability distribution
        cumulsum = np.cumsum(w)
        prob_val = (base_idx + 0.5) / num_coords
        # Find which bucket this falls into
        idx = np.searchsorted(cumulsum, prob_val, side='right')
        return min(idx, num_coords - 1)

    def _perturb(self, image, delta, allowed_coords=None):
        delta = np.array(delta)
        if len(delta.shape) < 2:
            delta = np.array([delta])
        num_delta = len(delta)
        adv_image = image.clone().detach().to(self.device)
        adv_images = torch.cat([adv_image] * num_delta, dim=0)

        for idx in range(num_delta):
            pixel_info = delta[idx].reshape(self.pixels, -1)
            for pixel in pixel_info:
                if allowed_coords is None:
                    pos_x, pos_y = pixel[:2]
                    channel_v = pixel[2:]
                    row, col = int(pos_x), int(pos_y)
                else:
                    # indexed pixel[0] - is the coordinate index
                    # With priority weights, map DE continuous value to discrete indices
                    coord_idx = int(np.clip(np.rint(pixel[0]), 0, len(allowed_coords) - 1))
                    # If priority weights exist, map continuous index to weighted indices
                    if self.mask_priority is not None:
                        coord_idx = self._weight_index_mapping(coord_idx, allowed_coords, self.mask_priority)
                    row, col = allowed_coords[coord_idx]
                    channel_v = pixel[1:]

                for channel, v in enumerate(channel_v):
                    adv_images[idx, channel, row, col] = v
        return adv_images

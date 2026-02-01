"""
A patch to fix the forward propagation module for **stable multimodal multitask training with deepspeed zero3**.
The official implementation is going to get stuck while GPU utilization is always 100% when we run multimodal multitask training with deepspeed zero3.
It is because the forward graphs are not fully consistent across different gpus. (reference: https://blog.csdn.net/qq_27061325/article/details/148960693?spm=1001.2014.3001.5501)
One direct solution is to align all forward pass with fake tensors if certain inputs are abscent.
Note that this is not necessary for inference and modality-consistent training.

This script is provided by Changxu Cheng @ Uni-Ubi.
"""

import torch
from typing import Optional, Union

from transformers.utils import is_torchdynamo_compiling
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast


def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen3VLModelOutputWithPast]:
    r"""
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None

    pixel_dim = \
        self.visual.patch_embed.in_channels * self.visual.patch_embed.temporal_patch_size * self.visual.patch_embed.patch_size ** 2
    llm_dim = inputs_embeds.shape[-1]

    #### Code Patch Start ####
    if pixel_values is not None:
        is_fake_image = False
    else:
        pixel_values = torch.ones(160, pixel_dim, device=inputs_embeds.device)
        image_grid_thw = torch.tensor([[1, 10, 16]], device=inputs_embeds.device)
        is_fake_image = True
    image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
    image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    if is_fake_image:
        image_mask = torch.zeros_like(inputs_embeds).bool()
    else:
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
    image_embeds = image_embeds[:image_mask.sum() // llm_dim]
    deepstack_image_embeds = [die[:image_mask.sum() // llm_dim] for die in deepstack_image_embeds]
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        is_fake_video = False
    else:
        pixel_values_videos = torch.ones(96, pixel_dim, device=inputs_embeds.device)
        video_grid_thw = torch.tensor([[4, 4, 6]], device=inputs_embeds.device)
        is_fake_video = True
    video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
    video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    if is_fake_video:
        video_mask = torch.zeros_like(inputs_embeds).bool()
    else:
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
    video_embeds = video_embeds[:video_mask.sum() // llm_dim]
    deepstack_video_embeds = [die[:video_mask.sum() // llm_dim] for die in deepstack_video_embeds]
    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
    #### Code Patch End ####

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        # aggregate visual_pos_masks and deepstack_visual_embeds
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds

    if position_ids is None:
        attention_mask_tensor = (
            attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            # Only apply conversion for floating point tensors (inverted masks)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )

    return Qwen3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
    )

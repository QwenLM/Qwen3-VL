import importlib
import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

import torch


# These are CPU contract tests for the adapter. The CUDA kernel is replaced
# at its import boundary; all Qwen and Transformers module code is imported.
if importlib.util.find_spec('flash_attn') is None:
    interface = types.ModuleType('flash_attn.flash_attn_interface')
    interface.flash_attn_varlen_func = None
    sys.modules['flash_attn.flash_attn_interface'] = interface
    try:
        trainer = importlib.import_module('qwenvl.train.trainer')
    finally:
        del sys.modules['flash_attn.flash_attn_interface']
else:
    trainer = importlib.import_module('qwenvl.train.trainer')


def attention_oracle(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                     dropout_p=0., softmax_scale=None, window_size=(-1, -1), softcap=0., causal=False):
    outputs = []
    for left, right in zip(cu_seqlens_q[:-1].tolist(), cu_seqlens_q[1:].tolist()):
        qq, kk, vv = q[left:right].float().transpose(0, 1), k[left:right].float().transpose(0, 1), v[left:right].float().transpose(0, 1)
        scale = qq.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
        scores = torch.matmul(qq, kk.transpose(-1, -2)) * scale
        if softcap:
            scores = softcap * torch.tanh(scores / softcap)
        indices = torch.arange(right - left)
        valid = indices[None, :] <= indices[:, None] if causal else torch.ones_like(scores[0], dtype=torch.bool)
        if window_size[0] >= 0:
            valid &= indices[None, :] >= indices[:, None] - window_size[0]
        scores = scores.masked_fill(~valid, -torch.inf)
        weights = torch.nn.functional.dropout(scores.softmax(-1), p=dropout_p, training=dropout_p > 0)
        outputs.append(torch.matmul(weights, vv).transpose(0, 1).to(q.dtype))
    return torch.cat(outputs)


class PackedAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.q, self.k, self.v = [torch.randn(1, 2, 7, 4) for _ in range(3)]
        self.cu = torch.tensor([0, 3, 7], dtype=torch.int32)
        self.module = torch.nn.Linear(4, 4).to(dtype=torch.bfloat16)
        self.module.config = types.SimpleNamespace()

    def test_upcast_inputs_are_restored_to_projection_dtype(self):
        with patch.object(trainer, 'flash_attn_varlen_func', side_effect=attention_oracle) as kernel:
            result, weights = trainer.flash_attention_forward(self.module, self.q, self.k, self.v, self.cu)
        for tensor in kernel.call_args.args[:3]:
            self.assertEqual(tensor.dtype, torch.bfloat16)
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertEqual(result.shape, (1, 7, 2, 4))
        self.assertIsNone(weights)

    def test_quantized_dtype_takes_precedence(self):
        self.module.config._pre_quantization_dtype = torch.float16
        with patch.object(trainer, 'flash_attn_varlen_func', side_effect=attention_oracle) as kernel:
            trainer.flash_attention_forward(self.module, self.q, self.k, self.v, self.cu)
        self.assertEqual(kernel.call_args.args[0].dtype, torch.float16)

    def test_scaling_window_and_softcap_match_causal_segment_attention(self):
        for window in (None, 2):
            with self.subTest(window=window):
                kwargs = dict(scaling=0.8, sliding_window=window, softcap=0.4)
                q, k, v = [x.to(torch.bfloat16).squeeze(0).transpose(0, 1) for x in (self.q, self.k, self.v)]
                expected = attention_oracle(q, k, v, self.cu, self.cu, 4, 4,
                    softmax_scale=0.8, window_size=(window - 1, 0) if window else (-1, -1), softcap=0.4, causal=True)
                with patch.object(trainer, 'flash_attn_varlen_func', side_effect=attention_oracle):
                    actual, _ = trainer.flash_attention_forward(self.module, self.q, self.k, self.v, self.cu, **kwargs)
                torch.testing.assert_close(actual.squeeze(0), expected)

    def test_training_dropout_reaches_the_kernel_and_eval_can_disable_it(self):
        for dropout in (0., 0.2):
            with self.subTest(dropout=dropout):
                with patch.object(trainer, 'flash_attn_varlen_func', side_effect=attention_oracle) as kernel:
                    trainer.flash_attention_forward(self.module, self.q, self.k, self.v, self.cu, dropout=dropout)
                self.assertEqual(kernel.call_args.kwargs.get('dropout_p', 0.), dropout)

    def test_packed_segments_remain_isolated(self):
        with patch.object(trainer, 'flash_attn_varlen_func', side_effect=attention_oracle):
            before, _ = trainer.flash_attention_forward(self.module, self.q, self.k, self.v, self.cu)
            changed = self.v.clone()
            changed[:, :, 3:] += 100.
            after, _ = trainer.flash_attention_forward(self.module, self.q, self.k, changed, self.cu)
        torch.testing.assert_close(before[:, :3], after[:, :3])


if __name__ == '__main__':
    unittest.main()

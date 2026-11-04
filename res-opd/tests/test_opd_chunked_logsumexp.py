import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from verl.utils.torch_functional import logsumexp_over_vocab_in_token_chunks


@unittest.skipIf(torch is None, "torch is not installed")
class ChunkedLogsumexpTest(unittest.TestCase):
    def test_values_and_gradients_match(self):
        source = torch.randn(2, 9, 17, dtype=torch.float32)
        full_logits = source.clone().requires_grad_()
        chunked_logits = source.clone().requires_grad_()

        expected = torch.logsumexp(full_logits, dim=-1, keepdim=True)
        actual = logsumexp_over_vocab_in_token_chunks(chunked_logits, max_chunk_bytes=2 * 17 * 4 * 3)

        self.assertEqual(actual.shape, expected.shape)
        torch.testing.assert_close(actual, expected)
        expected.sum().backward()
        actual.sum().backward()
        torch.testing.assert_close(chunked_logits.grad, full_logits.grad)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            logsumexp_over_vocab_in_token_chunks(torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            logsumexp_over_vocab_in_token_chunks(torch.zeros(2, 3, 4), max_chunk_bytes=0)


if __name__ == "__main__":
    unittest.main()
